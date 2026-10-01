"""
===============================================================================
HuMob 2026: Step 4 - Train Per-Origin Destination Map Flow Matching Model
===============================================================================
Flow Matching 訓練腳本。
Loss = E_t [ ||v_θ(x_t, t, c) - (x_0 - ε)||² ]
其中 x_t = (1-t)*ε + t*x_0，t ~ U(0,1)
===============================================================================
"""
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("[警告] tqdm 未安裝，將使用簡易進度。可用 pip install tqdm 安裝。", flush=True)

sys.stdout.reconfigure(encoding='utf-8')

# 防止 Windows 在訓練期間進入睡眠
try:
    import ctypes
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
except Exception:
    pass

PIPELINE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT  = PIPELINE_ROOT.parent

sys.path.insert(0, str(PIPELINE_ROOT / 'src'))

from origin_flow_matching import OriginDestFlowUNet, OriginFlowMatching

NPZ_PATH   = PROJECT_ROOT / 'data' / 'outputs' / 'origin_fm_dataset.npz'
CHECKPOINT = PROJECT_ROOT / 'data' / 'outputs' / 'origin_fm_checkpoint_ep5.pt'

# ── 超參數 ─────────────────────────────────────────────────────────────────────
BATCH_SIZE = 256   # 64→256：steps/epoch 縮 4 倍，攤薄 random data access 開銷
EPOCHS     = 5     # 肘點早停 (Elbow point at epoch 5)
LR         = 3e-4
BASE_CH    = 32
TIME_DIM   = 128
COND_DIM   = 8
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

print("=" * 75, flush=True)
print("[Step 4] Train Destination Map Flow Matching Model", flush=True)
print(f"Device: {DEVICE}", flush=True)
if DEVICE == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB", flush=True)
print("=" * 75, flush=True)

# ── 載入資料 ──────────────────────────────────────────────────────────────────
t0 = time.time()
print(f"正在載入資料集: {NPZ_PATH} ...", flush=True)
data = np.load(str(NPZ_PATH))
sample_z    = torch.from_numpy(data['sample_z'])      # (N, 1, 70, 100) float32
sample_cond = torch.from_numpy(data['sample_cond'])   # (N, 8) float32
N_SAMPLES   = len(sample_z)

print(f"✅ 資料載入完成 ({time.time()-t0:.1f}s)", flush=True)
print(f"   總樣本數: {N_SAMPLES:,}  |  形狀: {list(sample_z.shape)}", flush=True)
print(f"   條件維度: {list(sample_cond.shape)}", flush=True)
print(f"   Batch Size: {BATCH_SIZE}  |  Steps/Epoch: {N_SAMPLES // BATCH_SIZE + 1:,}", flush=True)

dataset = TensorDataset(sample_z, sample_cond)
loader  = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(DEVICE == 'cuda'),
    drop_last=False,
)

# ── 建立模型 ──────────────────────────────────────────────────────────────────
model = OriginDestFlowUNet(
    in_channels=1,
    base_ch=BASE_CH,
    time_dim=TIME_DIM,
    cond_dim=COND_DIM,
).to(DEVICE)

fm = OriginFlowMatching(model, device=DEVICE)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✅ 模型建立完成! 參數量: {n_params:,}", flush=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

# ── 訓練迴圈 ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 75, flush=True)
print(f"開始訓練 Flow Matching  (共 {EPOCHS} epochs, device={DEVICE})", flush=True)
print("=" * 75, flush=True)

best_loss = float('inf')
t_train_start = time.time()
total_steps = 0
step_loss_history = []
epoch_loss_history = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    t_epoch = time.time()

    # 進度條（若有 tqdm 則用，無則降級為文字印出）
    if HAS_TQDM:
        pbar = tqdm(
            loader,
            desc=f"Epoch [{epoch:2d}/{EPOCHS}]",
            dynamic_ncols=True,
            unit="batch",
            leave=True,
        )
    else:
        pbar = loader

    for b_idx, (x1, c) in enumerate(pbar):
        x1 = x1.to(DEVICE, non_blocking=True)
        c  = c.to(DEVICE, non_blocking=True)

        loss = fm.training_step(x1, c)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        b_loss = loss.item()
        total_loss += b_loss * len(x1)
        total_steps += 1

        # 記錄每 step loss (降採樣儲存，每 5 steps 記一次避免 log 過膨脹)
        if total_steps % 5 == 0:
            step_loss_history.append({'step': total_steps, 'loss': float(b_loss)})

        if HAS_TQDM:
            pbar.set_postfix({
                'loss': f"{b_loss:.5f}",
                'lr':   f"{optimizer.param_groups[0]['lr']:.1e}",
            })
        elif (b_idx + 1) % 100 == 0:
            print(f"  ep {epoch}/{EPOCHS} | batch {b_idx+1:4d}/{len(loader)} | loss={b_loss:.5f}", flush=True)

    # ── epoch 結束：印出摘要 + ETA ────────────────────────────────────────────
    scheduler.step()
    avg_loss  = total_loss / N_SAMPLES
    epoch_loss_history.append({'epoch': epoch, 'loss': float(avg_loss)})
    epoch_sec = time.time() - t_epoch
    remaining = epoch_sec * (EPOCHS - epoch)   # 以本 epoch 速度估算剩餘時間
    lr_cur    = optimizer.param_groups[0]['lr']

    saved_mark = ""
    if avg_loss < best_loss:
        best_loss = avg_loss
        CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        ckpt_data = {
            'epoch':     epoch,
            'model':     model.state_dict(),
            'best_loss': best_loss,
            'n_params':  n_params,
            'cond_dim':  COND_DIM,
            'base_ch':   BASE_CH,
            'time_dim':  TIME_DIM,
        }
        torch.save(ckpt_data, str(CHECKPOINT))
        # 同步另存該 epoch 專屬權重 (讓 ep4 與 ep5 都能自由選用)
        ep_specific_ckpt = CHECKPOINT.parent / f"origin_fm_checkpoint_ep{epoch}.pt"
        torch.save(ckpt_data, str(ep_specific_ckpt))
        saved_mark = f" [saved ep{epoch}]"

    rem_h = int(remaining // 3600)
    rem_m = int((remaining % 3600) // 60)
    print(
        f"Epoch [{epoch:3d}/{EPOCHS}] "
        f"loss={avg_loss:.6f} (best={best_loss:.6f}){saved_mark} | "
        f"LR={lr_cur:.1e} | "
        f"{epoch_sec:.0f}s/ep | "
        f"ETA {rem_h}h{rem_m:02d}m",
        flush=True
    )

print("=" * 75, flush=True)
print(f"✅ Training Done! Best Loss: {best_loss:.6f}", flush=True)
print(f"✅ Checkpoint → {CHECKPOINT}", flush=True)

# ── 自動繪製並儲存 Loss vs Steps / Epochs 曲線圖 ───────────────────────
import json
import pandas as pd
import matplotlib.pyplot as plt

history_json = PROJECT_ROOT / 'data' / 'outputs' / 'loss_history_ep5.json'
loss_png     = PROJECT_ROOT / 'data' / 'outputs' / 'loss_step_curve_ep5.png'

with open(history_json, 'w', encoding='utf-8') as f:
    json.dump({'step_loss': step_loss_history, 'epoch_loss': epoch_loss_history}, f, indent=2)
print(f"✅ Loss 數據日誌 → {history_json}")

plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=150)
fig.patch.set_facecolor('#0f172a')

# 子圖 1: Loss vs Steps
steps = [x['step'] for x in step_loss_history]
losses = [x['loss'] for x in step_loss_history]
ax1.set_facecolor('#1e293b')
ax1.grid(True, color='#334155', linestyle='--', alpha=0.5)
ax1.plot(steps, losses, color='#10b981', alpha=0.35, linewidth=1.0, label='Raw Batch Loss')

if len(losses) > 10:
    ema = pd.Series(losses).ewm(span=30).mean().values
    ax1.plot(steps, ema, color='#34d399', linewidth=2.2, label='EMA Smoothed (span=30)')

ax1.set_title("Flow Matching Loss vs Steps", color='#f8fafc', fontsize=13, fontweight='bold')
ax1.set_xlabel("Training Steps", color='#cbd5e1')
ax1.set_ylabel("CNF Vector Field MSE Loss", color='#cbd5e1')
ax1.legend(facecolor='#1e293b', edgecolor='#475569')

# 子圖 2: Loss vs Epochs
eps = [x['epoch'] for x in epoch_loss_history]
ep_losses = [x['loss'] for x in epoch_loss_history]
ax2.set_facecolor('#1e293b')
ax2.grid(True, color='#334155', linestyle='--', alpha=0.5)
ax2.plot(eps, ep_losses, color='#f43f5e', marker='o', linewidth=2.2, markersize=6, label='Epoch Avg Loss')
ax2.set_title("Flow Matching Loss vs Epochs", color='#f8fafc', fontsize=13, fontweight='bold')
ax2.set_xlabel("Epoch", color='#cbd5e1')
ax2.set_ylabel("Average Loss", color='#cbd5e1')
ax2.legend(facecolor='#1e293b', edgecolor='#475569')

plt.tight_layout()
plt.savefig(loss_png, facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()
print(f"✅ Loss Steps 曲線圖已繪製 → {loss_png}")
print("=" * 75, flush=True)
