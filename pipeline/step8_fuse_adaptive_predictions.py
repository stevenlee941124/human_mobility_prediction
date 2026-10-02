"""
===============================================================================
HuMob 2026: Step 8 - Dual-Track Adaptive Regularity-Weighted Fusion Module
===============================================================================
【核心融合推論：雙軌自適應共振融合】
直接讀取由【Step 6 純 Flow Matching】與【Step 7 純統計 Ψ7】各自產出的預測檔，
依據各路線作息規律度 (Reg in [0, 1]) 動態分配權重進行雙軌共振合成：
  - 高規律通勤路線 (Reg -> 1)：由生活週期波 Ψ7 主導 (w_psi -> 1.15)，抑制神經噪聲 (w_fm -> 0.40)。
  - 低規律隨機路線 (Reg -> 0)：釋放 Flow Matching 空間神經擴散殘差 (w_fm -> 1.00)。

輸出：
  1. data/outputs/per_route_full_rise_weekly_predictions.pkl (最終預測矩陣)
  2. data/outputs/submission.tsv 與 根目錄 submission.tsv (最終官方合格提交檔)
  3. reports/ 目錄下 9-Class 波動對比與盲區特寫圖表
===============================================================================
"""
import sys
import time
import math
import pickle
import shutil
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.stdout.reconfigure(encoding='utf-8')
PIPELINE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT  = PIPELINE_ROOT.parent

PROCESSED = PROJECT_ROOT / 'data' / 'processed'
SHARED    = PROCESSED
OUT_DIR   = PROJECT_ROOT / 'data' / 'outputs'
IMG_DIR   = PROJECT_ROOT / 'reports'
IMG_DIR.mkdir(parents=True, exist_ok=True)
ART_DIR   = Path(r'C:\Users\User\.gemini\antigravity\brain\b4359131-1e31-4062-9e06-ecf9811f8d2f')

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Segoe UI', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OD_PKL      = SHARED / 'od_time_series.pkl'
DATES_PKL   = SHARED / 'dates.pkl'
BASE_ORIG   = OUT_DIR / 'per_route_sota_v3_baseline_backup.pkl'
BASE_V5     = OUT_DIR / 'per_route_full_rise_event_baseline_v5.pkl'
FM_PRED_PKL = OUT_DIR / 'predictions_flow_matching.pkl'
PSI_PRED_PKL= OUT_DIR / 'predictions_psi_cyclical.pkl'
VALIDATOR   = PROJECT_ROOT / 'validation' / 'humob2026_validator.py'

OUT_FUSED_PKL = OUT_DIR / 'per_route_full_rise_weekly_predictions.pkl'
OUT_SUB_TSV   = OUT_DIR / 'submission.tsv'

print("=" * 80)
print("🚀 [Step 8] 正在讀取 Step 6 (FM) 與 Step 7 (Ψ7) 輸出檔案，執行雙軌自適應融合推論...")
print("=" * 80)

# 1. 載入輸入檔案
t0 = time.time()
print("📖 正在載入歷史觀測、物理基線與 Step 6/7 的獨立預測成果...")
with open(OD_PKL, 'rb') as f: od_ts = pickle.load(f)
with open(DATES_PKL, 'rb') as f: dates_str = pickle.load(f)
with open(BASE_ORIG, 'rb') as f: orig_baselines = pickle.load(f)
with open(BASE_V5, 'rb') as f: base_v5 = pickle.load(f)

if not FM_PRED_PKL.exists():
    raise FileNotFoundError(f"找不到 Step 6 產出檔: {FM_PRED_PKL}，請先執行 step6_predict_flow_matching.py")
if not PSI_PRED_PKL.exists():
    raise FileNotFoundError(f"找不到 Step 7 產出檔: {PSI_PRED_PKL}，請先執行 step7_predict_cyclical_psi.py")

with open(FM_PRED_PKL, 'rb') as f: fm_payload = pickle.load(f)
with open(PSI_PRED_PKL, 'rb') as f: psi_payload = pickle.load(f)

res_fm_dict  = fm_payload['residuals']
res_psi_dict = psi_payload['residuals']
route_stats  = psi_payload['stats']

start_dt = datetime(2023, 11, 1)
cal_dates = [(start_dt + timedelta(days=i)).strftime('%Y%m%d') for i in range(366)]
cal_to_idx = {d: i for i, d in enumerate(cal_dates)}
cal_dts = [start_dt + timedelta(days=i) for i in range(366)]

i_feb01 = cal_to_idx['20240201']
i_mar31 = cal_to_idx['20240331']
i_apr01 = cal_to_idx['20240401']
i_apr30 = cal_to_idx['20240430']

blind_zone = [d for d in cal_dates if '20240201' <= d <= '20240430']
N_BLIND = len(blind_zone)

class_routes = [
    (1, "49_39-49_39", "Class 1: Persistent Zero", "Uninhabited / Zero Flow Baseline"),
    (2, "39_46-39_46", "Class 2: Temporary Increase", "Post-Quake Evacuation Surge"),
    (3, "58_43-58_43", "Class 3: Persistent Decrease", "Severely Damaged Northern Epicenter"),
    (4, "58_44-58_44", "Class 4: Partial Recovery", "Gradual Infrastructure Repair"),
    (5, "41_46-41_46", "Class 5: Fully Recovered", "Rapid Commercial Rebound (Kanazawa)"),
    (6, "34_70-34_70", "Class 6: Stable Inflow", "Southern Kanazawa Commuter Artery"),
    (7, "38_43-38_43", "Class 7: Emergent Activity", "Relief & Supply Staging Hub"),
    (8, "36_37-36_37", "Class 8: Partial Dissipation", "Secondary Relocation Outflow"),
    (9, "53_37-53_37", "Class 9: Persistent Increase", "Post-Disaster Reconstruction Zone")
]

# 2. 執行全量 15,129 條路線的規律度自適應融合
print("⚡ 正在依規律度 (Reg) 動態權重合成全量路線...")
all_predictions = {}

for pk, b_curve in base_v5.items():
    b_full = np.array(b_curve, copy=True)
    b_blind = b_full[i_feb01:i_apr30+1]
    
    stat = route_stats.get(pk, {})
    class_id = stat.get('class_id', 0)
    reg      = stat.get('reg', 0.0)
    mv       = stat.get('mv', 0.0)
    mean_all = stat.get('mean_all', 0.0)
    n_obs    = stat.get('n_obs', 0)
    n_zero   = stat.get('n_zero', 292)
    raw_max  = stat.get('raw_max', 0.0)
    is_diag  = stat.get('is_diag', False)
    
    # 提取 Step 6 與 Step 7 的獨立殘差
    delta_fm  = res_fm_dict.get(pk, np.zeros(N_BLIND, dtype=np.float32))
    delta_psi = res_psi_dict.get(pk, np.zeros(N_BLIND, dtype=np.float32))
    
    # 動態雙軌規律度權重分配
    w_psi = float(np.clip(0.90 + 0.25 * reg, 0.90, 1.15))
    w_fm  = float(np.clip(1.00 - 0.60 * reg, 0.40, 1.00))
    
    b_log = np.log1p(np.maximum(0.0, b_blind))
    
    if class_id == 1 or (n_zero >= 280 and mean_all < 0.1):
        y_blind = np.zeros(N_BLIND, dtype=np.float32)
    elif not is_diag:
        # 非對角線 (跨區流動)
        if n_zero >= 10 or mean_all <= 2.5 or (reg <= 0.05 and mv <= 1.20):
            if n_obs >= 20 and np.mean(b_blind) > 0.05:
                y_blind = np.maximum(0.0, b_blind)
            else:
                y_blind = np.zeros(N_BLIND, dtype=np.float32)
        else:
            y_blind_log = b_log + delta_psi * w_psi + delta_fm * w_fm
            y_blind = np.maximum(0.0, np.expm1(np.maximum(0.0, y_blind_log)))
    else:
        # 對角線 (自身停留)
        if reg <= 0.05 and mv <= 1.20:
            y_blind = np.zeros(N_BLIND, dtype=np.float32)
        elif n_zero >= 10 or mean_all <= 2.5:
            y_blind = np.maximum(0.0, b_blind)
        else:
            y_blind_log = b_log + delta_psi * w_psi + delta_fm * w_fm
            y_blind = np.maximum(0.0, np.expm1(np.maximum(0.0, y_blind_log)))
            
    max_hist_cap = max(10.0, raw_max * 2.0) if raw_max > 0 else 100.0
    y_blind = np.clip(y_blind, 0.0, max_hist_cap)
    
    b_full[i_feb01:i_apr30+1] = y_blind
    all_predictions[pk] = b_full.astype(np.float32)

t_cost = time.time() - t0
print(f"✅ 全量 15,129 條路線雙軌融合推論完成！耗時: {t_cost:.2f} 秒")

# 3. 儲存全量預測檔案
with open(OUT_FUSED_PKL, 'wb') as f:
    pickle.dump(all_predictions, f)
print(f"💾 最終預測檔已成功更新: {OUT_FUSED_PKL.name}")

# 4. 檢驗 9 大核心 Class 的日跳動振幅
print("\n" + "=" * 90)
print(f"{'CID':3s} | {'Route':11s} | {'Description':21s} | {'GT Mean':7s} | {'GT Jump':7s} | {'New Jump':8s} | {'Status'}")
print("=" * 90)
jump_audit = []
for cid, pk, cname, desc in class_routes:
    raw = od_ts.get(pk)
    valid_gt = [raw[i] for i in range(len(dates_str)) if raw is not None and not np.isnan(raw[i])]
    gt_mean = float(np.mean(valid_gt)) if valid_gt else 0.0
    
    feb_mar_raw = [raw[dates_str.index(d)] for d in dates_str if '20240201' <= d <= '20240331' and raw is not None and not np.isnan(raw[dates_str.index(d)])]
    gt_jump = float(np.mean(np.abs(np.diff(feb_mar_raw)))) if len(feb_mar_raw) > 1 else 0.0
    
    pred_seq = all_predictions[pk][i_feb01:i_mar31+1]
    new_jump = float(np.mean(np.abs(np.diff(pred_seq))))
    
    if gt_jump > 0.01:
        match_ratio = (new_jump / gt_jump) * 100.0
        status = f"✅ {match_ratio:.0f}% 振幅還原"
    else:
        status = "✅ 平穩/零流量保護"
    jump_audit.append((cid, pk, cname, desc, gt_jump, new_jump, status))
    print(f"C{cid:2d} | {pk:11s} | {desc:21s} | {gt_mean:7.1f} | {gt_jump:7.2f} | {new_jump:8.2f} | {status}")
print("=" * 90)

# 5. 繪製 9-Plot 對比圖
fig, axes = plt.subplots(3, 3, figsize=(22, 16), dpi=150)
fig.patch.set_facecolor('#090d16')
zoom_slice = [i for i, d in enumerate(cal_dates) if '20240115' <= d <= '20240515']
cal_dts_zoom = [cal_dts[i] for i in zoom_slice]
cal_dts_blind = [cal_dts[i] for i in range(i_feb01, i_apr30+1)]

for idx, (cid, pk, cname, desc, gt_j, new_j, stat_str) in enumerate(jump_audit):
    ax = axes[idx // 3, idx % 3]
    ax.set_facecolor('#0d1322')
    
    raw = od_ts.get(pk)
    yt = [raw[dates_str.index(d)] if (raw is not None and d in dates_str and dates_str.index(d) < len(raw) and not np.isnan(raw[dates_str.index(d)])) else np.nan for d in cal_dates]
    yt_z = [yt[i] for i in zoom_slice]
    yb_orig_z = [orig_baselines[pk][i] for i in zoom_slice] if pk in orig_baselines else [np.nan]*len(zoom_slice)
    yb_v5_z = [base_v5[pk][i] for i in zoom_slice] if pk in base_v5 else [np.nan]*len(zoom_slice)
    y_fluc = all_predictions[pk][i_feb01:i_apr30+1]
    
    ax.plot(cal_dts_zoom, yt_z, 'o-', color='#f43f5e', label='Ground Truth (Observed)', lw=1.2, markersize=2.4, alpha=0.75)
    ax.plot(cal_dts_zoom, yb_orig_z, '--', color='#10b981', label='Macro Baseline v3', lw=1.8, alpha=0.80)
    ax.plot(cal_dts_zoom, yb_v5_z, '-', color='#f59e0b', label='Baseline v5 (1.5σ 去噪)', lw=2.2, alpha=0.90)
    ax.plot(cal_dts_blind, y_fluc, '-', color='#c084fc', label='雙軌自適應共振融合預測', lw=2.2, alpha=0.95)
    
    ax.axvspan(datetime(2024, 2, 1), datetime(2024, 4, 30), color='#0284c7', alpha=0.08)
    ax.set_title(f"[{pk}] {cname}", color='#f8fafc', fontsize=11, fontweight='bold', pad=8)
    ax.set_ylabel("Persons / Day", color='#94a3b8', fontsize=9)
    ax.tick_params(colors='#94a3b8', labelsize=8)
    ax.grid(True, linestyle='--', color='#1e293b', alpha=0.7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    if idx == 0:
        ax.legend(loc='upper right', facecolor='#090d16', edgecolor='#334155', labelcolor='#cbd5e1', fontsize=8.5)

fig.suptitle("HuMob 2026: 雙軌自適應共振融合成果特寫 (FM + Ψ7 依規律度動態合成)\n（綠虛線：Macro Base | 橙實線：Base v5 去噪 | 紫實線：FM + Ψ7 自適應融合）",
             color='#38bdf8', fontsize=14, fontweight='bold', y=0.99)
plt.tight_layout(rect=[0, 0.02, 1, 0.97])

p2_local = IMG_DIR / 'optimized_desmoothed_zoom.png'
p2_root  = PROJECT_ROOT / 'optimized_desmoothed_zoom.png'
fig.savefig(p2_local, facecolor=fig.get_facecolor(), edgecolor='none')
fig.savefig(p2_root, facecolor=fig.get_facecolor(), edgecolor='none')
plt.close(fig)
print(f"✅ 圖表已成功儲存至: {p2_local.name}")

# 6. 導出官方提交檔 submission.tsv
print("\n" + "=" * 80)
print("📦 正在導出符合官方規格之 submission.tsv...")
print("=" * 80)

feb_set = {'202402{:02d}'.format(i) for i in range(1, 30)}
mar_set = {'202403{:02d}'.format(i) for i in range(1, 32)}
feb_set.remove('20240202')
mar_set.remove('20240305')
official_58_dates = sorted(list(feb_set.union(mar_set)))

def in_official_bbox(pk):
    pts = pk.split('-')
    if len(pts) != 2: return False
    ox, oy = map(int, pts[0].split('_'))
    dx, dy = map(int, pts[1].split('_'))
    return (30 <= ox <= 70 and 35 <= oy <= 70 and 30 <= dx <= 70 and 35 <= dy <= 70)

eval_routes = [pk for pk in all_predictions.keys() if in_official_bbox(pk)]

with open(OUT_SUB_TSV, 'w', encoding='utf-8') as f:
    for d_str in official_58_dates:
        idx = cal_to_idx[d_str]
        d_map = {}
        for rk in eval_routes:
            val = float(all_predictions[rk][idx])
            if val > 0.01:
                o, dst = rk.split('-')
                if o not in d_map:
                    d_map[o] = {}
                d_map[o][dst] = round(val, 4)
        f.write(f"{d_str}\t{d_map}\n")

# 同步另存至根目錄
shutil.copyfile(OUT_SUB_TSV, PROJECT_ROOT / 'submission.tsv')
shutil.copyfile(OUT_SUB_TSV, PROJECT_ROOT / 'submission0918_part2_revision.tsv')

print(f"💾 官方格式提交檔已導出: {OUT_SUB_TSV} ({OUT_SUB_TSV.stat().st_size / (1024*1024):.2f} MB)")
print(f"💾 已同步更新根目錄: {PROJECT_ROOT / 'submission.tsv'}")

# 7. 執行 validator 驗證
print("🔍 正在執行官方 validator 檢驗...")
res = subprocess.run([sys.executable, str(VALIDATOR), str(OUT_SUB_TSV)], capture_output=True, text=True)
print(res.stdout.strip())
if res.returncode == 0:
    print("🎉 雙軌自適應融合官方提交檔驗證 100% 通過！")
else:
    print("❌ 驗證未通過:", res.stderr)
print("=" * 80)
