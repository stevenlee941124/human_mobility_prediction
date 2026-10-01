"""
===============================================================================
HuMob 2026: 官方提交檔案 submission0918_part2_revision.tsv 終極一鍵驗證與波形繪製工具
===============================================================================
本腳本提供使用者自我檢驗：
  1. 執行官方 100% 嚴格 Validator（天數、雙欄 Tab、邊界、非負、格式）。
  2. 直接從 submission0918_part2_revision.tsv 逐行解析資料，繪製 9-Plot 曲線圖。
  3. 證明圖中的紫色波動線條「完全來自 submission.tsv 本身」，而非外部變數。
===============================================================================
使用方法：
  python verify_submission.py
===============================================================================
"""
import sys
import os
import re
import ast
import math
import pickle
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.stdout.reconfigure(encoding='utf-8')
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Segoe UI', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

PACKAGE_ROOT = Path(__file__).resolve().parent
PROCESSED = PACKAGE_ROOT / 'data' / 'processed'
OUT_DIR = PACKAGE_ROOT / 'data' / 'outputs'
REPORTS_DIR = PACKAGE_ROOT / 'reports'
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
VALIDATOR = PACKAGE_ROOT / 'humob2026_validator.py'

candidate_tsvs = [
    PACKAGE_ROOT / 'submission0918_part2_revision.tsv',
    PACKAGE_ROOT / 'submission.tsv',
    OUT_DIR / 'submission0918_part2_revision.tsv',
    OUT_DIR / 'submission.tsv'
]

TSV_PATH = None
for p in candidate_tsvs:
    if p.exists():
        TSV_PATH = p
        break

if TSV_PATH is None:
    print("❌ 錯誤：找不到提交檔案 submission0918_part2_revision.tsv 或 submission.tsv！")
    sys.exit(1)

print("=" * 80)
print(f"🔍 [HuMob 2026] 開始執行 {TSV_PATH.name} 官方規格驗證與波形確認")
print("=" * 80)

# =============================================================================
# PART 1: 官方規格驗證 (Official Validator Check)
# =============================================================================
print(f"\n[步驟 1/3] 正在檢驗 {TSV_PATH.name} 官方規範格式...")

res = subprocess.run([sys.executable, str(VALIDATOR), str(TSV_PATH)], capture_output=True, text=True)
print("Validator 輸出結果:")
print("-" * 50)
print(res.stdout.strip())
if res.stderr:
    print("Validator 錯誤:", res.stderr.strip())
print("-" * 50)

if res.returncode != 0:
    print("❌ 官方 Validator 檢驗未通過！請檢查格式。")
    sys.exit(1)
print(f"✅ {TSV_PATH.name} 完美通過官方 Validator 所有約束檢驗！")

# =============================================================================
# PART 2: 逐行實體解析 submission.tsv (實體還原紫色線)
# =============================================================================
print(f"\n[步驟 2/3] 正在從 {TSV_PATH.name} 逐行實體提取 9 類代表路線預測值...")

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

target_pks = {pk: (cid, name, desc) for cid, pk, name, desc in class_routes}
tsv_parsed_preds = {pk: {} for pk in target_pks}
tsv_dates = []

with open(TSV_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t')
        d_str = parts[0]
        tsv_dates.append(d_str)
        record = ast.literal_eval(parts[1])
        for pk in target_pks:
            orig, dst = pk.split('-')
            val = record.get(orig, {}).get(dst, 0.0)
            tsv_parsed_preds[pk][d_str] = val

tsv_dates = sorted(list(set(tsv_dates)))
print(f"✅ 成功從 TSV 實體讀取 {len(tsv_dates)} 個評測日期之數值！")

# =============================================================================
# PART 3: 繪製對照圖 (證明紫實線來自 submission.tsv)
# =============================================================================
print("\n[步驟 3/3] 正在繪製驗證 9-Plot 圖表...")

with open(PROCESSED / 'od_time_series.pkl', 'rb') as f: od_ts = pickle.load(f)
with open(PROCESSED / 'dates.pkl', 'rb') as f: dates_str = pickle.load(f)
with open(OUT_DIR / 'per_route_full_rise_event_baseline_v5.pkl', 'rb') as f: base_v5 = pickle.load(f)

start_dt = datetime(2023, 11, 1)
cal_dates = [(start_dt + timedelta(days=i)).strftime('%Y%m%d') for i in range(366)]
cal_to_idx = {d: i for i, d in enumerate(cal_dates)}
cal_dts = [start_dt + timedelta(days=i) for i in range(366)]

tsv_dts = [datetime.strptime(d, '%Y%m%d') for d in tsv_dates]

fig, axes = plt.subplots(3, 3, figsize=(22, 14), dpi=150)
fig.patch.set_facecolor('#090d16')

zoom_slice = [i for i, d in enumerate(cal_dates) if '20240115' <= d <= '20240515']
cal_dts_zoom = [cal_dts[i] for i in zoom_slice]

for idx, (cid, pk, cname, desc) in enumerate(class_routes):
    r_idx = idx // 3
    c_idx = idx % 3
    ax = axes[r_idx, c_idx]
    ax.set_facecolor('#0d1322')
    
    raw = od_ts.get(pk)
    yt = [raw[dates_str.index(d)] if (raw is not None and d in dates_str and dates_str.index(d) < len(raw) and not np.isnan(raw[dates_str.index(d)])) else np.nan for d in cal_dates]
    yt_z = [yt[i] for i in zoom_slice]
    yb_v5_z = [base_v5[pk][i] for i in zoom_slice] if pk in base_v5 else [np.nan]*len(zoom_slice)
    
    y_tsv_curve = [tsv_parsed_preds[pk].get(d, np.nan) for d in tsv_dates]
    
    # 盲區陰影
    ax.axvspan(datetime(2024, 2, 1), datetime(2024, 4, 30), color='#0284c7', alpha=0.08, label='90-Day Blind Zone' if idx == 0 else None)
    
    # GT
    ax.plot(cal_dts_zoom, yt_z, 'o', color='#f43f5e', label='Ground Truth (真實觀測)' if idx == 0 else None, markersize=3.0, alpha=0.75)
    # Baseline v5
    ax.plot(cal_dts_zoom, yb_v5_z, '--', color='#f59e0b', label='Baseline v5 (1.5σ 去噪事件基線)' if idx == 0 else None, lw=1.8, alpha=0.9)
    # TSV parsed prediction
    ax.plot(tsv_dts, y_tsv_curve, '-', color='#a855f7', label=f'TSV 實體讀取數值 ({TSV_PATH.name})' if idx == 0 else None, lw=2.2, alpha=0.95)
    
    ax.set_title(f"[{pk}] {cname}\n({desc})", color='#f8fafc', fontsize=11, fontweight='bold', pad=6)
    ax.set_ylabel("Persons / Day", color='#94a3b8', fontsize=9)
    ax.tick_params(colors='#94a3b8', labelsize=8)
    ax.grid(True, linestyle='--', color='#1e293b', alpha=0.7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=15))
    
    if idx == 0:
        ax.legend(loc='upper right', facecolor='#090d16', edgecolor='#334155', labelcolor='#cbd5e1', fontsize=9)

fig.suptitle(
    f"HuMob 2026: 官方提交檔案實體數值驗證 9-Plot ({TSV_PATH.name})\n"
    f"（紫實線 100% 逐行讀取自提交檔 | 橙虛線：Baseline v5 | 紅點：真實觀測）",
    color='#38bdf8', fontsize=15, fontweight='bold', y=0.99
)
plt.tight_layout(rect=[0, 0.02, 1, 0.97])

out_plot_local = REPORTS_DIR / 'verify_submission_9plot.png'
out_plot_root = PACKAGE_ROOT / 'verify_submission_9plot.png'
fig.savefig(out_plot_local, facecolor=fig.get_facecolor(), edgecolor='none')
fig.savefig(out_plot_root, facecolor=fig.get_facecolor(), edgecolor='none')
plt.close(fig)

print(f"✅ 實體驗證圖表已繪製並儲存至: {out_plot_local.name} 與 {out_plot_root.name}")
print("\n🎉 驗證全數完成！提交檔百分之百合格無誤。")
