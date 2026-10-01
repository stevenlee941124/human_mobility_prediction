"""
HuMob 2026: Pure Baseline v5 Version Generator
Generates:
1. submission_pure_baseline.tsv (Pure Baseline v5 predictions without residual fluctuations)
2. pure_baseline_zoom_9plot.png (2~4 Month Zoom-In 9-Plot)
3. pure_baseline_full_year_9plot.png (Full Year 366-Day 9-Plot)
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import pickle
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

PACKAGE_ROOT = Path(__file__).resolve().parent
PROCESSED = PACKAGE_ROOT / 'data' / 'processed'
SHARED = PROCESSED
OUT_DIR = PACKAGE_ROOT / 'data' / 'outputs'
IMG_DIR = PACKAGE_ROOT / 'reports'
IMG_DIR.mkdir(parents=True, exist_ok=True)
ART_DIR = Path(r'C:\Users\User\.gemini\antigravity\brain\b4359131-1e31-4062-9e06-ecf9811f8d2f')
VALIDATOR = PACKAGE_ROOT / 'humob2026_validator.py'

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Segoe UI', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("🚀 正在生成純 Baseline v5 版本 (無波動預測與圖表)...")
print("=" * 80)

# Load data
od_ts = pickle.load(open(SHARED / 'od_time_series.pkl', 'rb'))
dates_str = pickle.load(open(SHARED / 'dates.pkl', 'rb'))
base_v5 = pickle.load(open(OUT_DIR / 'per_route_full_rise_event_baseline_v5.pkl', 'rb'))
base_orig = pickle.load(open(OUT_DIR / 'per_route_sota_v3_baseline_backup.pkl', 'rb'))
meta = pickle.load(open(OUT_DIR / 'per_route_exponential_baseline_meta_v1_sota.pkl', 'rb'))
info = meta['route_info']

start_dt = datetime(2023, 11, 1)
cal_dates = [(start_dt + timedelta(days=i)).strftime('%Y%m%d') for i in range(366)]
cal_to_idx = {d: i for i, d in enumerate(cal_dates)}
cal_dts = [start_dt + timedelta(days=i) for i in range(366)]

i_feb01 = cal_to_idx['20240201']
i_apr30 = cal_to_idx['20240430']

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

# ==============================================================================
# 1. 導出官方純 Baseline 提交檔 submission_pure_baseline.tsv
# ==============================================================================
print("📦 正在導出純 Baseline v5 之 submission_pure_baseline.tsv...")
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

eval_routes = [pk for pk in base_v5.keys() if in_official_bbox(pk)]
print(f"評測範圍內之有效 OD 路線數: {len(eval_routes):,} 條")

out_tsv = OUT_DIR / 'submission_pure_baseline.tsv'
out_root_pure = PACKAGE_ROOT / 'submission_pure_baseline.tsv'
with open(out_tsv, 'w', encoding='utf-8') as f, open(out_root_pure, 'w', encoding='utf-8') as f_root:
    for d_str in official_58_dates:
        idx = cal_to_idx[d_str]
        d_map = {}
        for rk in eval_routes:
            # 純 baseline 值 (無任何波動疊加)
            # Class 1 或長期零流量維持 0.0
            cid = info.get(rk, {}).get('class_id', 0)
            if cid == 1:
                val = 0.0
            else:
                val = float(base_v5[rk][idx])
                
            if val > 0.01:
                o, dst = rk.split('-')
                if o not in d_map:
                    d_map[o] = {}
                d_map[o][dst] = round(val, 4)
        f.write(f"{d_str}\t{d_map}\n")
        f_root.write(f"{d_str}\t{d_map}\n")

print(f"💾 純 Baseline 官方提交檔已導出: {out_tsv} ({out_tsv.stat().st_size / (1024*1024):.2f} MB)")

# 執行 validator 驗證
print("🔍 正在執行官方 validator 檢驗純 Baseline 提交檔...")
res = subprocess.run([sys.executable, str(VALIDATOR), str(out_tsv)], capture_output=True, text=True)
print(res.stdout.strip())
if res.returncode == 0:
    print("✅ 純 Baseline 提交檔檢驗 100% 通過！")
else:
    print("❌ 驗證未通過:", res.stderr)

# ==============================================================================
# 2. 繪製圖一：2~4 月評測盲區細節特寫 9-Plot（純 Baseline 版本）
# ==============================================================================
print("🎨 正在繪製圖一：2~4 月評測盲區特寫 9-Plot (純 Baseline 版本)...")
fig, axes = plt.subplots(3, 3, figsize=(24, 16), dpi=150)
fig.patch.set_facecolor('#090d16')

cal_dts_zoom = cal_dts[cal_to_idx['20240109']:cal_to_idx['20240515']+1]
i_z_start = cal_to_idx['20240109']
i_z_end = cal_to_idx['20240515']

for idx, (cid, pk, cname, subtitle) in enumerate(class_routes):
    r_idx = idx // 3
    c_idx = idx % 3
    ax = axes[r_idx, c_idx]
    ax.set_facecolor('#0d1322')
    
    raw = od_ts.get(pk)
    yt = [raw[dates_str.index(d)] if (raw is not None and d in dates_str and dates_str.index(d) < len(raw) and not np.isnan(raw[dates_str.index(d)])) else np.nan for d in cal_dates]
    
    yb_orig = base_orig.get(pk, np.zeros(366))
    yb_v5 = base_v5.get(pk, np.zeros(366))
    if cid == 1:
        yb_v5 = np.zeros(366)
    
    yt_z = yt[i_z_start:i_z_end+1]
    yb_orig_z = yb_orig[i_z_start:i_z_end+1]
    yb_v5_z = yb_v5[i_z_start:i_z_end+1]
    
    # GT points
    ax.plot(cal_dts_zoom, yt_z, 'o-', color='#f43f5e', label='Ground Truth (Observed 真實觀測)', lw=1.2, markersize=2.6, alpha=0.75)
    # Old Baseline
    ax.plot(cal_dts_zoom, yb_orig_z, '--', color='#10b981', label='原本平穩 Baseline (無事件去噪)', lw=2.0, alpha=0.85)
    # Pure Baseline v5 (Bold orange line)
    ax.plot(cal_dts_zoom, yb_v5_z, '-', color='#f59e0b', label='純 Baseline v5 預測 (1.5σ 去噪純淨基線)', lw=2.8, alpha=0.95)
    
    ax.axvspan(datetime(2024, 2, 1), datetime(2024, 4, 30), color='#0284c7', alpha=0.08, label='90-Day Blind Zone' if idx == 0 else None)
    ax.axvspan(datetime(2024, 4, 1), datetime(2024, 4, 30), color='#10b981', alpha=0.06, label='Official Eval (Apr)' if idx == 0 else None)
    
    # Range of pure baseline in blind zone
    b_blind = yb_v5[i_feb01:i_apr30+1]
    b_min, b_max = float(np.min(b_blind)), float(np.max(b_blind))
    
    ax.set_title(f"[{pk}] {cname}\n(純 Baseline 盲區範圍: [{b_min:.1f}, {b_max:.1f}] 人)", color='#f8fafc', fontsize=11, fontweight='bold', pad=8)
    ax.set_ylabel("Persons / Day", color='#94a3b8', fontsize=9)
    ax.tick_params(colors='#94a3b8', labelsize=8)
    ax.grid(True, linestyle='--', color='#1e293b', alpha=0.7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=15))
    
    if idx == 0:
        ax.legend(loc='upper right', facecolor='#090d16', edgecolor='#334155', labelcolor='#cbd5e1', fontsize=9)

fig.suptitle(
    "HuMob 2026: 2~4 月純 Baseline v5 版本盲區細節特寫\n"
    "（綠虛線：原本平穩Baseline | 橙實線：純 Baseline v5 宏觀去噪基線 — 無高頻波動干擾）",
    color='#38bdf8', fontsize=15, fontweight='bold', y=0.99
)
plt.tight_layout(rect=[0, 0.02, 1, 0.97])

p1_local = IMG_DIR / 'pure_baseline_zoom_9plot.png'
p1_art = ART_DIR / 'pure_baseline_zoom_9plot.png'
fig.savefig(p1_local, facecolor=fig.get_facecolor(), edgecolor='none')
fig.savefig(p1_art, facecolor=fig.get_facecolor(), edgecolor='none')
plt.close(fig)
print(f"✅ Saved Pure Baseline Zoom Plot to: {p1_local.name}")

# ==============================================================================
# 3. 繪製圖二：全年度 366 天 9-Plot（純 Baseline 版本）
# ==============================================================================
print("🎨 正在繪製圖二：全年度 366 天 9-Plot (純 Baseline 版本)...")
fig2, axes2 = plt.subplots(3, 3, figsize=(24, 16), dpi=150)
fig2.patch.set_facecolor('#090d16')

for idx, (cid, pk, cname, subtitle) in enumerate(class_routes):
    r_idx = idx // 3
    c_idx = idx % 3
    ax = axes2[r_idx, c_idx]
    ax.set_facecolor('#0d1322')
    
    raw = od_ts.get(pk)
    yt = [raw[dates_str.index(d)] if (raw is not None and d in dates_str and dates_str.index(d) < len(raw) and not np.isnan(raw[dates_str.index(d)])) else np.nan for d in cal_dates]
    
    yb_orig = base_orig.get(pk, np.zeros(366))
    yb_v5 = base_v5.get(pk, np.zeros(366))
    if cid == 1:
        yb_v5 = np.zeros(366)
        
    ax.plot(cal_dts, yt, 'o', color='#f43f5e', label='Ground Truth (Observed 真實觀測)', markersize=2.2, alpha=0.65)
    ax.plot(cal_dts, yb_orig, '--', color='#10b981', label='原本平穩 Baseline (無事件去噪)', lw=1.8, alpha=0.85)
    ax.plot(cal_dts, yb_v5, '-', color='#f59e0b', label='純 Baseline v5 (1.5σ 去噪基線)', lw=2.5, alpha=0.95)
    
    ax.axvspan(datetime(2024, 2, 1), datetime(2024, 4, 30), color='#0284c7', alpha=0.08, label='90-Day Blind Zone' if idx == 0 else None)
    ax.axvspan(datetime(2024, 4, 1), datetime(2024, 4, 30), color='#10b981', alpha=0.06, label='Official Eval (Apr)' if idx == 0 else None)
    
    ax.set_title(f"[{pk}] {cname}\n({subtitle})", color='#f8fafc', fontsize=11, fontweight='bold', pad=8)
    ax.set_ylabel("Persons / Day", color='#94a3b8', fontsize=9)
    ax.tick_params(colors='#94a3b8', labelsize=8)
    ax.grid(True, linestyle='--', color='#1e293b', alpha=0.7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    if idx == 0:
        ax.legend(loc='upper right', facecolor='#090d16', edgecolor='#334155', labelcolor='#cbd5e1', fontsize=9)

fig2.suptitle(
    "HuMob 2026: 366 天全年度純 Baseline v5 巨觀演化 9-Plot\n"
    "（純宏觀演化基線，杜絕任何高頻雜訊過擬合風險）",
    color='#38bdf8', fontsize=15, fontweight='bold', y=0.99
)
plt.tight_layout(rect=[0, 0.02, 1, 0.97])

p2_local = IMG_DIR / 'pure_baseline_full_year_9plot.png'
p2_art = ART_DIR / 'pure_baseline_full_year_9plot.png'
fig2.savefig(p2_local, facecolor=fig2.get_facecolor(), edgecolor='none')
fig2.savefig(p2_art, facecolor=fig2.get_facecolor(), edgecolor='none')
plt.close(fig2)
print(f"✅ Saved Pure Baseline Full Year Plot to: {p2_local.name}")

print("\n🎉 純 Baseline v5 版本全部生成、檢驗與繪圖完成！")
