"""
Visualizes other Class 5 and Class 8 routes to inspect fluctuations in the blind zone.
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
LOCAL_DIR = PACKAGE_ROOT
PROCESSED = PACKAGE_ROOT / 'data' / 'processed'
SHARED = PROCESSED
OUT_DIR = PACKAGE_ROOT / 'data' / 'outputs'
IMG_DIR = PACKAGE_ROOT / 'reports'
IMG_DIR.mkdir(parents=True, exist_ok=True)
ART_DIR = Path(r'C:\Users\User\.gemini\antigravity\brain\b4359131-1e31-4062-9e06-ecf9811f8d2f')

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Segoe UI', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Load data
od_ts = pickle.load(open(SHARED / 'od_time_series.pkl', 'rb'))
dates_str = pickle.load(open(SHARED / 'dates.pkl', 'rb'))
base_v5 = pickle.load(open(OUT_DIR / 'per_route_full_rise_event_baseline_v5.pkl', 'rb'))
preds = pickle.load(open(OUT_DIR / 'per_route_full_rise_weekly_predictions.pkl', 'rb'))
meta = pickle.load(open(OUT_DIR / 'per_route_exponential_baseline_meta_v1_sota.pkl', 'rb'))
info = meta['route_info']

import sys
sys.path.insert(0, str(LOCAL_DIR / 'src'))
from route_features import compute_route_statistics
route_stats = compute_route_statistics(od_ts, dates_str)

start_dt = datetime(2023, 11, 1)
cal_dates = [(start_dt + timedelta(days=i)).strftime('%Y%m%d') for i in range(366)]
cal_to_idx = {d: i for i, d in enumerate(cal_dates)}
cal_dts = [start_dt + timedelta(days=i) for i in range(366)]

i_zoom_start = cal_to_idx['20240101']
i_zoom_end = cal_to_idx['20240520']
i_feb01 = cal_to_idx['20240201']
i_apr30 = cal_to_idx['20240430']

dates_str_set = set(dates_str)

def plot_route_axis(ax, pk, title_extra=""):
    y_raw = od_ts.get(pk, np.full(len(dates_str), np.nan))
    b_curve = base_v5.get(pk, np.zeros(366))
    pred_curve = preds.get(pk, np.zeros(366))
    stat = route_stats.get(pk, {})
    mean_all = stat.get('mean_all', 0.0)
    reg = stat.get('reg', 0.0)
    
    # Extract GT points in zoom range
    gt_dts = []
    gt_vals = []
    for d_str, v in zip(dates_str, y_raw):
        if '20240101' <= d_str <= '20240520' and not np.isnan(v):
            gt_dts.append(cal_dts[cal_to_idx[d_str]])
            gt_vals.append(v)
            
    sub_dts = cal_dts[i_zoom_start:i_zoom_end+1]
    sub_b = b_curve[i_zoom_start:i_zoom_end+1]
    sub_pred = pred_curve[i_zoom_start:i_zoom_end+1]
    
    blind_sub_pred = pred_curve[i_feb01:i_apr30+1]
    amp = float(np.max(blind_sub_pred) - np.min(blind_sub_pred))
    p_min = float(np.min(blind_sub_pred))
    p_max = float(np.max(blind_sub_pred))
    daily_jumps = np.abs(np.diff(blind_sub_pred))
    avg_jump = float(np.mean(daily_jumps))
    
    # Background & shaded blind zone
    ax.set_facecolor('#0f172a')
    ax.axvspan(cal_dts[i_feb01], cal_dts[i_apr30], color='#1e293b', alpha=0.7, 
               label='評測盲區 (Feb 1 - Apr 30)' if ax.get_subplotspec().is_first_col() and ax.get_subplotspec().is_first_row() else "")
    
    # Plot GT
    ax.scatter(gt_dts, gt_vals, color='#f43f5e', s=18, alpha=0.65, edgecolors='none', label='真實觀測 (GT)' if ax.get_subplotspec().is_first_col() and ax.get_subplotspec().is_first_row() else "")
    
    # Plot Baseline
    ax.plot(sub_dts, sub_b, color='#f59e0b', linestyle='--', linewidth=1.5, alpha=0.85, label='物理 Baseline v5' if ax.get_subplotspec().is_first_col() and ax.get_subplotspec().is_first_row() else "")
    
    # Plot Final Prediction
    is_boosted = (mean_all > 5.0)
    line_color = '#a855f7' if is_boosted else '#38bdf8'
    label_pred = f"最終預測 (Option 2 雙軌增益)" if is_boosted else "最終預測 (標準無增益)"
    ax.plot(sub_dts, sub_pred, color=line_color, linewidth=1.8, label=label_pred if ax.get_subplotspec().is_first_col() and ax.get_subplotspec().is_first_row() else "")
    
    boost_badge = "[Boosted: Psi 1.8x + FM 2.5x]" if is_boosted else "[Standard: <= 5.0]"
    ax.set_title(f"{pk} {title_extra}\n{boost_badge} | 均值={mean_all:.1f}人, 盲區區間=[{p_min:.1f}, {p_max:.1f}], 振幅={amp:.1f}人, 跳動={avg_jump:.2f}人", 
                 fontsize=11, color='#e2e8f0', pad=6, fontweight='bold')
    
    ax.grid(True, linestyle=':', alpha=0.25, color='#94a3b8')
    ax.tick_params(colors='#94a3b8', labelsize=9)
    for spine in ax.spines.values():
        spine.set_color('#334155')

# ==============================================================================
# Plot 1: Class 8 Comparison (Boosted vs Non-Boosted)
# ==============================================================================
print("🎨 正在繪製 Class 8 各路線波動對比圖...")
fig, axes = plt.subplots(2, 2, figsize=(18, 11), dpi=160, facecolor='#0b0f19')
fig.suptitle("HuMob 2026: Class 8 (二次疏散外流停留) 各路線波動特性檢視\n(平均日人流 > 10 雙軌增益路線 vs <= 10 標準路線)", 
             fontsize=14, color='#38bdf8', fontweight='bold', y=0.98)

c8_routes = [
    ('36_37-36_37', "(代表路線: 二次避難集結外流)"),
    ('52_53-52_53', "(第二大高流量節點: 災後外移停留)"),
    ('8_97-8_97',   "(未增益對照: 日均 8.7 人)"),
    ('22_38-22_38', "(未增益對照: 日均 8.0 人)")
]

for idx, (pk, extra) in enumerate(c8_routes):
    ax = axes[idx // 2, idx % 2]
    plot_route_axis(ax, pk, extra)

fig.legend(loc='lower center', ncol=4, frameon=True, facecolor='#1e293b', edgecolor='#475569', labelcolor='#f1f5f9', fontsize=10, bbox_to_anchor=(0.5, 0.01))
plt.tight_layout(rect=[0, 0.05, 1, 0.94])
p1_path = IMG_DIR / 'class8_multi_route_fluctuations.png'
plt.savefig(p1_path, facecolor=fig.get_facecolor(), edgecolor='none')
plt.savefig(ART_DIR / 'class8_multi_route_fluctuations.png', facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()
print(f"✅ Saved Class 8 plot to {p1_path}")

# ==============================================================================
# Plot 2: Class 5 Multi-Route Comparison (Different Traffic Tiers > 10)
# ==============================================================================
print("🎨 正在繪製 Class 5 各階層高流量路線波動圖...")
fig, axes = plt.subplots(3, 2, figsize=(18, 15), dpi=160, facecolor='#0b0f19')
fig.suptitle("HuMob 2026: Class 5 (金澤商業核心快速復原) 各高流量路線波動特性檢視 (mean_all > 10.0)", 
             fontsize=14, color='#38bdf8', fontweight='bold', y=0.98)

c5_routes = [
    ('41_47-41_47', "(頂級商業核心樞紐 - 日均 638 人)"),
    ('41_46-41_46', "(金澤商業中心基準路線 - 日均 186 人)"),
    ('30_71-30_71', "(南部商業/客運核心 - 日均 86 人)"),
    ('42_47-42_47', "(核心相鄰熱區 - 日均 86 人)"),
    ('40_37-40_37', "(中高流量商圈停留 - 日均 46 人)"),
    ('3_11-3_11',   "(門檻邊緣骨幹路線 - 日均 15 人)")
]

for idx, (pk, extra) in enumerate(c5_routes):
    ax = axes[idx // 2, idx % 2]
    plot_route_axis(ax, pk, extra)

fig.legend(loc='lower center', ncol=4, frameon=True, facecolor='#1e293b', edgecolor='#475569', labelcolor='#f1f5f9', fontsize=10, bbox_to_anchor=(0.5, 0.01))
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
p2_path = IMG_DIR / 'class5_multi_route_fluctuations.png'
plt.savefig(p2_path, facecolor=fig.get_facecolor(), edgecolor='none')
plt.savefig(ART_DIR / 'class5_multi_route_fluctuations.png', facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()
print(f"✅ Saved Class 5 plot to {p2_path}")
print("🎉 繪圖全部完成！")
