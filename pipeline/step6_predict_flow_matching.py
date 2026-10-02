"""
===============================================================================
HuMob 2026: Step 6 - Standalone Pure Flow Matching Prediction Module
===============================================================================
【獨立方法一：純神經生成模型 (Pure Flow Matching)】
在 SOTA Base v5 物理基線基礎上，僅疊加 Flow Matching 連續正規化流學習出的
空間微觀擾動殘差場 Z (無任何 Ψ7 統計週波參與)。

輸出：
  1. data/outputs/predictions_flow_matching.pkl (包含全量 366 天預測與 90 天神經殘差)
  2. data/outputs/submission_flow_matching.tsv  (符合官方格式之純 Flow Matching 提交檔)
===============================================================================
"""
import sys
import time
import math
import pickle
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
from scipy.ndimage import gaussian_filter1d

sys.stdout.reconfigure(encoding='utf-8')
PIPELINE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT  = PIPELINE_ROOT.parent

PROCESSED = PROJECT_ROOT / 'data' / 'processed'
SHARED    = PROCESSED
OUT_DIR   = PROJECT_ROOT / 'data' / 'outputs'
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PIPELINE_ROOT / 'src'))
from route_features import compute_route_statistics

OD_PKL      = SHARED / 'od_time_series.pkl'
DATES_PKL   = SHARED / 'dates.pkl'
BASE_V5     = OUT_DIR / 'per_route_full_rise_event_baseline_v5.pkl'
CACHED_Z    = OUT_DIR / 'cached_active_routes_z.pkl'
FM_META_PKL = OUT_DIR / 'origin_fm_log1p_meta.pkl'
META_V1_PKL = OUT_DIR / 'per_route_exponential_baseline_meta_v1_sota.pkl'
VALIDATOR   = PROJECT_ROOT / 'validation' / 'humob2026_validator.py'

OUT_PRED_PKL = OUT_DIR / 'predictions_flow_matching.pkl'
OUT_SUB_TSV  = OUT_DIR / 'submission_flow_matching.tsv'

print("=" * 80)
print("🚀 [Step 6] 正在執行【純 Flow Matching 神經生成模型】獨立預測...")
print("=" * 80)

# 1. 載入資料
t0 = time.time()
with open(OD_PKL, 'rb') as f: od_ts = pickle.load(f)
with open(DATES_PKL, 'rb') as f: dates_str = pickle.load(f)
with open(BASE_V5, 'rb') as f: base_v5 = pickle.load(f)
with open(CACHED_Z, 'rb') as f: cached_z = pickle.load(f)
with open(FM_META_PKL, 'rb') as f: fm_meta = pickle.load(f)
with open(META_V1_PKL, 'rb') as f: meta_v1 = pickle.load(f)

route_classes = {pk: v.get('class_id', 0) for pk, v in meta_v1['route_info'].items()}
orig_sigmas = {m['o_str']: m['sigma_log'] for m in fm_meta['origin_meta_list']}
dest_coord_map = {d_k: (int(d_k.split('_')[0]) - 1, int(d_k.split('_')[1]) - 1) for d_k in fm_meta['eval_destinations']}

start_dt = datetime(2023, 11, 1)
cal_dates = [(start_dt + timedelta(days=i)).strftime('%Y%m%d') for i in range(366)]
cal_to_idx = {d: i for i, d in enumerate(cal_dates)}

i_feb01 = cal_to_idx['20240201']
i_apr30 = cal_to_idx['20240430']
blind_zone = [d for d in cal_dates if '20240201' <= d <= '20240430']
N_BLIND = len(blind_zone)

EXCLUDED_DATES = {
    '20231126', '20231130', '20231201', '20231203', '20231204', '20231205',
    '20231214', '20240118', '20240123', '20240124', '20240202', '20240305',
    '20240408', '20240426', '20240529', '20240708'
}

print("📊 正在提取各路線統計特性與規律度...")
route_stats = compute_route_statistics(od_ts, dates_str, EXCLUDED_DATES)

# 2. 逐路線推論：純 Flow Matching 神經殘差
print("⚡ 正在為全量 15,129 條路線生成純 Flow Matching 預測...")
all_predictions_fm = {}
all_residuals_fm   = {}

for pk, b_curve in base_v5.items():
    b_full = np.array(b_curve, copy=True)
    b_blind = b_full[i_feb01:i_apr30+1]
    
    stat = route_stats.get(pk, None)
    pts = pk.split('-')
    is_diag = (len(pts) == 2 and pts[0] == pts[1])
    
    if stat is None:
        stat = {
            'reg': 0.0, 'mv': 0.0, 'mean_all': 0.0, 'p_act': 0.0, 'pk_rat': 0.0,
            'n_obs': 0, 'n_zero': 292, 'is_diag': is_diag, 'raw_max': 0.0,
            'target_log_jump': 0.05
        }
    
    stat['class_id'] = route_classes.get(pk, 0)
    class_id = stat['class_id']
    reg      = stat['reg']
    mv       = stat['mv']
    mean_all = stat['mean_all']
    n_obs    = stat['n_obs']
    n_zero   = stat['n_zero']
    raw_max  = stat['raw_max']
    
    # 提取神經採樣序列 Z 與 sigma
    z_seq = cached_z.get(pk, np.zeros(N_BLIND, dtype=np.float32))
    z_eff = gaussian_filter1d(z_seq, sigma=1.0, mode='nearest')
    
    o_str = pts[0]
    d_str = pts[1] if len(pts) > 1 else pts[0]
    dx, dy = dest_coord_map.get(d_str, (0, 0))
    sigma_log = orig_sigmas.get(o_str)
    sig_val = float(sigma_log[0, dx, dy]) if sigma_log is not None else 0.20
    
    # 活躍主幹路線 Flow Matching 增益 (Class 5 & 8 且 mean > 5.0)
    if class_id in (5, 8) and mean_all > 5.0:
        fm_gain = float(1.00 + 1.50 * np.clip(1.0 - 1.0 * reg, 0.0, 1.0))
    else:
        fm_gain = 1.00
        
    delta_fm = (z_eff * fm_gain) * sig_val
    all_residuals_fm[pk] = delta_fm.astype(np.float32)
    
    # 防禦與分層推論
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
            y_blind_log = b_log + delta_fm
            y_blind = np.maximum(0.0, np.expm1(np.maximum(0.0, y_blind_log)))
    else:
        # 對角線 (自身停留)
        if reg <= 0.05 and mv <= 1.20:
            y_blind = np.zeros(N_BLIND, dtype=np.float32)
        elif n_zero >= 10 or mean_all <= 2.5:
            y_blind = np.maximum(0.0, b_blind)
        else:
            y_blind_log = b_log + delta_fm
            y_blind = np.maximum(0.0, np.expm1(np.maximum(0.0, y_blind_log)))
            
    max_hist_cap = max(10.0, raw_max * 2.0) if raw_max > 0 else 100.0
    y_blind = np.clip(y_blind, 0.0, max_hist_cap)
    
    b_full[i_feb01:i_apr30+1] = y_blind
    all_predictions_fm[pk] = b_full.astype(np.float32)

t_cost = time.time() - t0
print(f"✅ 全量 15,129 條路線純 Flow Matching 推論完成！耗時: {t_cost:.2f} 秒")

# 3. 儲存預測成果封裝檔
payload = {
    'predictions': all_predictions_fm,
    'residuals':   all_residuals_fm,
    'stats':       route_stats,
    'method':      'Pure Flow Matching (Continuous Normalizing Flow U-Net)',
}
with open(OUT_PRED_PKL, 'wb') as f:
    pickle.dump(payload, f)
print(f"💾 純 Flow Matching 預測檔已儲存: {OUT_PRED_PKL.name} ({OUT_PRED_PKL.stat().st_size / (1024*1024):.2f} MB)")

# 4. 導出官方提交檔 submission_flow_matching.tsv
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

eval_routes = [pk for pk in all_predictions_fm.keys() if in_official_bbox(pk)]

with open(OUT_SUB_TSV, 'w', encoding='utf-8') as f:
    for d_str in official_58_dates:
        idx = cal_to_idx[d_str]
        d_map = {}
        for rk in eval_routes:
            val = float(all_predictions_fm[rk][idx])
            if val > 0.01:
                o, dst = rk.split('-')
                if o not in d_map:
                    d_map[o] = {}
                d_map[o][dst] = round(val, 4)
        f.write(f"{d_str}\t{d_map}\n")

print(f"💾 純 Flow Matching 官方提交檔已導出: {OUT_SUB_TSV.name} ({OUT_SUB_TSV.stat().st_size / (1024*1024):.2f} MB)")

# 5. 執行 Validator 驗證
print("🔍 正在執行官方 Validator 檢驗...")
res = subprocess.run([sys.executable, str(VALIDATOR), str(OUT_SUB_TSV)], capture_output=True, text=True)
print(res.stdout.strip())
if res.returncode == 0:
    print("🎉 純 Flow Matching 提交檔驗證 100% 通過！")
else:
    print("❌ 驗證未通過:", res.stderr)
print("=" * 80)
