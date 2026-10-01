"""
===============================================================================
HuMob 2026: Train LightGBM Event Model with Route Standard Deviation Thresholding
===============================================================================
Filters out natural daily random noise using route-specific sigma:
  - Non-event days: label = 0.0
  - Event days: if |R - mu| <= 1.5 * sigma -> label = 0.0 (Daily natural noise)
                if |R - mu| >  1.5 * sigma -> label = R - mu - sign * 1.5*sigma (True shock)
Prevents overfitting event features to random everyday variance.
===============================================================================
"""
import sys
import pickle
import math
import re
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
from scipy.ndimage import gaussian_filter1d
import lightgbm as lgb

sys.stdout.reconfigure(encoding='utf-8')

PACKAGE_ROOT = Path(__file__).resolve().parent
PROCESSED = PACKAGE_ROOT / 'data' / 'processed'
OUT_DIR = PACKAGE_ROOT / 'data' / 'outputs'
OUT_DIR.mkdir(parents=True, exist_ok=True)
SHARED = PROCESSED

print("=" * 80)
print("🚀 [Step 1] 正在以各路線標準差 (1.5 * sigma) 門檻過濾日常噪音，重新訓練事件模型...")
print("=" * 80)

# 1. 載入資料
with open(SHARED / 'dates.pkl', 'rb') as f: dates_str = pickle.load(f)
with open(SHARED / 'od_time_series.pkl', 'rb') as f: od_ts = pickle.load(f)

# 基準線檔案
BASE_FILE = OUT_DIR / 'per_route_spatial_event_v4_baseline.pkl'
if not BASE_FILE.exists():
    BASE_FILE = OUT_DIR / 'per_route_sota_v3_baseline_backup.pkl'
with open(BASE_FILE, 'rb') as f: base_v4 = pickle.load(f)

start_dt = datetime(2023, 11, 1)
cal_dts = [start_dt + timedelta(days=i) for i in range(366)]
cal_dates = [dt.strftime('%Y%m%d') for dt in cal_dts]
cal_to_idx = {d: i for i, d in enumerate(cal_dates)}
obs_to_idx = {d: i for i, d in enumerate(dates_str)}

i_feb01 = cal_to_idx['20240201']
i_mar31 = cal_to_idx['20240331']
i_apr01 = cal_to_idx['20240401']
i_apr30 = cal_to_idx['20240430']
i_jan01 = cal_to_idx['20240101']
feb_mar_dates = cal_dates[i_feb01:i_mar31+1]

def parse_pk_safe(pk):
    m = re.match(r'^(-?\d+)_(-?\d+)-(-?\d+)_(-?\d+)$', pk)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    return 0, 0, 0, 0

# 建立路線特徵
route_info = {}
for pk, ts in od_ts.items():
    gx_o, gy_o, gx_d, gy_d = parse_pk_safe(pk)
    is_intra = 1 if (gx_o == gx_d and gy_o == gy_d) else 0
    
    b_curve = base_v4.get(pk, np.zeros(366))
    pre_val = np.mean(b_curve[:60])
    apr_val = np.mean(b_curve[i_apr01:i_apr30+1])
    jan_val = b_curve[i_feb01-1]
    
    if pre_val < 0.05 and apr_val < 0.05:
        arch = 1
    elif apr_val > 1.25 * max(pre_val, 1.0) and apr_val >= jan_val:
        arch = 9
    elif jan_val > 1.15 * max(apr_val, 1.0) and apr_val > 1.10 * max(pre_val, 1.0):
        arch = 8
    elif max(b_curve[i_jan01:i_feb01]) > 1.25 * max(pre_val, 1.0) and abs(apr_val - pre_val) <= 0.25 * max(pre_val, 1.0):
        arch = 7
    elif max(b_curve[i_jan01:i_feb01]) > 1.25 * max(pre_val, 1.0):
        arch = 2
    elif apr_val < 0.70 * max(pre_val, 1.0) and apr_val <= jan_val:
        arch = 3
    elif jan_val < 0.80 * max(pre_val, 1.0) and apr_val < 0.90 * max(pre_val, 1.0) and apr_val > jan_val:
        arch = 4
    elif jan_val < 0.85 * max(pre_val, 1.0) and apr_val >= 0.85 * max(pre_val, 1.0):
        arch = 5
    else:
        arch = 6
        
    route_info[pk] = {
        'gx_o': gx_o, 'gy_o': gy_o, 'gx_d': gx_d, 'gy_d': gy_d,
        'is_intra': is_intra,
        'arch': arch
    }

def get_ev_cls(dt_str):
    if dt_str in ['20231229', '20231230', '20231231', '20240101', '20240102', '20240103']: return 2
    if '20240429' <= dt_str <= '20240505' or '20240810' <= dt_str <= '20240816': return 2
    if dt_str in ['20231103', '20231104', '20231105', '20240106', '20240107', '20240108',
                  '20240210', '20240211', '20240212', '20240223', '20240224', '20240225',
                  '20240713', '20240714', '20240715', '20240810', '20240811', '20240812',
                  '20240914', '20240915', '20240916', '20240921', '20240922', '20240923',
                  '20241012', '20241013', '20241014']: return 1
    if '20240108' <= dt_str <= '20240114' or '20240226' <= dt_str <= '20240303' or \
       '20240506' <= dt_str <= '20240512' or '20240817' <= dt_str <= '20240823': return 3
    if '20240316' <= dt_str <= '20240320' or '20240405' <= dt_str <= '20240414' or \
       '20240531' <= dt_str <= '20240602' or dt_str == '20240727' or dt_str in ['20240803', '20240804']: return 4
    return 0

# 2. 計算各路線在非事件日的自然波動標準差 sigma_daily
print("📊 正在統計各路線非事件日日常自然波動標準差 sigma_daily...")
route_non_event_res = {}
for dt_str in dates_str:
    if get_ev_cls(dt_str) == 0:
        cal_i = cal_to_idx[dt_str]
        obs_i = obs_to_idx[dt_str]
        for pk, raw in od_ts.items():
            if raw is None: continue
            y = raw[obs_i]
            if np.isnan(y): continue
            b = base_v4[pk][cal_i]
            r = math.log1p(max(0.0, y)) - math.log1p(max(0.0, b))
            if pk not in route_non_event_res:
                route_non_event_res[pk] = []
            route_non_event_res[pk].append(r)

route_sigmas = {}
route_mus = {}
for pk, r_list in route_non_event_res.items():
    if len(r_list) >= 5:
        route_sigmas[pk] = float(np.std(r_list))
        route_mus[pk] = float(np.mean(r_list))
    else:
        route_sigmas[pk] = 0.20
        route_mus[pk] = 0.0

K_SIGMA = 1.5
print(f"✅ 路線日常標準差提取完成！設定統計顯著性閾值: K = {K_SIGMA} * sigma")

# 3. 構建訓練矩陣（套用 K*sigma 軟閾值過濾）
print("⚙️ 正在建構去除日常白噪音之純淨事件訓練樣本...")
X_all_list, y_all_list = [], []
total_shocks, filtered_noise = 0, 0

for dt_str in dates_str:
    cal_i = cal_to_idx[dt_str]
    obs_i = obs_to_idx[dt_str]
    ev_c = get_ev_cls(dt_str)
    
    for pk, r_info in route_info.items():
        if r_info['arch'] == 1:
            continue
        gt_y = od_ts[pk][obs_i]
        if np.isnan(gt_y):
            continue
        b_v = base_v4[pk][cal_i]
        r = math.log(gt_y + 1.0) - math.log(b_v + 1.0)
        
        sigma_r = route_sigmas.get(pk, 0.20)
        mu_r = route_mus.get(pk, 0.0)
        
        dev = r - mu_r
        if ev_c == 0:
            target = 0.0
        else:
            if abs(dev) <= K_SIGMA * sigma_r:
                target = 0.0
                filtered_noise += 1
            else:
                target = dev - math.copysign(K_SIGMA * sigma_r, dev)
                total_shocks += 1
                
        feat = [
            r_info['gx_o'], r_info['gy_o'], r_info['gx_d'], r_info['gy_d'],
            r_info['is_intra'], r_info['arch'], ev_c
        ]
        X_all_list.append(feat)
        y_all_list.append(target)

print(f"  • 識別出顯著外部事件衝擊樣本: {total_shocks:,} 筆")
print(f"  • 成功過濾日常自然隨機白噪音: {filtered_noise:,} 筆 (噪聲消除率: {filtered_noise/(filtered_noise+total_shocks)*100:.1f}%)")

X_all = np.array(X_all_list, dtype=np.float32)
y_all = np.array(y_all_list, dtype=np.float32)

categorical_features = [4, 5, 6]
train_full_data = lgb.Dataset(X_all, label=y_all, categorical_feature=categorical_features)
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'n_estimators': 300,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': 6,
    'min_child_samples': 50,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'random_state': 42,
    'verbose': -1,
    'n_jobs': -1
}

print("🌲 正在訓練純淨事件 LightGBM 模型...")
model_full = lgb.train(params, train_full_data)

# 4. 推論 2~3 月盲區事件響應
print("🔮 正在推論 2~3 月純淨事件增量調變項...")
feb_mar_feats = []
feb_mar_keys = []
for dt_str in feb_mar_dates:
    cal_i = cal_to_idx[dt_str]
    ev_c = get_ev_cls(dt_str)
    
    for pk, r_info in route_info.items():
        if r_info['arch'] == 1:
            continue
        feat = [
            r_info['gx_o'], r_info['gy_o'], r_info['gx_d'], r_info['gy_d'],
            r_info['is_intra'], r_info['arch'], ev_c
        ]
        feb_mar_feats.append(feat)
        feb_mar_keys.append((pk, cal_i))

X_test_fm = np.array(feb_mar_feats, dtype=np.float32)
pred_res_fm = model_full.predict(X_test_fm)

route_residuals = {pk: np.zeros(60, dtype=np.float32) for pk in od_ts.keys()}
for (pk, cal_i), r_pred in zip(feb_mar_keys, pred_res_fm):
    day_idx_in_fm = cal_i - i_feb01
    route_residuals[pk][day_idx_in_fm] = r_pred

# 5. 儲存新一代純淨事件調變檔
p_lgb_out = OUT_DIR / 'per_route_lgbm_residual_sigma_v5.pkl'
with open(p_lgb_out, 'wb') as f:
    pickle.dump(route_residuals, f)
print(f"💾 新版事件調變殘差已儲存: {p_lgb_out.name}")

# 6. 自 2 月 1 日起全幅疊加至 Baseline，生成全新 Baseline v5
print("🔧 正在自 2 月 1 日起全幅疊加生成全新 Baseline v5...")
base_v5 = {}
for pk in od_ts.keys():
    b_curve = np.array(base_v4[pk], copy=True)
    r_info = route_info[pk]
    if r_info['arch'] == 1:
        b_curve[i_feb01:i_mar31+1] = 0.0
    else:
        raw_res = route_residuals[pk]
        # 輕量平滑 (sigma=1.0) 保持連續無階梯
        smooth_res = gaussian_filter1d(raw_res, sigma=1.0, mode='nearest')
        b_slice = b_curve[i_feb01:i_mar31+1]
        y_hat = np.maximum(0.0, np.exp(np.log(b_slice + 1.0) + smooth_res) - 1.0)
        
        # 端點平滑銜接 (前3天與後3天)
        w_trans = np.linspace(0.0, 1.0, 4)[1:]
        for step_i in range(3):
            alpha = w_trans[step_i]
            y_hat[step_i] = (1.0 - alpha) * b_slice[step_i] + alpha * y_hat[step_i]
            y_hat[-(step_i+1)] = (1.0 - alpha) * b_slice[-(step_i+1)] + alpha * y_hat[-(step_i+1)]
            
        b_curve[i_feb01:i_mar31+1] = y_hat
        
    base_v5[pk] = b_curve.astype(np.float32)

p_base_out = OUT_DIR / 'per_route_full_rise_event_baseline_v5.pkl'
with open(p_base_out, 'wb') as f:
    pickle.dump(base_v5, f)
print(f"💾 全新 Baseline v5 已儲存: {p_base_out.name}")
print("🎉 [Step 1] 完成！日常噪音已成功過濾，事件純淨調變基線建立完畢！")
