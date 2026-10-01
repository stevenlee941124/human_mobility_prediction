# HuMob 2026: SOTA v3 Per-Route Physical Baseline Module (Midpoint Saturation)
import math
from datetime import datetime, timedelta
import numpy as np
from scipy.ndimage import gaussian_filter1d

EXCLUDED_DATES = {
    '20231126', '20231130', '20231201', '20231203', '20231204', '20231205',
    '20231214', '20240118', '20240123', '20240124', '20240202', '20240305',
    '20240408', '20240426', '20240529', '20240708'
}

def build_all_v3_baselines(od_ts, dates_str, cal_dates, cal_date_to_idx, route_info_map=None, baselines_fallback=None):
    total_days = len(cal_dates)
    obs_date_to_idx = {d: i for i, d in enumerate(dates_str)}

    i_jan01 = cal_date_to_idx['20240101']
    i_feb01 = cal_date_to_idx['20240201']
    i_apr01 = cal_date_to_idx['20240401']

    pre_weeks = []
    cur = datetime(2023, 11, 6)
    while cur + timedelta(days=6) < datetime(2023, 12, 31):
        w_idxs = [cal_date_to_idx[(cur + timedelta(days=k)).strftime('%Y%m%d')] for k in range(7)]
        mid_idx = w_idxs[3]
        pre_weeks.append((w_idxs, mid_idx))
        cur += timedelta(days=7)

    post_weeks = []
    cur = datetime(2024, 4, 1)
    while cur + timedelta(days=6) <= datetime(2024, 10, 31):
        w_idxs = [cal_date_to_idx[(cur + timedelta(days=k)).strftime('%Y%m%d')] for k in range(7)]
        mid_idx = w_idxs[3]
        post_weeks.append((w_idxs, mid_idx))
        cur += timedelta(days=7)

    baselines = {}
    v3_info = {}

    for pk, raw_arr in od_ts.items():
        if raw_arr is None or np.all(np.isnan(raw_arr)) or np.nanmax(raw_arr) < 0.05:
            baselines[pk] = np.zeros(total_days, dtype=np.float32)
            v3_info[pk] = {'class_name': 'Class 1: Persistent Zero', 'class_id': 1, 'a': 0.0, 'K': 0.0}
            continue

        pts = pk.split('-')
        is_diag = (len(pts) == 2 and pts[0] == pts[1])

        y_clean = np.full(total_days, np.nan, dtype=np.float64)
        for d_str, oi in obs_date_to_idx.items():
            if d_str in cal_date_to_idx and oi < len(raw_arr):
                v = raw_arr[oi]
                if not np.isnan(v):
                    y_clean[cal_date_to_idx[d_str]] = float(v)

        mean_v = float(np.nanmean(y_clean)) if np.any(~np.isnan(y_clean)) else 0.0

        anchor_indices, anchor_values = [], []
        for w_idxs, mid_idx in pre_weeks:
            vals = [y_clean[k] for k in w_idxs if cal_dates[k] not in EXCLUDED_DATES and not np.isnan(y_clean[k])]
            if vals:
                mid_val = 0.5 * (np.max(vals) + np.min(vals)) if is_diag else np.mean(vals)
                anchor_indices.append(mid_idx)
                anchor_values.append(mid_val)

        for w_idxs, mid_idx in post_weeks:
            vals = [y_clean[k] for k in w_idxs if cal_dates[k] not in EXCLUDED_DATES and not np.isnan(y_clean[k])]
            if vals:
                mid_val = 0.5 * (np.max(vals) + np.min(vals)) if is_diag else np.mean(vals)
                anchor_indices.append(mid_idx)
                anchor_values.append(mid_val)

        apr_w_vals = [v for idx, v in zip(anchor_indices, anchor_values) if cal_dates[idx].startswith('202404')]
        K = float(np.mean(apr_w_vals)) if apr_w_vals else mean_v

        rolling_jan = np.full(total_days, np.nan, dtype=np.float64)
        for j in range(i_jan01, i_feb01):
            win = [y_clean[k] for k in range(max(0, j - 3), min(total_days, j + 4))
                   if cal_dates[k] not in EXCLUDED_DATES and not np.isnan(y_clean[k])]
            if len(win) >= 3:
                rolling_jan[j] = np.mean(win)

        jan_vals = [(k, rolling_jan[k]) for k in range(i_jan01, i_feb01) if not np.isnan(rolling_jan[k])]
        y_pre = float(np.mean(anchor_values[:4])) if len(anchor_values) >= 4 else mean_v

        if not jan_vals or (y_pre < 0.05 and K < 0.05):
            baselines[pk] = np.zeros(total_days, dtype=np.float32)
            v3_info[pk] = {'class_name': 'Class 1: Persistent Zero', 'class_id': 1, 'a': 0.0, 'K': 0.0}
            continue

        y_jan31 = jan_vals[-1][1]
        v_max = max([v for _, v in jan_vals], default=y_pre)
        j_max = max(jan_vals, key=lambda x: x[1])[0] if jan_vals else i_jan01

        if K > 1.25 * max(y_pre, 1.0) and K >= y_jan31:
            cls_id = 9; cls_name = 'Class 9: Persistent Increase'
        elif y_jan31 > 1.15 * max(K, 1.0) and K > 1.10 * max(y_pre, 1.0):
            cls_id = 8; cls_name = 'Class 8: Partial Dissipation'
        elif v_max > 1.25 * max(y_pre, 1.0) and abs(K - y_pre) <= 0.25 * max(y_pre, 1.0) and j_max <= i_jan01 + 14:
            cls_id = 7; cls_name = 'Class 7: Short-term Surge'
        elif v_max > 1.25 * max(y_pre, 1.0) and y_jan31 > 1.10 * max(y_pre, 1.0):
            cls_id = 2; cls_name = 'Class 2: Temporary Shelter'
        elif K < 0.70 * max(y_pre, 1.0) and K <= y_jan31:
            cls_id = 3; cls_name = 'Class 3: Heavy Damage'
        elif y_jan31 < 0.80 * max(y_pre, 1.0) and K < 0.90 * max(y_pre, 1.0) and K > y_jan31:
            cls_id = 4; cls_name = 'Class 4: Partial Recovery'
        elif y_jan31 < 0.85 * max(y_pre, 1.0) and K >= 0.85 * max(y_pre, 1.0) and K > y_jan31:
            cls_id = 5; cls_name = 'Class 5: General Recovered'
        else:
            cls_id = 6; cls_name = 'Class 6: Normal Steady'

        a_used = 0.0
        if K > y_jan31 * 1.02 and y_jan31 > 0.05:
            min_j, min_v = min(jan_vals, key=lambda x: x[1])
            t_pts = [k - min_j for k, v in jan_vals if k >= min_j and v > 0]
            v_pts = [v for k, v in jan_vals if k >= min_j and v > 0]
            if len(t_pts) >= 4 and min_v > 0 and min_j < i_feb01 - 3:
                a_list = [math.log(v / min_v) / t for t, v in zip(t_pts[1:], v_pts[1:]) if t > 0]
                a_used = float(np.clip(np.median(a_list), 0.001, 0.06))
            else:
                a_used = float(np.clip(math.log(max(0.01, K) / max(0.01, y_jan31)) / 60.0, 0.001, 0.04))
            blind_proj = [min(K, y_jan31 * math.exp(a_used * d)) for d in range(1, 61)]
        elif K < y_jan31 * 0.98 and y_jan31 > 0.05:
            max_j, max_v = max(jan_vals, key=lambda x: x[1])
            t_pts = [k - max_j for k, v in jan_vals if k >= max_j and v > 0]
            v_pts = [v for k, v in jan_vals if k >= max_j and v > 0]
            if len(t_pts) >= 4 and max_v > 0 and max_j < i_feb01 - 3:
                a_list = [math.log(v / max_v) / t for t, v in zip(t_pts[1:], v_pts[1:]) if t > 0]
                a_used = float(np.clip(np.median(a_list), -0.06, -0.001))
            else:
                a_used = float(np.clip(math.log(max(0.01, K) / max(0.01, y_jan31)) / 60.0, -0.04, -0.001))
            blind_proj = [max(K, y_jan31 * math.exp(a_used * d)) for d in range(1, 61)]
        else:
            blind_proj = [y_jan31 + (K - y_jan31) * (d / 60.0) for d in range(1, 61)]

        curve = np.full(total_days, np.nan, dtype=np.float64)
        for idx, v in zip(anchor_indices, anchor_values):
            curve[idx] = v
        for k, v in jan_vals:
            curve[k] = v
        for d in range(60):
            curve[i_feb01 + d] = blind_proj[d]

        valid_mask = ~np.isnan(curve)
        x_all = np.arange(total_days)
        curve_filled = np.interp(x_all, x_all[valid_mask], curve[valid_mask])
        b_366 = gaussian_filter1d(curve_filled, sigma=2.0, mode='nearest')
        b_366 = np.maximum(0.0, b_366)

        baselines[pk] = b_366.astype(np.float32)
        v3_info[pk] = {
            'class_name': cls_name,
            'class_id': cls_id,
            'a': a_used,
            'K': K,
            'mean_v': mean_v
        }

    meta_out = {
        'route_info': v3_info,
        'cal_dates': cal_dates,
        'cal_date_to_idx': cal_date_to_idx,
        'baseline_version': 'SOTA_v3_weekly_midpoint_early_saturation'
    }

    return baselines, meta_out
