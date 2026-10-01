"""
===============================================================================
HuMob 2026: Per-Route Slope-Matched Adaptive Baseline Module
===============================================================================
Features:
1. Dynamic Dual-Anchor (Jan 20-31 -> Apr 01-14).
2. Nearest Post-Shock Extremum (Turning Point) Detection relative to Feb 1st.
3. C^1 Tangent Matching:
   v_0 = (y_jan31 - v_ext) / dt_days
   alpha / (1 - exp(-alpha)) = v_0 / ((y_apr - y_jan31) / 60)
4. Bayesian Shrinkage:
   Shrinks raw alpha towards class prior for low-flow routes, preventing noise overfitting.
===============================================================================
"""
import math
import numpy as np
from scipy.ndimage import gaussian_filter1d

EXCLUDED_DATES = {
    '20231126', '20231130', '20231201', '20231203', '20231204', '20231205',
    '20231214', '20240118', '20240123', '20240124', '20240202', '20240305',
    '20240408', '20240426', '20240529', '20240708'
}

CLASS_PRIORS = {
    1: 1.0, # Zero flow
    2: 2.0, # Temporary shelter (moderate dissipation)
    3: 1.0, # Heavy damage (slow flat recovery)
    4: 1.5, # Partial recovery (lag phase into spring)
    5: 1.5, # General recovered
    6: 1.0, # Normal steady
    7: 5.0, # Short-term surge (rapid collapse)
    8: 1.0, # Partial dissipation
    9: 1.0  # Persistent increase
}

def clean_iqr_outliers(series):
    valid_pos = series[~np.isnan(series) & (series > 0.0)]
    if len(valid_pos) < 8:
        return series
    q25, q75 = np.percentile(valid_pos, [25, 75])
    iqr = q75 - q25
    upper = q75 + 3.0 * iqr
    cleaned = np.copy(series)
    cleaned[cleaned > upper] = np.nan
    # 0.0 is legitimate physical zero traffic, NEVER an outlier
    return cleaned


def solve_alpha_from_slope_ratio(rho: float) -> float:
    """
    Solves for alpha in: alpha / (1 - exp(-alpha)) = rho
    For rho >= 1.05. Clamped between 0.35 and 6.0.
    """
    if rho <= 1.05:
        return 0.5
    if rho >= 6.0:
        return 6.0
    low, high = 0.1, 6.0
    for _ in range(20):
        mid = (low + high) / 2.0
        val = mid / (1.0 - math.exp(-mid))
        if val < rho:
            low = mid
        else:
            high = mid
    return round((low + high) / 2.0, 2)

def compute_per_route_adaptive_baseline(y_366_raw, cal_dates, cal_date_to_idx, mean_v=None):
    """
    Computes per-route adaptive baseline with nearest-extremum slope matching.
    Returns:
        b_366      : (366,) float64 baseline
        cls_name   : str
        cls_id     : int (1..9)
        alpha_used : float (effective alpha applied in blind zone)
    """
    total = len(cal_dates)
    b_366 = np.zeros(total, dtype=np.float64)

    if np.sum(y_366_raw > 0.05) < 5:
        return b_366, "Class 1: Persistent Zero", 1, 1.0

    i_jan01 = cal_date_to_idx['20240101']
    i_jan20 = cal_date_to_idx['20240120']
    i_feb01 = cal_date_to_idx['20240201']
    i_apr01 = cal_date_to_idx['20240401']
    i_apr14 = min(total, cal_date_to_idx['20240414'])

    # 1. IQR 清洗
    y_clean = clean_iqr_outliers(y_366_raw)

    # 2. 7天滾動平均 (排除官方異常日與 2~3 月盲區)
    rolling = np.full(total, np.nan, dtype=np.float64)
    for i in range(total):
        d_str = cal_dates[i]
        if d_str in EXCLUDED_DATES or (i_feb01 <= i < i_apr01):
            continue
        win = [y_clean[j] for j in range(max(0, i - 3), min(total, i + 4))
               if cal_dates[j] not in EXCLUDED_DATES and not (i_feb01 <= j < i_apr01) and not np.isnan(y_clean[j])]
        if win:
            rolling[i] = np.mean(win)

    # 3. 雙錨點與分類特徵
    pre_slice = [rolling[j] for j in range(max(0, i_jan01 - 28), i_jan01) if not np.isnan(rolling[j])]
    y_pre = float(np.mean(pre_slice)) if pre_slice else 0.0

    jan_vals = [(j, rolling[j]) for j in range(i_jan01, i_feb01) if not np.isnan(rolling[j])]
    j_max, v_max = max(jan_vals, key=lambda x: x[1]) if jan_vals else (i_jan01, y_pre)

    jan_anchor_slice = [rolling[j] for j in range(i_jan20, i_feb01) if not np.isnan(rolling[j])]
    y_jan_anchor = float(np.mean(jan_anchor_slice)) if jan_anchor_slice else y_pre
    y_jan31 = rolling[i_feb01 - 1] if not np.isnan(rolling[i_feb01 - 1]) else y_jan_anchor

    apr_anchor_slice = [rolling[j] for j in range(i_apr01, i_apr14) if not np.isnan(rolling[j])]
    y_apr_anchor = float(np.mean(apr_anchor_slice)) if apr_anchor_slice else y_jan_anchor

    # 4. 嚴格區分 1 到 9 大類別
    if y_pre < 0.05 and y_apr_anchor < 0.05:
        cls_id = 1
        cls_name = "Class 1: Persistent Zero"
    elif y_apr_anchor > 1.25 * max(y_pre, 1.0) and y_apr_anchor >= y_jan_anchor:
        cls_id = 9
        cls_name = "Class 9: Persistent Increase"
    elif y_jan_anchor > 1.15 * max(y_apr_anchor, 1.0) and y_apr_anchor > 1.10 * max(y_pre, 1.0):
        cls_id = 8
        cls_name = "Class 8: Partial Dissipation"
    elif v_max > 1.25 * max(y_pre, 1.0) and abs(y_apr_anchor - y_pre) <= 0.25 * max(y_pre, 1.0) and j_max <= i_jan01 + 14:
        cls_id = 7
        cls_name = "Class 7: Short-term Surge & Dissipation"
    elif v_max > 1.25 * max(y_pre, 1.0) and y_jan_anchor > 1.10 * max(y_pre, 1.0):
        cls_id = 2
        cls_name = "Class 2: Temporary Shelter & Assembly"
    elif y_apr_anchor < 0.70 * max(y_pre, 1.0) and y_apr_anchor <= y_jan_anchor:
        cls_id = 3
        cls_name = "Class 3: Heavy Damage & Long-term Decline"
    elif y_jan_anchor < 0.80 * max(y_pre, 1.0) and y_apr_anchor < 0.90 * max(y_pre, 1.0) and y_apr_anchor > y_jan_anchor:
        cls_id = 4
        cls_name = "Class 4: Partial Recovery"
    elif y_jan_anchor < 0.85 * max(y_pre, 1.0) and y_apr_anchor >= 0.85 * max(y_pre, 1.0) and y_apr_anchor > y_jan_anchor:
        cls_id = 5
        cls_name = "Class 5: General Recovered"
    else:
        cls_id = 6
        cls_name = "Class 6: Normal Steady"

    if cls_id == 1:
        return b_366, cls_name, cls_id, 1.0

    # 5. 逐路線自適應 Alpha 求解 (最鄰近轉折點 + 切線速度匹配)
    c_prior = CLASS_PRIORS.get(cls_id, 1.5)
    jan_idxs = [j for j in range(i_jan01, i_feb01) if not np.isnan(rolling[j])]

    if len(jan_idxs) >= 7 and abs(y_apr_anchor - y_jan31) > 0.1:
        jan_vals_arr = np.array([rolling[j] for j in jan_idxs])
        need_decrease = (y_jan31 > y_apr_anchor)
        best_ext_idx = None
        sub_w = min(15, len(jan_idxs))

        if need_decrease:
            for k in range(len(jan_idxs) - 2, 1, -1):
                if jan_vals_arr[k] >= jan_vals_arr[k-1] and jan_vals_arr[k] >= jan_vals_arr[k+1] and jan_vals_arr[k] > y_jan31 + 0.05:
                    best_ext_idx = jan_idxs[k]
                    break
            if best_ext_idx is None:
                sub_k = np.argmax(jan_vals_arr[-sub_w:])
                best_ext_idx = jan_idxs[-sub_w + sub_k]
        else:
            for k in range(len(jan_idxs) - 2, 1, -1):
                if jan_vals_arr[k] <= jan_vals_arr[k-1] and jan_vals_arr[k] <= jan_vals_arr[k+1] and jan_vals_arr[k] < y_jan31 - 0.05:
                    best_ext_idx = jan_idxs[k]
                    break
            if best_ext_idx is None:
                sub_k = np.argmin(jan_vals_arr[-sub_w:])
                best_ext_idx = jan_idxs[-sub_w + sub_k]

        dt_days = float(i_feb01 - 1 - best_ext_idx)
        if dt_days >= 2.0:
            v_ext = rolling[best_ext_idx]
            v0 = (y_jan31 - v_ext) / dt_days
            v_blind_avg = (y_apr_anchor - y_jan31) / 60.0
            rho = v0 / v_blind_avg if abs(v_blind_avg) > 1e-4 else 1.0

            if rho > 0:
                raw_alpha = solve_alpha_from_slope_ratio(rho)
                # 貝氏平滑：大流量權重高，微流量向先驗收縮
                flow_ref = mean_v if mean_v is not None else y_pre
                conf = min(1.0, max(0.2, flow_ref / 5.0))
                alpha_route = round(conf * raw_alpha + (1.0 - conf) * c_prior, 2)
            else:
                alpha_route = c_prior
        else:
            alpha_route = c_prior
    else:
        alpha_route = c_prior

    # 6. 盲區動態插值
    total_blind = float(i_apr01 - i_feb01) # 60 days
    for ci in range(i_feb01, i_apr01):
        tau = float(ci - i_feb01) / total_blind
        if cls_id == 6:
            # Class 6 常態平穩區：採用三次 Hermite S 曲線保證零端點導數
            f_tau = 3.0 * (tau ** 2) - 2.0 * (tau ** 3)
        elif cls_id == 9:
            f_tau = math.sqrt(tau)
        else:
            # 專屬自適應指數
            f_tau = (1.0 - math.exp(-alpha_route * tau)) / (1.0 - math.exp(-alpha_route))
        rolling[ci] = y_jan_anchor + (y_apr_anchor - y_jan_anchor) * f_tau

    # 7. 一體化平滑濾波 (消除接縫微噪聲)
    valid_mask = ~np.isnan(rolling)
    if valid_mask.sum() < 5:
        return np.zeros(total, dtype=np.float64), cls_name, cls_id, alpha_route

    x_all = np.arange(total)
    rolling_filled = np.interp(x_all, x_all[valid_mask], rolling[valid_mask])
    b_366 = gaussian_filter1d(rolling_filled, sigma=3.0, mode='nearest')
    b_366 = np.maximum(0.0, b_366)

    return b_366, cls_name, cls_id, alpha_route
