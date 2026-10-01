"""
===============================================================================
HuMob 2026: Positive Exponential Rise to Plateau Baseline Module (正指數漸近平穩基線)
===============================================================================
Physical Model:
  1. Exponential Rise: y(t) = min(K, y_start * exp(r * t)) until t* = (1/r) * ln(K / y_start)
  2. Early Plateau: for t >= t*, y(t) = K (holds flat and stable, does not rise further)
  3. Dissipation: y(t) = K + excess * exp(-decay * t)

Features:
1. Carrying Capacity K = y_apr_anchor (April post-disaster equilibrium level).
2. Completely Smooth Baseline:
   Provides a clean, continuous macroscopic trend b(t) across all 366 days.
   Micro-cyclical weekly fluctuations are modeled exclusively in the prediction residual!
===============================================================================
"""
import math
from datetime import datetime
import numpy as np
from scipy.ndimage import gaussian_filter1d

EXCLUDED_DATES = {
    '20231126', '20231130', '20231201', '20231203', '20231204', '20231205',
    '20231214', '20240118', '20240123', '20240124', '20240202', '20240305',
    '20240408', '20240426', '20240529', '20240708'
}

CLASS_PRIORS_K = {
    1: 0.0,    # Zero flow
    2: 0.12,   # Temporary shelter (rapid dissipation)
    3: 0.04,   # Heavy damage (slow lingering recovery)
    4: 0.07,   # Partial recovery (medium pace)
    5: 0.14,   # General recovered (rapid recovery, ~20 days to 95%)
    6: 0.10,   # Normal steady
    7: 0.18,   # Short-term surge (fast collapse within 15 days)
    8: 0.10,   # Partial dissipation
    9: 0.08    # Persistent increase
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
    return cleaned

def compute_per_route_exponential_baseline(y_366_raw, cal_dates, cal_date_to_idx, mean_v=None):
    """
    Computes per-route exponential recovery baseline.
    Returns:
        b_366    : (366,) float64 baseline
        cls_name : str
        cls_id   : int (1..9)
        k_used   : float (effective recovery rate k)
    """
    total = len(cal_dates)
    b_366 = np.zeros(total, dtype=np.float64)

    if np.sum(y_366_raw > 0.05) < 5:
        return b_366, "Class 1: Persistent Zero", 1, 0.0

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
        return b_366, cls_name, cls_id, 0.0

    # 5. 自適應連續指數人口動力學基線 (Route-Adaptive Continuous Exponential Baseline)
    # 每條路線擁有專屬的自適應指數變化率 alpha_k：
    # y(t) = y_start * exp(alpha_k * t), t \in [0, T]
    # 絕不人為鉗位截斷成水平直線，保持 100% 自然平滑指數彎曲！
    K = y_apr_anchor
    y_start = y_jan31
    T = float(i_apr01 - i_feb01)  # 60 天 (2月1日 ~ 3月31日)

    if y_start > 0.05 and K > 0.05:
        if K > y_start:
            # 依 1 月真實中心線搜尋震後最低點，擬合指數恢復率 a (y_t = y_min * exp(a * dt))
            jan_pts = [(j, rolling[j]) for j in range(i_jan01 + 2, min(i_jan01 + 25, i_feb01 - 5)) if not np.isnan(rolling[j])]
            j_min, v_min = min(jan_pts, key=lambda x: x[1]) if jan_pts else (None, None)
            a_list = []
            if j_min is not None and v_min > 0.05:
                for j in range(j_min + 1, i_feb01):
                    v_j = rolling[j]
                    if not np.isnan(v_j) and v_j > v_min:
                        dt = float(j - j_min)
                        a_list.append(math.log(v_j / v_min) / dt)
            if a_list and np.mean(a_list) > 0.0005:
                a_used = float(np.clip(np.mean(a_list), 0.001, 0.04))
            else:
                a_used = float((1.0 / T) * math.log(max(0.01, K) / max(0.01, y_start)))
                a_used = float(np.clip(a_used, 0.001, 0.04))
            for ci in range(i_feb01, i_apr01):
                t_day = float(ci - i_feb01)
                val = y_start * math.exp(a_used * t_day)
                rolling[ci] = min(K, val)  # 達到四月水準後平穩接回四月
            k_used = a_used
        else:
            alpha_k = float((1.0 / T) * math.log(max(0.01, K) / max(0.01, y_start)))
            alpha_k = float(np.clip(alpha_k, -0.04, 0.04))
            for ci in range(i_feb01, i_apr01):
                t_day = float(ci - i_feb01)
                val = y_start * math.exp(alpha_k * t_day)
                rolling[ci] = max(K, val)
            k_used = alpha_k
    elif y_start <= 0.05 and K > 0.05:
        # 震後從零增長至活躍：正指數漸進平穩
        alpha_k = 0.05
        for ci in range(i_feb01, i_apr01):
            t_day = float(ci - i_feb01)
            rolling[ci] = max(0.0, K * (1.0 - math.exp(-alpha_k * t_day)))
        k_used = alpha_k
    elif y_start > 0.05 and K <= 0.05:
        # 震後歸零消散：連續指數消散
        alpha_k = -0.05
        for ci in range(i_feb01, i_apr01):
            t_day = float(ci - i_feb01)
            rolling[ci] = max(0.0, y_start * math.exp(alpha_k * t_day))
        k_used = alpha_k
    else:
        k_used = 0.0
        for ci in range(i_feb01, i_apr01):
            rolling[ci] = 0.0

    # 6. 一體化平滑濾波 (消除接縫微噪聲，輸出 100% 平滑宏觀趨勢基線)
    valid_mask = ~np.isnan(rolling)
    if valid_mask.sum() < 5:
        return np.zeros(total, dtype=np.float64), cls_name, cls_id, k_used

    x_all = np.arange(total)
    rolling_filled = np.interp(x_all, x_all[valid_mask], rolling[valid_mask])
    b_366 = gaussian_filter1d(rolling_filled, sigma=2.0, mode='nearest')
    b_366 = np.maximum(0.0, b_366)

    # 宏觀基線保持 100% 圓滑連續 (Smooth)，微觀起伏與生活週波規律由預測殘差負責疊加！
    return b_366, cls_name, cls_id, k_used
