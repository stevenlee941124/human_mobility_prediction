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
GOLDEN_WEEK = {'20240429', '20240430', '20240501', '20240502', '20240503', '20240504', '20240505', '20240506'}

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
    Computes per-route exponential recovery baseline across all 366 days.
    Provides a clean, continuous macroscopic trend b(t) across all 366 days.
    Guarantees:
      - 100% continuous and smooth across all boundaries
      - Eliminates all holiday/Golden Week humps in May (Golden Week belongs to residual)
      - Pure continuous exponential transition in Feb-Mar blind zone: y_start * exp(alpha_k * t)
    Returns:
        b_366    : (366,) float64 baseline
        cls_name : str
        cls_id   : int (1..9)
        k_used   : float (effective recovery rate alpha_k)
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
    i_may01 = cal_date_to_idx['20240501']

    # 1. IQR 清洗
    y_clean = clean_iqr_outliers(y_366_raw)

    # 2. 宏觀物理特徵與雙錨點計算 (使用完整週期的平均值，避免中位數偏向工作日的高位偏差)
    pre_slice = [y_clean[j] for j in range(max(0, i_jan01 - 28), i_jan01)
                 if cal_dates[j] not in EXCLUDED_DATES and not np.isnan(y_clean[j])]
    y_pre = float(np.mean(pre_slice)) if pre_slice else 0.0

    jan_slice = [y_clean[j] for j in range(i_jan01, i_feb01)
                 if cal_dates[j] not in EXCLUDED_DATES and not np.isnan(y_clean[j])]
    jan_anchor_slice = [y_clean[j] for j in range(i_jan20, i_feb01)
                        if cal_dates[j] not in EXCLUDED_DATES and not np.isnan(y_clean[j])]
    y_jan_anchor = float(np.mean(jan_anchor_slice)) if jan_anchor_slice else y_pre
    v_max_jan = float(np.percentile(jan_slice, 90)) if jan_slice else y_pre

    apr_anchor_slice = [y_clean[j] for j in range(i_apr01, i_apr14)
                        if cal_dates[j] not in EXCLUDED_DATES and not np.isnan(y_clean[j])]
    y_apr_anchor = float(np.mean(apr_anchor_slice)) if apr_anchor_slice else y_jan_anchor

    post_slice = [y_clean[j] for j in range(i_may01, total)
                  if cal_dates[j] not in EXCLUDED_DATES and cal_dates[j] not in GOLDEN_WEEK and not np.isnan(y_clean[j])]
    y_post = float(np.mean(post_slice)) if post_slice else y_apr_anchor

    # 3. 嚴格區分 1 到 9 大類別
    if y_pre < 0.05 and y_apr_anchor < 0.05:
        cls_id = 1
        cls_name = "Class 1: Persistent Zero"
    elif y_apr_anchor > 1.25 * max(y_pre, 1.0) and y_apr_anchor >= y_jan_anchor:
        cls_id = 9
        cls_name = "Class 9: Persistent Increase"
    elif y_jan_anchor > 1.15 * max(y_apr_anchor, 1.0) and y_apr_anchor > 1.10 * max(y_pre, 1.0):
        cls_id = 8
        cls_name = "Class 8: Partial Dissipation"
    elif v_max_jan > 1.25 * max(y_pre, 1.0) and abs(y_apr_anchor - y_pre) <= 0.25 * max(y_pre, 1.0):
        cls_id = 7
        cls_name = "Class 7: Short-term Surge & Dissipation"
    elif v_max_jan > 1.25 * max(y_pre, 1.0) and y_jan_anchor > 1.10 * max(y_pre, 1.0):
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

    # 4. 構建貫穿全年 366 天的全局連續平滑物理宏觀基線
    macro = np.zeros(total, dtype=np.float64)

    # 災前階段 (2023/11/01 ~ 2023/12/31): 平穩常態水位 y_pre
    macro[:i_jan01] = y_pre

    # 震災應急階段 (2024/01/01 ~ 2024/01/31): 高斯平滑應急演化 (消除單日雜訊，保留宏觀波形)
    jan_raw = np.array([y_clean[j] if not np.isnan(y_clean[j]) else y_pre for j in range(i_jan01, i_feb01)], dtype=np.float64)
    jan_smooth = gaussian_filter1d(jan_raw, sigma=3.5, mode='nearest')
    macro[i_jan01:i_feb01] = jan_smooth
    y_start = float(macro[i_feb01 - 1])

    # 盲區階段 (2024/02/01 ~ 2024/03/31): 純連續自適應正指數漸近平穩模型
    # y(t) = y_start * exp(alpha_k * t), t \in [0, T]
    K = y_apr_anchor
    T = float(i_apr01 - i_feb01)  # 60 天
    if y_start > 0.05 and K > 0.05:
        alpha_k = float((1.0 / T) * math.log(max(0.01, K) / max(0.01, y_start)))
        alpha_k = float(np.clip(alpha_k, -0.04, 0.04))  # 物理邊界防護
        for ci in range(i_feb01, i_apr01):
            t_day = float(ci - i_feb01)
            macro[ci] = max(0.0, y_start * math.exp(alpha_k * t_day))
        k_used = alpha_k
    elif y_start <= 0.05 and K > 0.05:
        alpha_k = 0.05
        for ci in range(i_feb01, i_apr01):
            t_day = float(ci - i_feb01)
            macro[ci] = max(0.0, K * (1.0 - math.exp(-alpha_k * t_day)))
        k_used = alpha_k
    elif y_start > 0.05 and K <= 0.05:
        alpha_k = -0.05
        for ci in range(i_feb01, i_apr01):
            t_day = float(ci - i_feb01)
            macro[ci] = max(0.0, y_start * math.exp(alpha_k * t_day))
        k_used = alpha_k
    else:
        alpha_k = 0.0
        macro[i_feb01:i_apr01] = 0.0
        k_used = 0.0

    # 災後重建與平穩階段 (2024/04/01 ~ 2024/10/31):
    # 宏觀物理承載力由 K 平滑漸進至長期水位 y_post，絕不被人為滑動平均或黃金週連假污染！
    post_len = total - i_apr01
    tau_post = np.linspace(0.0, 1.0, post_len)
    macro[i_apr01:] = np.maximum(0.0, K + (y_post - K) * (1.0 - np.exp(-3.0 * tau_post)))

    # 5. 全域平滑過渡 (消除邊界微小轉折，輸出 100% 平滑純淨的宏觀基線)
    b_366 = gaussian_filter1d(macro, sigma=2.0, mode='nearest')
    b_366 = np.maximum(0.0, b_366)

    return b_366, cls_name, cls_id, k_used
