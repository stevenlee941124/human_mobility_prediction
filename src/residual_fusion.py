"""
===============================================================================
HuMob 2026: Residual Fusion and Prediction Decoding Module (SOTA v3)
===============================================================================
Fuses:
  - SOTA v3 Physical Baseline B_log = log1p(B) (Direct early saturation to April K)
  - Weekly Cyclical Residual psi_week_log (adaptive log residual R = log(Y+1) - log(B+1))
  - Neural Residual Field Z * sigma * diff_w (with temporal smoothing)
With:
  - Dual-track dynamic Regularity weighting (w_psi, w_fm_reg)
  - Strict low-flow / high-sparsity safety fallback to baseline (n_zero >= 10 or mean <= 2.5 -> Baseline)
  - Clean inverse log expm1 decoding and historical capacity protection
===============================================================================
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d


def fuse_predictions(b_blind, z_seq, psi_7_log, blind_weekdays, sig_val,
                     route_stat, cur_gate=None):
    """
    Fuses baseline and residuals for a single route across the 90-day blind zone.

    Args:
        b_blind: np.ndarray(N_BLIND,) baseline values in persons count
        z_seq: np.ndarray(N_BLIND,) sampled neural residual sequence
        psi_7_log: np.ndarray(7,) 7-day cyclical log residual
        blind_weekdays: list of int weekday for each blind day
        sig_val: float, standard deviation of log residual
        route_stat: dict of route statistics (reg, mv, pk_rat, n_zero, mean_all, n_obs, is_diag, raw_max)
        cur_gate: optional np.ndarray(N_BLIND,) dynamic expansion gate

    Returns:
        y_pred: np.ndarray(N_BLIND,) final reconstructed predictions in original persons count
    """
    N_BLIND = len(b_blind)
    is_diag  = route_stat['is_diag']
    reg      = route_stat['reg']
    mv       = route_stat['mv']
    pk_rat   = route_stat['pk_rat']
    n_obs    = route_stat['n_obs']
    n_zero   = route_stat['n_zero']
    mean_all = route_stat['mean_all']
    raw_max  = route_stat['raw_max']

    b_log = np.log1p(np.maximum(0.0, b_blind))

    # 保留高斯平滑 (sigma=1.0) 濾除神經潛在序列之日點間獨立白噪音，杜絕高頻折線暴走
    z_eff = gaussian_filter1d(z_seq, sigma=1.0, mode='nearest')

    # Weekly cyclical residual sequence across the blind days
    if psi_7_log is not None and len(psi_7_log) == 7:
        resid_week_log = np.array([psi_7_log[w] for w in blind_weekdays], dtype=np.float32)
    else:
        resid_week_log = np.zeros(N_BLIND, dtype=np.float32)

    # 規律度動態雙軌權重調配 (Dual-Track Regularity Allocation)
    # 低規律路線 (reg -> 0)：作息不固定，充分釋放 Flow Matching 空間神經殘差 (w_fm -> 1.00)
    # 高規律路線 (reg -> 1)：由生活週期波 psi 主導 (w_psi -> 1.15)，Flow Matching 回歸平緩輔助 (w_fm -> 0.40)
    w_psi = float(np.clip(0.90 + 0.25 * reg, 0.90, 1.15))
    w_fm  = float(np.clip(1.00 - 0.60 * reg, 0.40, 1.00))

    # Option 2 規律度自適應雙軌共振增益 (Adaptive Dual-Boost)：
    # 依據路線類別 (Class 5 & 8) 與平均日人流門檻 (mean_all > 5.0) 進行全域系統性調控
    # 同時結合規律度 reg 進行平滑自適應插值：
    # - 低規律度 (reg -> 0)：週波中位數被非規律稀釋，全額給予增益 (psi_boost -> 1.80, fm_gain -> 2.50)
    # - 高規律度 (reg -> 1)：原本作息已極度分明，平緩退回標準倍率 (psi_boost -> 1.00, fm_gain -> 1.00)，避免週末波谷超跌
    class_id = route_stat.get('class_id', 0)
    flow_threshold = route_stat.get('boost_flow_threshold', 5.0)
    if class_id in (5, 8) and mean_all > flow_threshold:
        psi_boost = float(1.00 + 0.80 * np.clip(1.0 - 1.2 * reg, 0.0, 1.0))
        fm_gain   = float(1.00 + 1.50 * np.clip(1.0 - 1.0 * reg, 0.0, 1.0))
    else:
        psi_boost = 1.00
        fm_gain   = 1.00

    # 平滑宏觀神經擴散殘差 (由 w_fm 動態調控強度，經 sigma=1.0 高斯平滑保證連續性，杜絕高頻暴走)
    resid_week_log = resid_week_log * psi_boost
    resid_fm_log   = (z_eff * fm_gain) * (sig_val * w_fm)

    if not is_diag:
        # 【非對角線 (跨區流動)】
        # 稀疏/低流量保底：直接回退到物理 Baseline
        if n_zero >= 10 or mean_all <= 2.5 or (reg <= 0.05 and mv <= 1.20):
            if n_obs >= 20 and np.mean(b_blind) > 0.05:
                y_pred = np.maximum(0.0, b_blind)
            else:
                y_pred = np.zeros(N_BLIND, dtype=np.float32)
        else:
            y_pred_log = b_log + resid_week_log * w_psi + resid_fm_log
            y_pred = np.maximum(0.0, np.expm1(np.maximum(0.0, y_pred_log)))
    else:
        # 【對角線 (自身停留)】
        if reg <= 0.05 and mv <= 1.20:
            y_pred = np.zeros(N_BLIND, dtype=np.float32)
        elif n_zero >= 10 or mean_all <= 2.5:
            # 稀疏/低流量對角線：直接回退到物理 Baseline
            y_pred = np.maximum(0.0, b_blind)
        else:
            # 活躍高流量對角線：純連續雙軌融合 (由規律度自適應分配，無任何人工 if-else 特判)
            y_pred_log = b_log + resid_week_log * w_psi + resid_fm_log
            y_pred = np.maximum(0.0, np.expm1(np.maximum(0.0, y_pred_log)))

    # 物理安全上限保護
    max_hist_cap = max(10.0, raw_max * 2.0) if raw_max > 0 else 100.0
    y_pred = np.clip(y_pred, 0.0, max_hist_cap)

    return y_pred.astype(np.float32)
