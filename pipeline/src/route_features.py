"""
===============================================================================
HuMob 2026: Route Features & Regularity Extraction Module
===============================================================================
Extracts per-route statistical features across historical observations:
  - Route regularity (Reg in [0, 1]) based on normalized weekly variance
  - Unconditional mean flow (mv)
  - Active ratio (p_act)
  - Peak ratio (pk_rat based on P95)
  - Missing/zero count (n_zero)
===============================================================================
"""
import math
from datetime import datetime, timedelta
import numpy as np

def compute_route_statistics(od_ts, dates_str, excluded_dates=None):
    """
    Computes statistical profiles for all active routes in od_ts.

    Args:
        od_ts: dict {pair_key: list/array of observed daily flows}
        dates_str: list of str (YYYYMMDD) corresponding to od_ts columns
        excluded_dates: set of date strings to exclude from consideration

    Returns:
        route_stats: dict {pair_key: dict of statistical properties}
    """
    if excluded_dates is None:
        excluded_dates = set()

    normal_dates = [d for d in dates_str if (d < '20240101' or d >= '20240501') and d not in excluded_dates]
    mondays = [d for d in normal_dates if datetime.strptime(d, '%Y%m%d').weekday() == 0]
    d_to_i = {d: i for i, d in enumerate(dates_str)}
    week_indices = []
    for m_str in mondays:
        m_dt = datetime.strptime(m_str, '%Y%m%d')
        w_idx = [d_to_i[(m_dt + timedelta(days=di)).strftime('%Y%m%d')] 
                 for di in range(7) if (m_dt + timedelta(days=di)).strftime('%Y%m%d') in d_to_i]
        if len(w_idx) == 7:
            week_indices.append(w_idx)

    stats = {}

    for pk, raw in od_ts.items():
        if raw is None:
            continue

        valid_v = [float(x) for x in raw if not np.isnan(x)]
        if len(valid_v) == 0:
            continue

        pos_v = [x for x in valid_v if x > 0.0]
        n_obs = len(valid_v)
        n_zero = 292 - n_obs
        p_act = float(len(pos_v) / 292.0)
        mean_all = float(np.sum(pos_v) / 292.0)
        mv = mean_all

        # Regularity: Weekly normalized variance on normal pre/post disaster weeks
        weeks_list = []
        for w_idx in week_indices:
            w_vals = [raw[oi] for oi in w_idx if oi < len(raw) and not np.isnan(raw[oi])]
            if len(w_vals) == 7 and np.std(w_vals) > 1e-5:
                w_arr = np.array(w_vals, dtype=np.float32)
                weeks_list.append((w_arr - np.mean(w_arr)) / (np.std(w_arr) + 1e-6))

        if len(weeks_list) >= 2:
            norm_matrix = np.stack(weeks_list, axis=0)
            mean_var = np.mean(np.var(norm_matrix, axis=0))
            reg = float(np.clip(1.0 - mean_var, 0.0, 1.0))
        else:
            reg = 0.0

        # P95 Peak Ratio
        p95 = float(np.percentile(valid_v, 95)) if len(valid_v) >= 10 else mv
        peak_ratio = float((p95 - mv) / (mv + 1.0))

        # Empirical daily target log jump
        if len(pos_v) > 1:
            log_pos = [math.log1p(x) for x in pos_v]
            target_log_jump = float(np.mean([abs(log_pos[i] - log_pos[i-1]) for i in range(1, len(log_pos))]))
        else:
            target_log_jump = 0.05

        pts = pk.split('-')
        is_diag = (len(pts) == 2 and pts[0] == pts[1])

        stats[pk] = {
            'reg': reg,
            'mv': mv,
            'mean_all': mean_all,
            'p_act': p_act,
            'pk_rat': peak_ratio,
            'n_obs': n_obs,
            'n_zero': n_zero,
            'is_diag': is_diag,
            'raw_max': float(np.nanmax(valid_v)) if valid_v else 0.0,
            'target_log_jump': target_log_jump
        }

    return stats
