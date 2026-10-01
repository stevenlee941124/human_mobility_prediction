"""
===============================================================================
HuMob 2026: Official Proxy Validation Evaluation Module
===============================================================================
Calculates official NRMSE on April 2024 (Proxy Validation):
  - Global 2,178,576 OD Pairs (41x36 grids)
  - Active 1,802 Routes (999 Diagonal + 803 Off-Diagonal)
  - Full 9-class disaster mobility breakdown table
Official Metric Constants:
  - mu_diag = 26.5659 (or 26.57)
  - mu_off  = 0.017596 (or 0.0176)
===============================================================================
"""
import sys
import json
import math
from pathlib import Path
import numpy as np
from tabulate import tabulate

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

EXCLUDED_DATES = {
    '20231126', '20231130', '20231201', '20231203', '20231204', '20231205',
    '20231214', '20240118', '20240123', '20240124', '20240202', '20240305',
    '20240408', '20240426', '20240529', '20240708'
}

MU_DIAG_CONST = 26.5659
MU_OFF_CONST  = 0.017596

def parse_tsv(filepath):
    data = {}
    filepath = Path(filepath)
    if not filepath.exists():
        return data
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            try:
                raw = parts[1].replace(': NA', ': None').replace(':NA', ':None')
                od = eval(raw, {'__builtins__': {}}, {'None': None})
                if od is not None:
                    data[parts[0]] = od
            except:
                pass
    return data


def evaluate_official_proxy(gt_tsv_path, pred_tsv_path, benchmark_json_path=None, model_key="Flow_Matching_SOTA"):
    """
    Evaluates prediction TSV against ground truth dataset for April 2024.
    If benchmark_json_path is provided, outputs the standard verified 9-class active route metrics.
    """
    gt_data = parse_tsv(gt_tsv_path)
    pred_data = parse_tsv(pred_tsv_path)

    eval_dates = [d for d in gt_data.keys() if d.startswith('202404') and d not in EXCLUDED_DATES]
    eval_dates.sort()
    n_days = len(eval_dates)

    if n_days == 0:
        print("❌ 未找到四月份評測有效天數！")
        return None

    all_grids = [f"{x}_{y}" for x in range(30, 71) for y in range(35, 71)]
    N_TOTAL_DIAG = len(all_grids)
    N_TOTAL_OFF = len(all_grids) * (len(all_grids) - 1)

    diag_se = 0.0
    off_se = 0.0

    for d in eval_dates:
        gt_day = gt_data.get(d, {})
        pred_day = pred_data.get(d, {})

        for o_str in all_grids:
            gt_o = gt_day.get(o_str, {})
            pred_o = pred_day.get(o_str, {})

            # Diag
            y_t_d = gt_o.get(o_str, 0.0) or 0.0
            y_p_d = pred_o.get(o_str, 0.0) or 0.0
            diff_d = y_t_d - y_p_d
            diag_se += diff_d * diff_d

            # Off-diag
            for d_str in all_grids:
                if o_str == d_str:
                    continue
                y_t_o = gt_o.get(d_str, 0.0) or 0.0
                y_p_o = pred_o.get(d_str, 0.0) or 0.0
                diff_o = y_t_o - y_p_o
                off_se += diff_o * diff_o

    rmse_diag_global = math.sqrt(diag_se / (N_TOTAL_DIAG * n_days))
    rmse_off_global  = math.sqrt(off_se / (N_TOTAL_OFF * n_days))
    nrmse_diag_global = rmse_diag_global / MU_DIAG_CONST
    nrmse_off_global  = rmse_off_global / MU_OFF_CONST
    combined_global   = 0.5 * (nrmse_diag_global + nrmse_off_global)

    print("\n" + "=" * 85)
    print("🏆【官方 2024 年 4 月份 Proxy Validation 評測結果】")
    print("=" * 85)
    print(f"評測有效天數 : {n_days} 天")
    print(f"全島全域指標 (Global 2,178,576 OD Pairs):")
    print(f"  • Diagonal RMSE (對角線停留)   : {rmse_diag_global:.4f} 人 (NRMSE: {nrmse_diag_global:.4f})")
    print(f"  • Off-Diagonal RMSE (非對角流動): {rmse_off_global:.6f} 人 (NRMSE: {nrmse_off_global:.4f})")
    print(f"  • Global Combined NRMSE         : {combined_global:.4f}")
    print("=" * 85)

    if benchmark_json_path and Path(benchmark_json_path).exists():
        display_active_9class_table(benchmark_json_path, model_key)

    return {
        'rmse_diag_global': rmse_diag_global,
        'rmse_off_global': rmse_off_global,
        'nrmse_diag_global': nrmse_diag_global,
        'nrmse_off_global': nrmse_off_global,
        'combined_global': combined_global
    }


def display_active_9class_table(benchmark_json_path, model_key="Flow_Matching_SOTA"):
    with open(benchmark_json_path, 'r', encoding='utf-8') as f:
        bench = json.load(f)

    active_group = bench.get('active_mode', {})
    if model_key not in active_group:
        return

    m_res = active_group[model_key]
    table_rows = []
    for r in m_res.get('classes', []):
        table_rows.append([
            r['cid'],
            r['class'],
            f"{r['count']:,} 條",
            f"{r['rmse_diag']:.2f}",
            f"{r['nrmse_diag']:.5f}",
            f"{r['rmse_off']:.2f}",
            f"{r['nrmse_off']:.4f}",
            f"{r['combined_nrmse']:.4f}"
        ])

    tot = m_res.get('total', {})
    table_rows.append([
        "Total",
        tot.get('class', 'All Active Routes'),
        f"{tot.get('count', 0):,} 條",
        f"{tot.get('rmse_diag', 0.0):.2f}",
        f"{tot.get('nrmse_diag', 0.0):.5f}",
        f"{tot.get('rmse_off', 0.0):.2f}",
        f"{tot.get('nrmse_off', 0.0):.4f}",
        f"{tot.get('combined_nrmse', 0.0):.4f}"
    ])

    headers = ["ID", "Class Name", "數量", "RMSE_diag (人)", "NRMSE_diag", "RMSE_off (人)", "NRMSE_off", "Combined NRMSE"]
    print("\n📊【1,802 條核心活躍路線 9 大類別詳細表現】:")
    print(tabulate(table_rows, headers=headers, tablefmt="grid"))
    print("=" * 85 + "\n")
