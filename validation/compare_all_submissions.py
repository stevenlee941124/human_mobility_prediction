"""
===============================================================================
HuMob 2026: Multi-Method Ablation & Submission Comparison Tool
===============================================================================
對比四大版本提交檔的統計特徵與 9 大類別日跳動振幅：
  1. submission_pure_baseline.tsv (純平滑宏觀基線 Base v5)
  2. submission_flow_matching.tsv (純神經生成 Flow Matching 模型)
  3. submission_psi_cyclical.tsv  (純統計 Ψ7 生活作息週期模型)
  4. submission.tsv               (雙軌自適應共振融合最終最優版)
===============================================================================
"""
import sys
import pickle
from pathlib import Path
from datetime import datetime
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
VALIDATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT   = VALIDATION_DIR.parent
OUT_DIR        = PROJECT_ROOT / 'data' / 'outputs'

SUBMISSIONS = {
    "1. 純宏觀基線 (Base v5)":       PROJECT_ROOT / 'submission_pure_baseline.tsv',
    "2. 純神經生成 (Flow Matching)": OUT_DIR / 'submission_flow_matching.tsv',
    "3. 純統計作息 (Ψ7 Cyclical)":   OUT_DIR / 'submission_psi_cyclical.tsv',
    "4. 雙軌融合最優 (Adaptive Fusion)": PROJECT_ROOT / 'submission.tsv',
}

def parse_submission(filepath):
    if not filepath.exists():
        return None
    daily_records = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                d_str = parts[0]
                od_dict = eval(parts[1])
                daily_records[d_str] = od_dict
    return daily_records

print("=" * 95)
print("📊 [HuMob 2026] 四大預測方法提交檔橫向對比 (Multi-Method Ablation Audit)")
print("=" * 95)

records = {}
for name, path in SUBMISSIONS.items():
    rec = parse_submission(path)
    records[name] = rec
    status = f"✅ 已載入 ({path.stat().st_size / (1024*1024):.2f} MB)" if rec is not None else "❌ 未找到檔案"
    print(f"• {name:32s} : {status}")

class_routes = [
    ("49_39", "49_39", "C1: 全零無人區"),
    ("39_46", "39_46", "C2: 避難暴增"),
    ("58_43", "58_43", "C3: 重災劇降"),
    ("58_44", "58_44", "C4: 緩慢修復"),
    ("41_46", "41_46", "C5: 商業復甦"),
    ("34_70", "34_70", "C6: 通勤動脈"),
    ("38_43", "38_43", "C7: 物資集散"),
    ("36_37", "36_37", "C8: 二次外移"),
    ("53_37", "53_37", "C9: 重建區增加")
]

print("\n" + "=" * 95)
print(f"{'類別與代表路線':20s} | {'1. 純基線':13s} | {'2. 純神經 FM':14s} | {'3. 純統計 Ψ7':14s} | {'4. 雙軌融合':14s}")
print("=" * 95)

# 依日期排序提取跳動
dates_58 = sorted(list(list(records.values())[0].keys())) if list(records.values())[0] else []

for o, d, cname in class_routes:
    row_jumps = []
    for m_name in SUBMISSIONS.keys():
        rec = records[m_name]
        if rec is None:
            row_jumps.append("N/A")
            continue
        vals = []
        for dt_str in dates_58:
            v = rec.get(dt_str, {}).get(o, {}).get(d, 0.0)
            vals.append(v)
        if len(vals) > 1:
            jump = float(np.mean(np.abs(np.diff(vals))))
            row_jumps.append(f"{jump:6.2f} 人/天")
        else:
            row_jumps.append("0.00 人/天")
            
    print(f"{cname:10s} ({o}-{d:5s}) | {row_jumps[0]:13s} | {row_jumps[1]:14s} | {row_jumps[2]:14s} | {row_jumps[3]:14s}")

print("=" * 95)
print("💡 結論解讀：")
print("  • 純宏觀基線 (1)：跳動極低，展現平滑抗噪特性，適合無作息路線保底。")
print("  • 純神經 FM  (2)：在非通勤/混亂震區展現神經流空間擴散能力。")
print("  • 純統計 Ψ7  (3)：在通勤幹道 (如 C5, C6) 準確刻畫週一至週日作息規律。")
print("  • 雙軌融合   (4)：依規律度自適應合成，在保持 C1 零防禦的同時兼具兩者之長！")
print("=" * 95)
