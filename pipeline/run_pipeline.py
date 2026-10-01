"""
===============================================================================
HuMob 2026: End-to-End Pipeline Runner (Step 1 to Step 6)
===============================================================================
一鍵依序執行從 Raw Data 到最終 Submission 的全部 6 大流程：
  1. step1_extract_od_time_series.py
  2. step2_compute_macro_baseline.py
  3. step3_build_flow_matching_dataset.py
  4. step4_train_origin_flow_matching.py
  5. step5_train_event_baseline_v5.py
  6. step6_generate_adaptive_predictions.py
===============================================================================
"""
import sys
import time
import subprocess
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
PIPELINE_ROOT = Path(__file__).resolve().parent

STEPS = [
    ("Step 1", "step1_extract_od_time_series.py", "從原始 TSV 提取 15,129 條 OD 序列與 292 觀測日"),
    ("Step 2", "step2_compute_macro_baseline.py", "計算 SOTA v3 物理宏觀基線 (中點法 + 指數阻尼 + Class 1 隔離)"),
    ("Step 3", "step3_build_flow_matching_dataset.py", "構建起點條件張量 (70, 100) 與 8 維時空條件特徵"),
    ("Step 4", "step4_train_origin_flow_matching.py", "訓練條件 Flow Matching 向量場 U-Net 神經網路"),
    ("Step 5", "step5_train_event_baseline_v5.py", "1.5σ 標準差門檻過濾日常噪聲，建立 Base v5"),
    ("Step 6", "step6_generate_adaptive_predictions.py", "Euler ODE 積分 + Ψ7 週期生活作息 + 自適應雙軌共振推論並導出 submission.tsv"),
]

def main():
    print("=" * 80)
    print("🚀 HuMob 2026: 端到端核心預測管線 (Raw Data ➔ Final Submission)")
    print("=" * 80)
    t_start = time.time()

    for idx, (s_name, script_name, desc) in enumerate(STEPS, start=1):
        script_path = PIPELINE_ROOT / script_name
        print(f"\n[{idx}/6] 正在啟動 {s_name}: {script_name}")
        print(f"      說明: {desc}")
        print("-" * 80)
        t_step = time.time()
        
        cmd = [sys.executable, str(script_path)]
        res = subprocess.run(cmd, cwd=str(PIPELINE_ROOT))
        
        if res.returncode != 0:
            print(f"\n❌ {s_name} ({script_name}) 執行失敗！Exit code: {res.returncode}")
            sys.exit(res.returncode)
            
        print(f"✅ {s_name} 順利完成！耗時: {time.time() - t_step:.2f} 秒\n")

    t_total = time.time() - t_start
    print("=" * 80)
    print(f"🎉 恭喜！全套管線 (Step 1 ~ Step 6) 圓滿執行完畢！總耗時: {t_total:.2f} 秒")
    print(f"📁 官方提交檔已成功產出至根目錄 submission.tsv")
    print("=" * 80)

if __name__ == '__main__':
    main()
