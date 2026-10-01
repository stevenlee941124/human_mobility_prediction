# HuMob 2026: 核心預測流程庫 (Raw Data ➔ Final Submission)

本資料夾專門收納**從原始觀測資料（Raw Data）到最終合格提交檔（`submission.tsv`）的唯一純淨代碼路徑**。

---

## 包含檔案與結構

```text
pipeline/
├── step1_extract_od_time_series.py          # Step 1: 提取 15,129 條 OD 序列
├── step2_compute_macro_baseline.py          # Step 2: SOTA v3 物理宏觀基線 (含 Class 1 全零防禦)
├── step3_build_flow_matching_dataset.py     # Step 3: 建構 Flow Matching 空間張量與條件特徵
├── step4_train_origin_flow_matching.py      # Step 4: 訓練條件 U-Net 向量場網路
├── step5_train_event_baseline_v5.py         # Step 5: 1.5σ 去噪訓練事件基線 Base v5
├── step6_generate_adaptive_predictions.py   # Step 6: 雙軌共振推論並導出 submission.tsv
├── run_pipeline.py                          # 一鍵依序執行 Step 1 ~ 6
└── src/                                     # ★ 管線運行所必備的 7 大演算法核心模組
    ├── per_route_sota_v3_baseline.py        # 9 類分層、中點法與指數阻尼橋樑
    ├── origin_flow_matching.py              # Flow Matching 神經網路與 Euler ODE
    ├── cyclical_psi.py                      # Ψ7 作息週波分桶與零均值中心化
    ├── route_features.py                    # 路線規律度 (Reg) 與統計特徵提取
    ├── residual_fusion.py                   # 雙軌共振融合引擎與三級分層防線
    ├── japan_calendar.py                    # 日本節假日日曆工具
    └── baseline_module.py                   # 空間網格基線向量化計算
```

---

## 執行方式

在本地進入本目錄後，依序執行 Step 1~6：

```bash
# 方式 A：一鍵執行全套流程
python run_pipeline.py

# 方式 B：單步逐步執行
python step1_extract_od_time_series.py
python step2_compute_macro_baseline.py
python step3_build_flow_matching_dataset.py
python step4_train_origin_flow_matching.py
python step5_train_event_baseline_v5.py
python step6_generate_adaptive_predictions.py
```
執行完畢後，合格提交檔將自動同步生成至專案根目錄的 `submission.tsv`！
