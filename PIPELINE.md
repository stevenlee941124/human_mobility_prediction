# HuMob 2026: Flow Matching + Ψ7 Weekly Waveform + Adaptive Dual-Boost Revision
## 端到端模組化執行手冊 (對應 `submission0918_part2_revision.tsv` 與 `submission.tsv`)

本專案已完成**清晰專業的模組化目錄解耦重構**，核心預測管線、驗證評估、專題視覺化與歷史版本歸檔各司其職，實現真正乾淨易讀、可直接報告的代碼架構。

---

## 目錄功能劃分總覽

```text
├── pipeline/          ➔ 核心流程：從 Raw Data 提取、特徵張量建構、模型訓練到最終生成 submission.tsv
│   └── src/          ➔ 管線專用演算法庫：SOTA v3 基線、Flow Matching、Ψ7 作息、雙軌共振融合等 7 大模組
├── validation/        ➔ 官方規範驗證 (100% Validator)、9-Plot 實體數值解析、純基線對照組與指標庫
├── visualization/     ➔ 專題視覺化與群體多路線波動特性繪圖腳本
├── archive_baselines/ ➔ 抵達 SOTA v3 前的歷史基線探索版本存檔 (v0 ~ v2)
├── data/              ➔ raw / processed / outputs 數據存儲庫
├── reports/           ➔ 視覺化圖表產出庫
└── submission.tsv     ➔ 官方提交檔
```

---

## 一、核心預測管線 (pipeline/)

本目錄包含從原始觀測資料到最終生成提交檔的 6 大步驟：

### 【步驟 1】從 Raw Data 提取時間序列與日期清單
```bash
python pipeline/step1_extract_od_time_series.py
```
* **核心功能**：解析 `data/raw/humob2026-dataset.tsv`，提取 15,129 條 OD 路線序列與 292 個觀測日，完成 BBox 空間邊界過濾。
* **輸入**：`data/raw/humob2026-dataset.tsv`
* **輸出**：
  * `data/processed/od_time_series.pkl`
  * `data/processed/dates.pkl`
  * `data/processed/eval_destinations.pkl`

---

### 【步驟 2】計算 SOTA v3 物理宏觀演化基線 (Macro Baseline)
```bash
python pipeline/step2_compute_macro_baseline.py
```
* **核心功能**：
  1. **全零/無人區硬性隔離**：歷史最大流量 $< 0.05$ 人之路線直接歸入 `Class 1: Persistent Zero`（共 7,976 條，佔比 52.7%），基線賦值全 0。
  2. **中點法與微觀演化**：以每週「波峰波谷中點（Midpoint: $0.5 \times (\max + \min)$）」為中軸錨點，搭配震後一月 7 日滾動滑動平均過渡、四月飽和值 $K$ 與盲區 60 天指數阻尼微分方程橋接，並經高斯濾波（$\sigma=2.0$）達成 $C^1$ 連續。
* **核心模組**：`pipeline/src/per_route_sota_v3_baseline.py`
* **輸出**：
  * `data/outputs/per_route_sota_v3_baseline_backup.pkl`
  * `data/outputs/per_route_exponential_baseline_meta_v1_sota.pkl`

---

### 【步驟 3】建立起點空間張量與特徵資料集
```bash
python pipeline/step3_build_flow_matching_dataset.py
```
* **核心功能**：將目的地人流殘差建構為以起點為中心的空間張量 $Z \in \mathbb{R}^{70 \times 100}$，並批次計算星期正餘弦、月份與空間座標等 8 維條件特徵向量。
* **核心模組**：`pipeline/src/japan_calendar.py`, `pipeline/src/baseline_module.py`
* **輸出**：
  * `data/outputs/origin_fm_meta.pkl`
  * `data/outputs/origin_fm_dataset.npz`

---

### 【步驟 4】訓練條件 Flow Matching 向量場神經網路 (U-Net)
```bash
python pipeline/step4_train_origin_flow_matching.py
```
* **核心功能**：訓練 2D U-Net 連續正規化流，學習在給定空間起點與時間條件下，目的地人流動態擾動向量場。
* **核心模型**：`pipeline/src/origin_flow_matching.py`
* **輸出權重**：`data/outputs/origin_fm_log1p_checkpoint_ep5.pt`

---

### 【步驟 5】以 1.5σ 標準差門檻動態去噪，訓練事件基線 Base v5
```bash
python pipeline/step5_train_event_baseline_v5.py
```
* **核心功能**：以各路線殘差標準差 $\sigma$ 為尺度，僅對超出 $1.5\sigma$ 的非隨機外生衝擊調變基線；小於 $1.5\sigma$ 視為常態隨機噪聲（殘差歸零），徹底消除盲區基線向下拉偏。
* **輸出**：
  * `data/outputs/per_route_full_rise_event_baseline_v5.pkl`
  * `data/outputs/per_route_lgbm_residual_sigma_v5.pkl`

---

### 【步驟 6】Euler ODE 採樣 + 規律度自適應雙軌共振融合 + 官方 TSV 導出
```bash
python pipeline/step6_generate_adaptive_predictions.py
```
* **核心功能**：
  1. **Class 1 硬性 0.0 防禦**：針對 7,976 條無人區/全零路線（或 $n_{\text{zero}} \ge 280$ 且均值 $<0.1$），強制輸出 `0.0`，完全繞過神經網絡與週波生成，杜絕「鬼影人流」。
  2. **極低流量安全保底回退 (Sparsity Fallback)**：對 $n_{\text{zero}} \ge 10$ 或日均流量 $\le 2.5$ 人的稀疏路線，直接回退到平滑物理基線，關閉高頻神經殘差與週波振幅放大，防止泊松小數抽樣雜訊被放大。
  3. **骨幹自適應雙軌共振 (Dual-Boost)**：對日均 $>5.0$ 人的 Class 5/8 骨幹路線，執行 20 步 Euler ODE 數值積分 + $\Psi_7$ 週期生活作息波合成，依規律度自適應釋放自然起伏振幅。
* **核心模組**：`pipeline/src/residual_fusion.py`, `pipeline/src/cyclical_psi.py`, `pipeline/src/route_features.py`
* **輸出**：
  * `data/outputs/per_route_full_rise_weekly_predictions.pkl`
  * `data/outputs/submission.tsv`
  * 專案根目錄：`submission.tsv` 與 `submission0918_part2_revision.tsv`

---

### 【管線快捷執行】一鍵跑完 Step 1 ~ 6
```bash
python pipeline/run_pipeline.py
```

---

## 二、驗證與評估模組 (validation/)

```bash
# 1. 執行官方規格檢驗 (Validation passed!) 並實體繪製 9-Plot 對照圖
python validation/verify_submission.py

# 2. 生成穩健純基線對照版 (無高頻波動的保底提交檔)
python validation/generate_pure_baseline.py
```

---

## 三、專題視覺化模組 (visualization/)

```bash
# 1. 繪製 5.0 人門檻受惠之 6 條骨幹路線特寫圖
python visualization/plot_threshold_5_benefited_routes.py

# 2. 繪製 Class 5 與 Class 8 多路線群體波動圖
python visualization/plot_other_class5_class8_routes.py
```

---

## 四、歷史基線版本存檔 (archive_baselines/)

此目錄收納探索過程中的先導版本（v0 自適應、v1 初版指數、v1_sota 改進版、v2 9類初版分層），詳見 [`archive_baselines/README.md`](archive_baselines/README.md)。
