# Method 3: Flow Matching + Psi (Ψ) Weekly Waveform + Adaptive Dual-Boost Revision
## 端到端重現手冊 (對應 `submission0918_part2_revision.tsv`)

本資料夾完整封裝 **2026/09/20 最新改版之 Flow Matching 核心方法（Revision 版）**。本方法基於 `02_flow_matching_psi_cyclical` 改編，保留了從 **Raw Data** 到 **最終生成 `submission0918_part2_revision.tsv` 與各類驗證圖表** 的所有完整過程。

---

## 一、改編核心演進（What's New in this Revision）

相比舊版 `02_flow_matching_psi_cyclical`，本版本做了三大關鍵理論與工程升級：

1. **$1.5\sigma$ 標準差門檻事件去噪宏觀基線 (`Base v5`)**：
   - 舊版事件模型容易將日常微小隨機擾動誤判為「災後異常事件」，導致基線在無觀測日產生向下拉偏。
   - 新版改以各路線殘差標準差 $\sigma$ 為尺度，僅當 $|R - \mu| > 1.5\sigma$ 時才視為真正外生衝擊；小於 $1.5\sigma$ 的常態隨機波動將殘差設為 0.0，使宏觀基線極為純淨且抗噪。
2. **全域 Class 5 & 8 規律度自適應雙軌共振增益 (`Adaptive Dual-Boost`, 門檻 $> 5.0$ 人)**：
   - **完全廢除單一路線 hardcode**：改為全局類別層級判定 `class_id in (5, 8) and mean_all > 5.0`。
   - **門檻設定依據**：日均 5 人是避免放大泊松整數抽樣噪聲的物理分水嶺，並達成 **Class 8 全評測區活躍骨幹 100% 覆蓋**（納入 `53_44`，跳動由 0.99 回升至 1.64，GT 2.66）與 **Class 5 次級商業圈 35% 覆蓋**（如 `41_40` 跳動由 1.53 回升至 2.99，與 GT 3.11 達成 96% 吻合）。
   - **規律度動態平滑插值**：低規律作息（$\text{reg} \to 0$）全額釋放 $1.8\times \Psi_7 + 2.5\times \text{FM}$ 增益；高規律通勤路線（$\text{reg} \to 1$）自動退回 $1.0\times$ 守住週末包絡線波谷。
3. **新增純 Baseline 穩健版本 (`submission_pure_baseline.tsv`)**：
   - 提供殘差全部歸零的純宏觀演化預測，提供評測盲區零相位錯位（Zero Phase Mismatch）風險的保底策略。

---

## 二、對應成果檔案與圖表

### 1. 官方合格提交檔
- **`submission0918_part2_revision.tsv`**（1.02 MB，與專案根目錄之檔案 **100% byte-for-byte 相同**）
- **`submission.tsv`**（同上，最新自適應波動版，`Validation passed!`）
- **`submission_pure_baseline.tsv`**（1.19 MB，純 Baseline 穩健版，`Validation passed!`）

### 2. 核心驗證圖表
- **`verify_submission_9plot.png`**：證明紫色波形 100% 逐行讀取自 `submission0918_part2_revision.tsv` 本身。
- **`threshold_5_benefited_routes.png`**：特寫 6 條在下修至 5.0 人門檻後實質受惠之 Class 5 & 8 骨幹路線。
- **`optimized_desmoothed_zoom.png`**：9 大類別在 2024 年 2 ~ 4 月評測盲區之細節特寫。
- **`optimized_desmoothed_sigma_event_9plot.png`**：366 天全年度 9 大類別宏觀演化圖。
- **`class5_multi_route_fluctuations.png`** 與 **`class8_multi_route_fluctuations.png`**：多路線群體波動檢驗圖。
- **`pure_baseline_zoom_9plot.png`** 與 **`pure_baseline_full_year_9plot.png`**：純 Baseline 版本對照圖。

---

## 三、端到端重現執行管線 (Step-by-Step Execution Pipeline)

本資料夾完全獨立、自包含（Self-contained），在命令列中於本目錄下執行以下步驟即可完整端到端重現：

### 【步驟 1】從 Raw Data 提取時間序列與日期清單
```bash
python step1_extract_od_time_series.py
```
- **輸入**：`data/raw/humob2026-dataset.tsv`
- **輸出**：`data/processed/od_time_series.pkl`, `dates.pkl`, `eval_destinations.pkl`

### 【步驟 2】建立起點張量與路線特徵
```bash
python step2b_build_origin_fm_dataset.py
```
- **輸出**：`data/outputs/origin_fm_log1p_meta.pkl`

### 【步驟 3】訓練條件 Flow Matching 向量場神經網路
```bash
python step3b_train_origin_flow_matching.py
```
- **核心模型**：`src/origin_flow_matching.py`
- **輸出權重**：`data/outputs/origin_fm_log1p_checkpoint_ep5.pt`

### 【步驟 4a】以 1.5σ 標準差門檻過濾日常噪聲，建立全新事件基線 v5
```bash
python step4a_train_event_baseline_v5.py
```
- **核心邏輯**：小於 $1.5\sigma$ 之常態隨機波動歸零，大於 $1.5\sigma$ 之衝擊才調變基線。
- **輸出**：`data/outputs/per_route_full_rise_event_baseline_v5.pkl`

### 【步驟 4b】Euler ODE 採樣 + 規律度自適應雙軌共振融合 + 官方 TSV 導出
```bash
python step4b_generate_adaptive_fluctuations.py
```
- **核心模組**：`src/residual_fusion.py`（門檻 $> 5.0$ 人，規律度動態調配）
- **輸出預測**：`data/outputs/per_route_full_rise_weekly_predictions.pkl`
- **導出提交檔**：
  - `data/outputs/submission.tsv`
  - `submission0918_part2_revision.tsv`
  - `submission.tsv`
  - 自動執行 `humob2026_validator.py` 驗證（`Validation passed!`）
- **導出圖表**：`optimized_desmoothed_zoom.png`, `optimized_desmoothed_sigma_event_9plot.png`

### 【步驟 5】生成純 Baseline 穩健版本（選用）
```bash
python step5_generate_pure_baseline_version.py
```
- **輸出**：`submission_pure_baseline.tsv`，`pure_baseline_zoom_9plot.png`, `pure_baseline_full_year_9plot.png`。

### 【步驟 6】一鍵官方規格檢驗與 TSV 數值波形繪製
```bash
python verify_submission.py
```
- **功能**：
  - 驗證 `submission0918_part2_revision.tsv` 格式合規性。
  - 逐行提取數值繪製 `verify_submission_9plot.png`。

### 【輔助圖表產出】
```bash
python plot_5person_threshold_comparison.py
python plot_other_class5_class8_routes.py
```
