# HuMob 2026: 官方規格驗證與評估模組 (Validation & Evaluation)

本資料夾負責對生成的提交檔進行**官方 100% 格式規範檢驗、數值實體解析繪圖與評估指標計算**。

---

## 包含檔案

* **`verify_submission.py`**：一鍵檢驗 `submission.tsv` 是否符合官方 58 天、Tab 分隔、邊界約束等規則，並逐行實體解析資料繪製 9-Plot 對照圖。
* **`humob2026_validator.py`**：競賽官方提供的嚴格格式驗證腳本。
* **`generate_pure_baseline.py`**：生成純宏觀 Base v5 提交檔（`submission_pure_baseline.tsv`），作為無高頻波動的零風險保底基準。
* **`evaluation.py`**：MAE、RMSE、MAPE 與日跳動率審查（Jump Audit）評估函數庫。

---

## 執行方式

```bash
# 1. 驗證最終預測檔並繪製 9 類驗證圖
python validation/verify_submission.py

# 2. 生成並驗證純 Baseline 穩健版本
python validation/generate_pure_baseline.py
```
