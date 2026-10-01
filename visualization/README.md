# HuMob 2026: 專題視覺化與群體波動分析 (Visualization & Analysis)

本資料夾包含用於深入分析預測波動行為、群體路線表現與自適應門檻特性的視覺化分析腳本。

---

## 包含檔案

* **`plot_threshold_5_benefited_routes.py`**：特寫展示日均門檻下修至 5.0 人後，實質受惠之 6 條骨幹路線（波峰波谷自然復甦，與真實觀測高度吻合）。
* **`plot_other_class5_class8_routes.py`**：Class 5（商業核心快速復原）與 Class 8（二次疏散外流停留）各高流量路線的群體波動圖譜。

---

## 執行方式

```bash
# 繪製 5.0 門檻受惠路線圖
python visualization/plot_threshold_5_benefited_routes.py

# 繪製 Class 5 與 Class 8 群體波動對比圖
python visualization/plot_other_class5_class8_routes.py
```
輸出圖表將自動存檔於 `reports/` 資料夾。
