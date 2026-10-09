# 06 重現執行紀錄（2026-10-09）

## 環境

- Python 3.11（venv），套件版本見根目錄 `requirements.txt`（torch 2.14 CPU、pandas 2.3.3、numpy 2.4、TA-Lib 0.8.1、yfinance 1.7.0）。
- 原程式是在 **pandas 1.x** 上寫的：在 DatetimeIndex 上用整數做 `.loc` 切片，pandas 2.x 已經會拋出 TypeError。所以做了最小的相容性修正（只改語法、不改邏輯）：
  - `featureFactory.py`：`data.loc[prev_idx:idx, 'Trend']` → `data.iloc[prev_idx:idx, data.columns.get_loc('Trend')]`（pandas 1.x 對整數 `.loc` 切片會退回成位置切片、不含右端點，`iloc` 與之相同）。
  - `preprocessor.py` / `featureFactory.py` 的 `yf.download`：加上 `auto_adjust=False, multi_level_index=False`。新版 yfinance 預設會調整價格並回傳 MultiIndex 欄位，這兩個參數讓行為回到 2024 年的版本。
- `evaluator.py` 有 `import backtrader`（實際未使用），所以要裝 `backtrader`。
- 本次執行用 stub 取代 TensorFlow（只用於 `tf.random.set_seed`），不影響 torch 的結果。

## 資料

已將下載的原始資料快取在 `data/raw/`（2001-01-01 ~ 2024-01-01，yfinance 1.7.0，`auto_adjust=False`）：

| 代號 | 起 | 迄 | 筆數 |
|---|---|---|---|
| ^GSPC / ^IXIC / ^DJI | 2001-01-02 | 2023-12-29 | 5785 |
| ^RUA | 2001-01-02 | 2023-12-29 | 5756 |
| ^IRX / ^FVX / ^TNX / ^TYX | 2001-01-02 | 2023-12-29 | 5779 |
| **^VIX3M** | **2006-07-17** | 2023-12-29 | 4395 |

**證實了 03 §9 的推論**：`^VIX3M` 從 2006-07-17 才有資料，`auto` 清洗會刪掉這天之前的所有列，所以實際資料期間是 2006-07 ~ 2023-12，而不是 README 寫的 2001 年。

資料量：train 8,416 個視窗（4 個指數交錯）、val 800、test 1,240。

## 結果：與 repo 中的 `outputs/` 幾乎完全一致

| 指標 | repo 原始 | 本次重跑 |
|---|---|---|
| 訓練 log（Epoch 10） | train loss 0.6329 / val acc 0.6829 | **完全相同** |
| 驗證 ROC-AUC / PR-AUC | 0.5503 / 0.3591 | **完全相同** |
| 驗證趨勢 Accuracy | 0.3158 | 相同 |
| 測試 ROC-AUC / PR-AUC | 0.6581 / 0.4374 | 0.6594 / 0.4387 |
| 測試趨勢 Accuracy | 0.2929 | 相同 |
| 預測到的反轉數（±5 天內） | 0 | 0 |
| 多空翻轉回測損益（val / test） | +3,694 / +12,808 | 相同 |

測試集 AUC 有千分之一的差異，最可能的原因是 Yahoo 對近年的歷史資料做過微調。其餘數字完全重現。這也證實了 03 列出的問題確實存在於原始結果中，例如：回測與模型預測無關，所以損益一模一樣；預測的反轉數為 0；趨勢 Accuracy 等於 downtrend 占比。

執行時間：約 1 分 46 秒（CPU）。

## 如何執行

```bash
python3.11 -m venv venv && . venv/bin/activate
pip install -r requirements.txt
python main.py        # 會覆寫 outputs/ 與附加 log.txt
```
