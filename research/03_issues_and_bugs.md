# 03 問題與錯誤清單

嚴重度：🔴 會讓結論錯誤　🟠 影響結果可信度或公平性　🟡 品質 / 可重現性問題

---

## 🔴 1. 回測使用的是真實未來標籤，不是模型預測

**位置**：`postprocessor/postprocessor.py:232-233`、`evaluator/evaluator.py:1035-1048`

```python
passing_trade_signals = \
    self.get_trade_signals(y_test_reverse_signals, y_test_reverse_idx, test_dates, target_dataset)
#                          ^^^^^^ 真實值
```

`main.py` 把它當作 `pass_pred_trade_signals` 傳給 evaluator，四種回測全部使用它。

**證據**（`outputs/reports/summary.json`，測試期 2019–2023）：
- 多空翻轉策略 26 筆交易中 **25 筆獲利**，第一筆 2019-04-30 賣在 2,945.83、2019-06-03 買在 2,744.45，恰好是該段的局部高低點。
- 加 10% 停損的結果與不加停損**完全相同**（完美時機下不會觸發停損）。
- 驗證期 19 筆交易中 18 筆獲利。

**結論**：`outputs/plots/trading_details_*.png` 與報告中的損益是「完美預知」上限（oracle），不是模型績效。若論文或簡報曾引用這些數字，需要重新確認當時使用的是哪一版程式。

**修正**：`passing_trade_signals` 改為由（閾值化後的）`y_preds` 產生，並保留真實訊號的回測作為 oracle 上限對照。

---

## 🔴 2. 反轉三分類 / 二分類混淆矩陣是「真實 vs 真實」

**位置**：`postprocessor.py:238-240`、`evaluator.py:1012-1018`

`reversals_test = calculate_reversal_dates(y_test...)`、`reversals_pred_pass = calculate_reversal_dates_with_signals(y_test...)[0]`，兩者都由 `y_test` 產生（差別只在是否跳過重複視窗）。

**證據**：`summary.json` 與 `val_summary.json` 中 `class_reversals_confusion_matrix_info`、`overall_reversals_confusion_matrix_info`、`pass_reversal_confusion_matrix_info` 全部為 **1.0**。

**修正**：預測端改用 `calculate_reversal_dates_with_signals(y_preds_reverse_signals, y_preds_reverse_idx, ...)`。

---

## 🔴 3. logits 從未經過 sigmoid / 閾值就被當成 0/1 使用

**位置**：`main.py:71,106`（`y_preds = model(X)` 直接傳入）→ `postprocessor.py:219` → `postprocessor.py:63-69`、`evaluator.py:46-48`

模型以 `BCEWithLogitsLoss` 訓練，輸出是 logits（實數）。後處理：

1. `change_values_after_first_reverse_point` 判斷「相鄰值是否相等」—— 連續值在 i=1 幾乎必然不同，於是整列被改成 `[p0, p1, p1, …, p1]`。
2. `get_first_trend_reversal_and_idx_signals` 判斷 `== 1` / `== 0` —— 永遠不成立 → 預測的反轉訊號全為 0。
3. `evaluate_trend_predictions` 判斷 `== 0` 為 uptrend —— 全部被判為 downtrend。

**證據**：
- `summary.json` 的 `y_preds` 每列都是 `[-0.68, -0.61, -0.61, …]` 這種「第 2 格之後全相同」的形態。
- `reverse_difference.predicted_reverse_signals` 全為 0；`reverse_in_range_num = 0`；`reverse_idx_difference_mean = NaN`。
- 趨勢 Accuracy 0.293（測試）/ 0.316（驗證）、uptrend 的 Precision/Recall = 0 —— 正好等於「全部猜 downtrend」時的準確率（= 測試期 downtrend 天數占比）。

**修正**：`y_bin = (torch.sigmoid(logits) > threshold).float()`，threshold 在驗證集上調（不一定是 0.5，因為類別約 7:3）。ROC/PR 用 `sigmoid(logits)` 的**原始**（未經 change_values 修改的）副本計算。

---

## 🟠 4. 舊版實驗（2024-05）把目標 `Trend` 放進輸入特徵

**位置**：git commit `dd800b2` 所有 108 組實驗的 `parameters.json`，`feature_cols` 都包含 `"Trend"`。

`Trend` 由前後 20 天的局部極值決定。輸入視窗最後一天 t 的 Trend 值，需要知道 t ~ t+20 的價格才能確定 —— 等於把未來資訊餵進模型。

此外舊版 `standardize_and_split_data` 有 `X_test = x_data.iloc[train_split_idx:]`（應為 `val_split_idx`），**測試集包含了驗證集**。

**影響**：舊版的趨勢準確率（中位數 0.60、最高 0.79）不能當作模型真實能力。有趣的是，即使有洩漏，準確率也不高，可能因為視窗內 MinMax 正規化 + DNN 攤平後未能有效利用該欄。目前版本已將 `Trend` 從 `feature_cols` 移除 ✅。

---

## 🟠 5. Early stopping 的 rollback 無效

**位置**：`model/model.py:162`

```python
best_model = model.state_dict()   # 回傳的是參數 tensor 的參考
```

後續 `optimizer.step()` 會就地更新參數，`best_model` 也跟著變。觸發 early stopping 時 `load_state_dict(best_model)` 載入的是**最新**權重。另外驗證集 loss 沒改善時也沒有保存最佳模型給最終評估使用（未觸發 early stopping 時直接用最後一個 epoch）。

**修正**：`best_model = copy.deepcopy(model.state_dict())`，並在訓練結束後一律載入最佳權重。

目前設定 `patience=100 > epochs=10`，所以實際上不會觸發，但舊版實驗（1000 epochs、patience 50）有觸發（`log.txt` 中 4 次 "Early stopping at epoch ..."）。

---

## 🟠 6. 訓練設定幾乎沒有訓練到模型

- `learning_rate = 1e-5`、`training_epoch_num = 10`：`log.txt` 最後一次執行（第 1194–1203 行）train loss 0.72 → 0.63，val acc 0.56 → **0.683**。驗證期 downtrend 占 31.6%（見問題 3），所以「全部猜 uptrend」的準確率就是 0.684 —— 模型實質上只學到猜多數類。
- `batch_size` 參數被 `main.py:55` 寫死的 32 覆蓋。
- `weight_decay` 未傳給 optimizer。
- 驗證集同時用於 early stopping 與最終報告的 `val_summary.json`，沒有獨立的模型選擇集（這點可接受，但要在論文中說清楚）。

---

## 🟠 7. Transformer 沒有位置編碼

**位置**：`model/modelFactory.py:301-317`

Self-attention 對輸入順序置換不變；沒有 positional encoding 時，模型無法區分「64 天前」和「昨天」。`transformer(x, x)` 把 encoder-decoder 當成 encoder 用，也沒有 causal mask。與 GRU/LSTM 比較時對 Transformer 不公平。

---

## 🟠 8. 評估設計問題

1. **反轉類型矩陣只看真實有反轉的視窗**（`reverse_difference.iloc[valid_signals]`）→ false positive（預測有反轉但實際沒有）完全不計，會高估 precision。
2. **反轉位置幾乎都在 idx 15**：滑動步長 1 時，真實反轉第一次出現在視窗最後一格，之後的視窗被跳過。等於只評估「提前 15 天預知反轉」，沒有評估「反轉在 1–14 天後」的情況。建議改以「日期」為單位彙整（對每個真實反轉日，看模型在其前 k 天內是否發出訊號）。
3. **第 0 天的反轉偵測不到**：`get_first_trend_reversal_and_idx_signals` 從 i=1 開始比較，若視窗第 0 天與「今天」趨勢不同就漏掉。
4. **回測缺乏基準與風險指標**：沒有 Buy & Hold、沒有年化報酬、Sharpe、最大回撤；1 股為單位使損益以絕對金額表示，很難與其他策略比較。
5. **成交價假設**：訊號當天收盤成交。實務上訊號要等收盤後才知道，應以隔日開盤價成交。

---

## 🟡 9. 資料期間與 README 不符

`^VIX3M`（3M Volatility）在 Yahoo 的歷史比 2001 短，`CleanerMissingValue('auto')` 會刪除開頭所有含 NaN 的列。由 `val_summary.json` 的日期回推，有效起點約為 2006 年中，而非 README 所寫的 2001 年（推導見 01 §4）。若要使用 2001 年起的資料，需要拿掉 VIX3M 或改用 `^VIX`（1990 起）。

> 本次環境無法連線 Yahoo（HTTP 403），以上為由輸出結果反推，建議重跑時印出 `dataset.index.min()` 確認。

## 🟡 10. 標籤邊界效應

- `argrelextrema(..., order=20)` 在序列頭尾 20 天內的判斷不完整；資料最後一段（測試集末端）的 Trend 實際上是 ffill 出來的，可能不正確。
- 訓練/驗證/測試之間沒有 gap：訓練段最後幾天的標籤依賴驗證段前 20 天的價格。建議切分時留 `look_back + predict_steps + order_days` 天的間隔。

## 🟡 11. 程式碼品質 / 相容性

| 問題 | 位置 |
|---|---|
| 依賴 pandas 2.x 的寫法（`fillna(method=)`、鏈式賦值、`Series[int]`、`.loc[int:int]` on DatetimeIndex），已在 pandas 3.0.5 驗證會拋出 TypeError / KeyError | `featureFactory.py:79,88,113-114,134,137,141`、`processorFactory.py:70,76`、evaluator 多處 `.iloc[idx] =` |
| `online_train_model` 解包數量錯誤（`run_training_epoch` 回傳 3 個值） | `model.py:248` |
| 為了一行 seed import TensorFlow | `main.py:8,30` |
| 推論沒有 `torch.no_grad()` / `model.eval()` | `main.py:71,106` |
| Postprocessor 就地修改傳入的 `y_preds` / `y_test` tensor | `postprocessor.py:22-43` |
| 沒有 `requirements.txt`；README 漏列 TA-Lib、scipy、seaborn | — |
| `__pycache__/`、`outputs/models/model.pth`、`log.txt` 被 commit | 建議加 `.gitignore` |
| 多個參數宣告但未使用（`online_*`、`filter`、`resample`、`shuffle`、`min_delta`、`data_update_mode`、`ma_days`、`trend_days`） | `parameters.json` |
| `LSTM_many2many` 與 `LSTM_many_to_many` 重複 | `modelFactory.py:74,658` |
| 沒有任何單元測試 | — |
