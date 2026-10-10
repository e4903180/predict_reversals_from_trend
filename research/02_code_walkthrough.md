# 02 程式碼解析

程式總量約 3,850 行 Python（不含 notebook）。

| 檔案 | 行數 | 角色 |
|---|---|---|
| `main.py` | 141 | 主流程 `ReversePrediction.run` |
| `preprocessor/preprocessor.py` | 232 | 下載資料、套特徵、切分、合併多指數 |
| `preprocessor/featureFactory.py` | 667 | 25 種特徵 + Trend 標籤（Factory 模式） |
| `preprocessor/processorFactory.py` | 190 | 缺值清洗、時間切分、滑動視窗 + 視窗內正規化 |
| `model/model.py` | 265 | 訓練迴圈、加權 BCE、early stopping、(未使用的) online training |
| `model/modelFactory.py` | 1151 | 20 種網路架構 |
| `postprocessor/postprocessor.py` | 253 | 趨勢序列 → 反轉點 → 交易訊號 |
| `evaluator/evaluator.py` | 1092 | 混淆矩陣、ROC/PR、天數差、四種回測、繪圖、JSON 輸出 |

---

## 1. `main.py`

```
set_seed(42)                         # numpy / tensorflow / random / torch
Preprocessor(params).get_datasets()  # → X/y train/val/test, dates, target_dataset(^GSPC 原始 DataFrame)
DataLoader(batch_size=32)            # ⚠ 寫死，忽略 params['batch_size']
Model(params).train_model(...)       # → model, history
torch.save(model, ...)
# 驗證集與測試集各跑一次：
y_preds = model(X)                   # ⚠ 未包 torch.no_grad()、未 model.eval()（Transformer dropout=0 所以影響不大）
Postprocessor.postprocess_predictions(y_preds, y, dates, target_dataset)
Evaluator.evaluate_and_generate_results(...)  → json.dump
```

- 只為了 `tf.random.set_seed` 而 import 整個 TensorFlow（`main.py:8,30`），是舊版 Keras 實作的殘留，可移除。
- `online_training_history` 傳入空 dict，線上學習功能在此版已停用。

## 2. Preprocessor

### 2.1 `get_datasets()`（`preprocessor.py:133`）

1. 對 `train_indices` 中每個指數：下載 → 依序套用 `features_params` → 清洗 → `process_data` 取**訓練段**視窗。
2. 對 `test_indices`（`['^GSPC']`）再下載一次、處理，取**驗證段與測試段**視窗。
   - 注意 `get_symbol_dataset(test_indices)` 傳入的是 list，`yf.download(['^GSPC'])` 在新版 yfinance 會回傳 MultiIndex 欄位。
3. 多指數訓練視窗依「第 i 個樣本 × 指數」交錯合併，長度取最短者（`preprocessor.py:213-216`）。
4. 若 `filter_reverse_trend_train_test`，每個 16 天標籤在第一次反轉後全部設為新值（每個視窗最多一次反轉）。

### 2.2 `ProcessorFactory`

- `split_datasets`：純時間順序切分，**沒有 gap**。訓練段最後幾天的 Trend 標籤依賴驗證段前 20 天的價格（局部極值的 order=20），邊界有輕微洩漏。
- `standardize_and_split_data`：`for i in range(0, len - 64 - 16 + 1, slide)`，每個 64×32 視窗各自 `MinMaxScaler().fit_transform`。
- `CleanerMissingValue.clean('auto')`：`while data.iloc[0].isnull().any(): data = data.iloc[1:]` → 只要任一欄（例如 `^VIX3M`、Treasury yield）開頭缺值，就整列刪除，直到所有欄都有值。

### 2.3 `FeatureFactory`

- 每個特徵都是 `FeatureBase` 子類別，`compute(data, **kwargs)` 直接在 DataFrame 新增欄位並回傳。
- 外部序列（`^IRX ^FVX ^TNX ^TYX ^VIX3M`）以 `data[col] = series` 依日期對齊；不同交易日曆的缺口由後續 ffill 補上。
- `IndicatorTrend` 的 `ma_days`、`trend_days` 參數目前未被使用（`method` 只支援 `LocalExtrema`；舊版有 `MA` 方法）。
- 使用了 pandas 2.x 才允許的寫法：`data['Close'][int]`（位置索引 fallback）、`data['Local Max'].iloc[...] = ...`（鏈式賦值）、`fillna(method='ffill')`。在 pandas 3.x 會失敗或靜默無效。

## 3. Model

### 3.1 `model/model.py`

- Loss：`BCEWithLogitsLoss`，可選依「反轉前/後」給不同權重（`apply_weights`，目前權重皆 1）。
- `binary_accuracy`：`round(sigmoid(logits)) == y` 的逐元素平均（16 天 × batch）。
- `early_stopping`：記錄 `best_model = model.state_dict()` —— 這是**參考**，之後權重更新會一起變，rollback 等於沒 rollback（見 03）。
- `online_train_model`：呼叫 `run_training_epoch` 只接兩個回傳值，但該函式回傳三個 → 若被呼叫會直接報錯（目前沒被呼叫）。
- `weight_decay` 參數未傳給 Adam。

### 3.2 `model/modelFactory.py`（20 種架構）

所有 RNN 類模型的 hidden size 都等於特徵數（32），單層，參數量很小。

| 系列 | many-to-one（取最後時間步） | many-to-many（攤平全部時間步再接 FC） |
|---|---|---|
| RNN | `GRU`, `LSTM`, `BiLSTM` | `GRU_many_to_many`, `LSTM_many_to_many`, `BiLSTM_many_to_many` |
| Attention RNN | `AttentionLSTM`, `AttentionBiLSTM` | 對應的 `_many_to_many` |
| CNN | `CNN`（Conv1d + MaxPool → FC）, `CNN_LSTM` | 對應的 `_many_to_many` |
| TCN | `TCN`（標準 TemporalBlock + Chomp1d） | `TCN_many_to_many` |
| Attention | `SelfAttention`, `TransformerModel` | 對應的 `_many_to_many` |

`TransformerModel`（目前使用）的特點：
- `nn.Transformer(d_model=32, nhead=1, encoder=decoder=1 層, dim_feedforward=64)`，`forward` 是 `transformer(x, x)` —— 同一序列同時當 src 與 tgt。
- **沒有位置編碼、沒有 causal mask**：self-attention 對時間順序是置換不變的，模型只能靠「取最後一個位置」的輸出來間接得知順序，時間結構資訊大部分遺失。
- 總參數量 22,032（見 `summary.json` 的 `model_summary`），偏小。

> 另有 `LSTM_many2many`（`modelFactory.py:74`）與 `LSTM_many_to_many` 重複，前者未註冊在 factory。

## 4. Postprocessor

`postprocess_predictions(y_preds, y_test, test_dates, target_dataset)`：

| 步驟 | 函式 | 說明 |
|---|---|---|
| 1 | `change_values_after_first_reverse_point(y_preds)` | 每列在第一次「值改變」後全部設為該值。**直接作用在 logits 上**，連續值幾乎在 i=1 就「改變」，結果是 `[p0, p1, p1, ..., p1]`（可在 `summary.json` 的 `y_preds` 看到）。也會**就地修改**傳入的 tensor。 |
| 2 | `get_first_trend_reversal_and_idx_signals` | 找第一個 `1→0`（Valley, -1）或 `0→1`（Peak, +1），回傳訊號與位置。用 `== 0/1` 判斷，對 logits 永遠找不到。 |
| 3 | `get_trade_signals` | Peak→Sell、Valley→Buy，放在 `test_dates[idx][reverse_idx]` 那一天。 |
| 4 | `compare_reverse_predictions` | 逐視窗比對預測與真實的反轉類型、位置差、是否落在 ±5 天。 |
| 5 | `calculate_reversal_dates_with_signals(y_test...)` | 走訪視窗，遇到反轉就跳過 `reverse_idx` 個視窗，避免同一反轉重複計算。回傳的 `valid_signals` 用來篩選 `reverse_difference`。 |
| — | `passing_trade_signals` | **由 `y_test` 產生**（`postprocessor.py:232`），與 `test_trade_signals` 完全相同。命名暗示原意是「通過過濾的預測訊號」，但實際傳的是真實值。 |

> 備註：因為步驟 5 每遇到一個反轉就從該視窗開始計算，而視窗是逐日滑動的，真實反轉第一次出現時幾乎都在視窗最後一格（`actual_reverse_idx` 多為 15）。也就是說反轉評估實際上是在問「能否提前 15 天預知反轉」，這是最難的設定。

## 5. Evaluator

`evaluate_and_generate_results` 依序呼叫：

| 方法 | 輸入（實際傳入的東西） | 備註 |
|---|---|---|
| `evaluate_trend_predictions` | `y_test_indices` vs `y_preds_indices`（logits） | 用 `== 0` 判斷 uptrend，logits 永遠非 0 → 全判為 downtrend |
| `evaluate_reversals_predictions_three_type` | `reversals_test` vs `reversals_pred_pass` | **兩者都源自 y_test** → 恆為 1.0 |
| `evaluate_reversal_type_predictions` | `reverse_difference` 的 actual vs predicted label | 只看「真實有反轉」的視窗 → 不計 false positive |
| `evaluate_reversals_predictions_two_type` | 同 three_type | 恆為 1.0 |
| `plot_roc_pr_curve` | `y_test` vs `y_preds`（被步驟 1 修改過的 logits） | logits 可直接算 AUC（單調轉換不影響），但第 2–16 天的值已被覆寫 |
| `plot_days_difference_bar_chart` | `reverse_difference` | |
| `execute_trades*` ×4 | `pass_pred_trade_signals`（= 真實訊號） | 見下 |

**回測規則**（皆以收盤價成交，單位為 1 股）：

- `execute_trades`：Sell 時若無部位則放空 1 股，若有多單則賣出全部再放空 1 股；Buy 反之 → 永遠持有 ±1 股的多空翻轉策略。初始現金 10,000。
- `execute_trades_with_stop_loss`：加 10% 停損。
- `execute_trades_with_stop_loss_stop_win`：加 10% 停損與 10% 停利。
- `execute_trades_long_only`：只做多，初始現金 100,000。
- 手續費 `max(shares × 0.000008 × price, 0.01)`；沒有滑價、沒有放空成本；報酬以絕對金額計，沒有年化報酬、Sharpe、最大回撤，也沒有 Buy & Hold 基準。
