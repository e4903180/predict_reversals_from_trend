# 05 建議與後續路線圖

依優先順序排列。前兩個階段完成之前，任何模型比較的結論都不可靠。

## 階段 0：讓程式能重跑（約半天）

1. 建立 `requirements.txt` 並鎖版本（建議 Python 3.10、`pandas<2.2`、`numpy<2`、`torch`、`yfinance`、`TA-Lib`、`scipy`、`scikit-learn`、`matplotlib`、`seaborn`、`tqdm`），或改寫 pandas 3 不相容的寫法：
   - `fillna(method='ffill')` → `.ffill()`
   - `data['Close'][i]` → `data['Close'].iloc[i]`
   - `data['Local Max'].iloc[idx] = ...` → `data.iloc[idx, data.columns.get_loc('Local Max')] = ...`
   - `data.loc[prev_idx:idx, 'Trend']`（整數位置在 DatetimeIndex 上）→ `data.iloc[prev_idx:idx+1, col]`
   - evaluator 中大量 `df['col'].iloc[idx] = ...` 鏈式賦值
2. **把下載的原始資料快取成 CSV/Parquet**（例如 `data/raw/^GSPC.csv`）並納入版控或另存。Yahoo 的歷史資料會被修正、代號會下架（如 `^VIX3M`、`^RUA` 可能變動），沒有快取就無法重現。
3. 移除 TensorFlow 依賴；加 `.gitignore`（`__pycache__/`、`outputs/models/`、`log.txt`）。

## 階段 1：修正會讓結論錯誤的 bug（約 1 天）

對應 03 的 🔴 項目：

```python
# main.py（概念示意）
model.eval()
with torch.no_grad():
    logits = model(X_test)
probs = torch.sigmoid(logits)
y_preds_bin = (probs > threshold).float()          # threshold 在驗證集上調
postprocess_results = postprocessor.postprocess_predictions(y_preds_bin.clone(), y_test.clone(), ...)
roc_auc, pr_auc = evaluator.plot_roc_pr_curve(y_test, probs, ...)   # 用未修改的機率
```

```python
# postprocessor.py
passing_trade_signals = self.get_trade_signals(y_preds_reverse_signals, y_preds_reverse_idx, ...)  # 改用預測
filtered_pred_reversal_dates, _, _ = self.calculate_reversal_dates_with_signals(
    y_preds_reverse_signals, y_preds_reverse_idx, ...)                               # 預測端
```

- `early_stopping`：`copy.deepcopy(model.state_dict())`，訓練結束一律載回最佳權重。
- `DataLoader(batch_size=params['batch_size'])`；Adam 加上 `weight_decay`。
- 為後處理與 evaluator 寫幾個小單元測試（例如手造 `[0,0,1,1]` → Peak at idx 2），避免再出現「真實 vs 真實」這類問題。

## 階段 2：建立可信的評估框架（約 2–3 天）

1. **基準線（baselines）**，所有模型都要跟這些比：
   - 永遠猜多數類（uptrend）
   - 「明天的趨勢 = 今天已知的趨勢」（persistence；注意今天的 Trend 標籤本身需要未來 20 天才能確定，所以要用「最後一個已確認的極值」推出的趨勢）
   - 簡單規則：MA(20/50) 交叉、RSI 超買超賣、MACD 交叉
   - 邏輯迴歸 / LightGBM（輸入最後一天的特徵即可）
2. **反轉事件層級的評估**：以「真實反轉日」為單位，看模型在反轉前 k 天內是否發出正確類型的訊號（命中率、提前天數分佈），同時計算 false alarm 率（每年誤報次數）。取代目前只看 idx 15 的做法。
3. **回測**：
   - 隔日開盤價成交、加入滑價與放空成本
   - 以資金比例而非 1 股下單；報告年化報酬、Sharpe、最大回撤、交易次數、勝率
   - 永遠同時列出 Buy & Hold 與 oracle（真實訊號）上限，模型落在兩者之間的哪裡才是重點
4. **Walk-forward / 滾動視窗驗證**：目前只有一次 50/20/30 切分，測試期 2019–2023 包含 COVID 與 2022 熊市等特殊行情。建議做多個時間折（例如每年滾動），報告平均與變異。
5. 切分時加入 `look_back + predict_steps + order_days` 的 gap，避免邊界洩漏。
6. 多個 seed（≥5）並報告平均 ± 標準差。

## 階段 3：模型與方法改進（研究延伸）

1. **Transformer 加位置編碼**，改為只用 encoder（`nn.TransformerEncoder`），或採用時間序列專用架構（PatchTST、TFT、iTransformer 等）。
2. **訓練設定**：學習率 1e-3 ~ 1e-4 搭配 scheduler、epoch 50–200 + 正確的 early stopping；hidden size 不要綁定特徵數。
3. **正規化方式**：視窗內 MinMax 會抹掉「目前價位相對長期的位置」。可以改用報酬率 / log return、z-score（用訓練期統計量）、或同時保留視窗內與全域兩種正規化。
4. **標籤設計**：
   - 研究 MA 天數 / order_days 對「可預測性 vs 時機精度」的取捨（舊實驗已看到明顯張力，見 04 §2.2），畫成 trade-off 曲線會是很好的論文圖表。
   - 考慮 triple-barrier labeling（López de Prado）或 ZigZag（以漲跌幅閾值而非天數定義反轉）。
   - 直接預測「距離下一次反轉的天數」（回歸 / 存活分析）作為另一種問題轉換。
5. **損失函數**：反轉附近的時間步使用 focal loss 或依「距反轉天數」加權，比起目前的「反轉後權重」（舊實驗顯示有害）更有針對性。
6. **特徵**：加入 `^VIX`（1990 起，可取代 VIX3M 拿回 2001–2006 的資料）、市場廣度、期限利差（10Y−3M）、信用利差等；做特徵重要性（permutation / SHAP）分析，32 個特徵中有大量高度共線（Open/High/Low/Close/MA/Bollinger/SAR）。
7. **多標的**：progress.txt 顯示曾嘗試 AAPL / AMZN / ^IXIC，可延伸為「在多個指數上訓練、在未見過的指數上測試」的泛化實驗。

## 建議的第一個具體步驟

如果只有一點時間，最有價值的是：

1. 修正 03 的 bug 1–3（約 20 行程式）
2. 用 GRU 與 Transformer（加位置編碼）各跑一次，lr=1e-4、50 epochs
3. 與「永遠猜 uptrend」和 MA 交叉策略比較 AUC 與回測

這樣就能在一天內回答最根本的問題：**這個方法到底有沒有贏過簡單基準？** 之後再決定要往哪個方向延伸。
