# 04 實驗歷史與結果解讀

## 1. Git 時間線

| Commit | 日期 | 內容 |
|---|---|---|
| `4b10b5d` | — | 初始 README |
| `dd800b2` | 2024-05-20 | **舊版**：Keras / PyTorch 雙實作（`*_keras.py` / `*_pytorch.py`）、10 個 `DNN_Projects_*` 實驗資料夾（共 1,564 檔、108 組有 `summary.json`）、批次執行腳本 `reversePrediction*.py`、結果分析 notebook `result.ipynb` |
| `f8c6135` | 2024-09-13 | **重構版**（目前的程式）：模組化成 preprocessor / model / postprocessor / evaluator，移除所有舊實驗資料夾，新增 20 種模型架構、四種回測策略 |
| `0ee6949` | 2025-05-27 | 新增英文 README |

`progress.txt`（目前版）顯示 2024-09 之後還跑過 `DNN_Projects_slide1_20_64_16_GRU_targets_2/...` 系列實驗，標的包含 **AAPL、AMZN、^IXIC**（GRU、lr=1e-5、look_back 64、predict 16、dropout 0.2），但這些結果**沒有被 commit**。若論文最終結果來自這批實驗，需要從當時的電腦找回。

## 2. 舊版實驗（`dd800b2`，2024-05）

全部 108 組都是 **`DNN_5layers`**（1920→1024→512→256→128→64 的全連接網路 + LayerNorm，輸入 64 天 × 30 特徵攤平），資料 2001–2021、切分 70/10/20、預測 32 天、測試滑動步長 32。完整指標見 [data/legacy_experiments_2024-05.csv](data/legacy_experiments_2024-05.csv)。

> ⚠ 這批實驗的 `feature_cols` 都包含目標欄位 `Trend`（未來資訊洩漏），且測試集包含驗證集（見 03 §4）。以下數字僅供了解「當時探索了什麼」，不代表模型真實能力。

### 2.1 各實驗群組

| 資料夾 | 組數 | 掃描的變數 | 趨勢 Acc 平均 / 最高 | ROC-AUC 平均 |
|---|---|---|---|---|
| `DNN_Projects` | 12 | 標籤方法（LocalExtrema / MA）× 反轉後權重（1,2,5,10,15,20） | 0.549 / 0.717 | 0.532 |
| `DNN_Projects_l2` | 12 | 同上 + L2 | 0.554 / 0.722 | 0.537 |
| `DNN_Projects_l2_xavier` | 12 | 同上 + Xavier 初始化 | 0.558 / 0.722 | 0.541 |
| `DNN_Projects_l2_xavier_learningRate` | 8 | 學習率 1e-6 ~ 1e-3 | 0.607 / 0.722 | 0.575 |
| `DNN_Projects_l2_xavier_learningRate_epoch` | 4 | 學習率（更多 epoch） | 0.676 / 0.722 | 0.619 |
| `DNN_Projects_weights_learningRate` | 18 | 權重 × 學習率 | 0.590 / 0.724 | 0.554 |
| `DNN_Projects_remake` | 30 | 重跑（seed 0–5） | 0.602 / 0.722 | 0.562 |
| `DNN_Projects_MA` | 5 | MA 天數 10–50 | 0.699 / 0.788 | 0.644 |
| `DNN_Projects_steps` | 6 | look_back × predict_steps | 0.616 / 0.640 | 0.564 |
| `DNN_Projects_drop` | 1 | dropout 0.6 | 0.664 | 0.596 |

整體（108 組）：趨勢 Acc 中位數 **0.603**、ROC-AUC 中位數 **0.577**、預測反轉日與實際差距平均 **41.8 天**。

### 2.2 觀察

1. **反轉後權重越大越差**：`weight_after_reversal` = 1 / 2 / 5 / 10 / 15 / 20 時，平均趨勢 Acc 為 0.690 / 0.654 / 0.605 / 0.504 / 0.465 / 0.471。加重反轉後的樣本並沒有幫助（這可能是目前版本把權重設回 1 的原因）。
2. **MA 天數的取捨**：MA 越長，標籤越平滑、趨勢越好預測（MA-50 Acc 0.788、AUC 0.716），但反轉日的誤差也越大（MA-10 平均差 24 天 → MA-50 差 110 天）。**趨勢準確率與反轉時機準確度是互相衝突的目標**，這是本研究的核心張力。
3. **LocalExtrema 標籤比 MA 標籤難預測**（Acc 0.510 vs 0.588），但 LocalExtrema 才是真正的價格高低點。
4. **`DNN_Projects_steps` 的 look_back 掃描無效**：資料夾名稱寫 64 / 128 / 256，但 `parameters.json` 的 `look_back` 全都是 64，所以三組結果完全相同。只有 predict_steps（32 vs 64）真的有變化：32 天較好（0.640 vs 0.593）。
5. **學習率 1e-5 平均最好**（Acc 0.655），這大概是目前版本選用 1e-5 的原因；但重構後的模型與訓練輪數不同，這個選擇不一定適用。
6. 舊版 `log.txt` 中的 early stopping（epoch 215、245、474、117）受到 03 §5 rollback bug 影響。

## 3. 目前 `outputs/`（`f8c6135` 版，TransformerModel）

設定：Transformer（22,032 參數）、10 epochs、lr 1e-5、train 用 4 個指數、val/test 用 ^GSPC。執行時間 25.8 秒。

| 指標 | 驗證集（約 2015-04 – 2018-10） | 測試集（約 2018-10 – 2023-12） | 可信度 |
|---|---|---|---|
| ROC-AUC（逐日趨勢） | 0.550 | **0.658** | ⚠ 部分可信：logits 第 2–16 天被後處理覆寫 |
| PR-AUC | 0.359 | 0.437 | ⚠ 同上 |
| 趨勢 Accuracy | 0.316 | 0.293 | ✗ logits 沒閾值化，等於全部猜 downtrend |
| 反轉三分類 Accuracy / F1 | 1.0 | 1.0 | ✗ 真實 vs 真實 |
| 預測到的反轉數 | 0 | 0 | ✗ logits 沒閾值化 |
| 多空翻轉回測（1 股，起始 10,000） | +3,694（18/19 勝） | +12,808（25/26 勝） | ✗ 用真實訊號，是 oracle 上限 |
| 只做多回測（1 股，起始 100,000） | +2,249 | +6,990 | ✗ 同上 |

**解讀**：
- 唯一能大致反映模型能力的是測試集 ROC-AUC 0.658 —— 表示模型的 logits 對「隔天是漲勢或跌勢」有一些排序能力，但距離實用還很遠；驗證集只有 0.550，代表結果不穩定。
- 回測數字可以當作「若能完美預測反轉」的**理論上限**：5 年 26 次多空翻轉、每次 1 股，獲利約 12,800 美元。同期 S&P 500 約從 2,500–2,900（視起點而定）漲到 4,770，Buy & Hold 1 股約 +1,800 ~ +2,300；這個差距說明「準確預測反轉」的價值很高，也解釋了研究動機。
- 從訓練 log 看，模型在驗證集上的逐日準確率 0.683 ≈ 驗證集 uptrend 比例 0.684，基本上只學到「猜漲」。

## 4. 輸出圖表（`outputs/plots/`）

| 檔案 | 內容 | 目前是否有意義 |
|---|---|---|
| `training_curve.png` | train/val loss & acc | ✓ |
| `online_training_curve.png` | 空（線上學習未啟用） | — |
| `trend_confusion_matrix.png` | 逐日趨勢混淆矩陣 | ✗（bug 3） |
| `reversal_confusion_three_type_matrix.png`、`pass_reversal_confusion_matrix.png` | 反轉分類 | ✗（bug 2） |
| `reversal_confusion_type_matrix.png` | 真實有反轉的視窗中，預測類型 | ✗（bug 3：預測全為 No reversal） |
| `pred_days_difference_bar_chart.png` | 反轉日差距 | ✗（無預測反轉） |
| `roc_pr_curve.png` | ROC / PR | ⚠ 部分 |
| `trading_details_*_kbar.png` | K 線 + 買賣點 | ✗ 為真實訊號（但可作為「理想買賣點」示意圖） |
