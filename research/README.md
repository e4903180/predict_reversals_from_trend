# 研究文件：predict_reversals_from_trend 專案分析

> 分析日期：2026-10-09
> 分析對象：`main` 分支最新 commit `0ee6949`（2025-05-27），以及 git 歷史中的舊版實驗 `dd800b2`（2024-05-20）、`f8c6135`（2024-09-13）

本資料夾是對「以深度學習預測股價趨勢反轉點——以 S&P 500 為例」（成功大學碩士研究，林承平，指導教授：吳謂勝）這個專案的完整回顧。目的是讓多年後重新接手的人能快速理解：**它在做什麼、程式怎麼運作、目前結果能不能信、下一步該做什麼**。

## 文件索引

| 檔案 | 內容 |
|---|---|
| [01_project_overview.md](01_project_overview.md) | 研究動機、問題定義、整體流程、資料與參數 |
| [02_code_walkthrough.md](02_code_walkthrough.md) | 逐模組程式碼解析（preprocessor / model / postprocessor / evaluator） |
| [03_issues_and_bugs.md](03_issues_and_bugs.md) | 發現的錯誤、資料洩漏與方法論問題，依嚴重度排序，附程式行號與證據 |
| [04_experiment_history.md](04_experiment_history.md) | git 歷史、舊版 108 組實驗彙整、目前 `outputs/` 結果解讀 |
| [05_recommendations.md](05_recommendations.md) | 修正與後續研究路線圖 |
| [**13_multi_index_validation.md**](13_multi_index_validation.md) | **在 4 個指數上驗證（381 個反轉）：反轉預測的優勢跨市場成立；交易優勢沒有重現（最新結論）** |
| [11_v2_study.md](11_v2_study.md) | **v2 研究總結：1993–2023 資料、hazard 輸出頭、walk-forward 8 個時段、顯著性檢定、反轉警報策略** |
| [LOG.md](LOG.md) | 研究日誌 |
| [10_training_diagnostics.md](10_training_diagnostics.md) | 模型有沒有真的訓練到：學習曲線、能力測試、打亂標籤對照 |
| [09_model_review.md](09_model_review.md) | 21 個模型的設計審查、新增 DLinear / PatchTST / iTransformer / TSMixer、LLM 與基礎模型的評估 |
| [12_architecture_grid.md](12_architecture_grid.md) | 舊流程中 8 種架構（含 PatchTST、iTransformer、TSMixer、DLinear）× 3 seeds 的比較 |
| [08_model_comparison.md](08_model_comparison.md) | GRU / LSTM / Transformer 正確訓練後與規則、隨機、Buy & Hold 的比較 |
| [results/](results/) | 實驗結果 CSV |
| [07_fixed_rerun.md](07_fixed_rerun.md) | **修正 🔴 錯誤後重跑的真實結果**與基準比較 |
| [06_reproduction.md](06_reproduction.md) | 2026-10 實際重跑的環境、資料與結果比對（結果已重現） |
| [scripts/](scripts/) | 指標對照與基準計算腳本 |
| [data/raw/](data/raw/) | 下載快取的原始價格資料（9 個代號） |
| [data/legacy_experiments_2024-05.csv](data/legacy_experiments_2024-05.csv) | 從 git 歷史抽出的 108 組舊實驗指標 |
| [data/extract_legacy_experiments.py](data/extract_legacy_experiments.py) | 產生上述 CSV 的腳本（在 repo 根目錄執行 `python3 research/data/extract_legacy_experiments.py > out.csv`） |

## 執行摘要（TL;DR）

**研究構想是合理且有價值的**：反轉點是極稀有事件（嚴重類別不平衡），改為預測「未來 16 天每天的趨勢（漲/跌）」，再從趨勢序列的變化推出反轉點，是個聰明的問題轉換。程式架構（Factory 模式、可插拔的特徵/模型、參數全部放在 JSON）也相當乾淨。

**原始 repo 裡的結果不能採信**，因為有幾個關鍵錯誤（1–3 與 5 的 early stopping 已修正，見 07）：

1. 🔴 **回測使用真實標籤而非模型預測**：`postprocessor.py:232` 的 `passing_trade_signals` 是由 `y_test`（未來真實的趨勢）產生，而 `evaluator` 所有回測都用它。因此 `outputs/` 的交易結果（26 筆交易 25 筆獲利）其實是「完美預知未來」的上限，不是模型表現。
2. 🔴 **反轉三分類矩陣是「真實 vs 真實」**：`reversal_confusion_three_type_matrix` / `pass_reversal_confusion_matrix` 兩邊傳入的都是由 `y_test` 產生的資料，所以 Accuracy/F1 全為 1.0。
3. 🔴 **模型輸出的 logits 從未轉成 0/1**：模型輸出 logits（使用 `BCEWithLogitsLoss`），後處理卻直接拿 logits 判斷 `== 0` / `== 1`，導致「預測的反轉」永遠為 0、趨勢混淆矩陣把所有預測都當成 downtrend（Accuracy 0.29 = 測試期 downtrend 比例）。
4. 🟠 **舊版實驗（2024-05 的 108 組）把目標欄位 `Trend` 放進輸入特徵**，而 `Trend` 是用未來 20 天價格決定的局部極值計算的 → 資料洩漏；且舊版 `X_test` 包含了驗證集。
5. 🟠 Early stopping 存的是 `state_dict()` 的參考而非複本，rollback 無效；`batch_size` 參數被寫死的 32 覆蓋；Transformer 沒有位置編碼。
6. 🟡 README 寫資料期間 2001–2023，但 `^VIX3M` 歷史較短，`auto` 清洗會把開頭有 NaN 的列全部刪掉，實際有效起點為 2006-07-17（已實測確認，詳見 06）。

**修正後重跑的結果（見 07）**：🔴 1–3 已修正（commit 於本分支）。使用同一組模型權重，正確評估後：測試集 ROC-AUC 0.659 → **0.543**、趨勢準確率 0.712（「永遠猜漲」為 0.707）、26 個真實反轉只抓到 1 個、回測從 +12,808 變成 **−7,509**（Buy & Hold +1,172）。原本的 0.659 恰好等於「把第 2 天的預測複製到全部 16 天」的 AUC。目前模型實質上只會猜漲；唯一的正面訊號是未來第 1–2 天的 AUC 約 0.74–0.77（測試集）。

**模型比較的結論（見 08）**：以 lr 1e-4、50 epochs、自動閾值、固定 1 股部位重新訓練 GRU / LSTM / Transformer / 加位置編碼的 Transformer（各 3 個 seed）。最好的 GRU 測試 AUC 0.649，但只和「收盤價相對 20 日均線」這條一行規則打平（0.627；第 1–2 天 0.780 vs 0.779；驗證集上規則反而較好）。反轉 26 個抓到約 6 個，精確率約 5%。回測無法穩定勝過 Buy & Hold 或隨機進出場。`parameters.json` 的預設值已改為 GRU 設定。

**v2 研究的最終結論（見 [11](11_v2_study.md)、[12](12_architecture_grid.md)、[LOG](LOG.md)）**：資料延長到 1993–2023，改用跨視窗可比的特徵、hazard 輸出頭，以 walk-forward 8 個時段（2008–2023）搭配 bootstrap 檢定：

- **趨勢**：沒有任何模型或架構（共 12 種）在多個時段都勝過「收盤價相對 MA20」規則。第 6 天以後所有方法都接近隨機。
- **反轉**：**hazard 輸出頭的 GRU 在 8/8 個時段勝過基準**（10 天事件 AUC 0.594 vs 0.521，AP 0.304 vs 0.236）；集成後相對 logreg 的差距達統計顯著（p = 0.018）。效果一致，但幅度小。
- **貢獻最大的改動**：hazard 頭（+0.037）> 多指數訓練（+0.028）> 不含未來資訊的已確認趨勢特徵（+0.022）。更新的架構、正則化、總經特徵、國際資料、LightGBM 都沒有幫助。
- **多指數驗證（[13](13_multi_index_validation.md)，381 個反轉）**：GRU-hazard 的反轉預測在 32 格（4 個指數 × 8 個時段）中勝過基準 27/32（5 天，p = 0.0001）、25/32（10 天，p = 0.001）；集成的合併 bootstrap p = 0.001。**訊號真實且跨市場一致，但很微弱**（AUC 約 0.58–0.60）。
- **交易**：^GSPC 上的「波峰警報當天空手」（Sharpe 0.55 vs 0.50）**沒有在 ^IXIC / ^DJI / ^RUT 上重現**；回撤降低主要來自 2008–09。反轉訊號目前不足以直接轉化為交易優勢。

**下一步建議**見 [13 §6](13_multi_index_validation.md)：把反轉機率用於調整部位大小（風險控管）而非買賣、檢驗不同的反轉定義（order 10/30、ZigZag）、加入價格以外的資訊（新聞、選擇權）。

## 注意事項

- 最初的靜態分析之後，已在 2026-10-09 實際重跑，結果與 repo 中的 `outputs/` 幾乎一致，見 [06_reproduction.md](06_reproduction.md)。
- 原程式需要 pandas 1.x；已做最小的相容性修正，可在 pandas 2.3 上執行（見 06）。
