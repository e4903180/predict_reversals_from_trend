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
| [10_training_diagnostics.md](10_training_diagnostics.md) | 模型有沒有真的訓練到：學習曲線、能力測試、打亂標籤對照 |
| [09_model_review.md](09_model_review.md) | 21 個模型的設計審查、新增 DLinear / PatchTST / iTransformer / TSMixer、LLM 與基礎模型的評估 |
| [08_model_comparison.md](08_model_comparison.md) | **GRU / LSTM / Transformer 正確訓練後與規則、隨機、Buy & Hold 的比較（最終結論）** |
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

**建議下一步**（詳見 08 §7）：改正規化方式（加入相對均線等特徵）、改用較容易預測的反轉標籤、walk-forward 驗證，並以均線規則為基準只學殘差。

## 注意事項

- 最初的靜態分析之後，已在 2026-10-09 實際重跑，結果與 repo 中的 `outputs/` 幾乎一致，見 [06_reproduction.md](06_reproduction.md)。
- 原程式需要 pandas 1.x；已做最小的相容性修正，可在 pandas 2.3 上執行（見 06）。
