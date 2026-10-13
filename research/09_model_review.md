# 09 模型設計審查與新模型（2026-10-09）

## 1. 保留並修正的模型

| 模型 | 為何保留 | 修正 |
|---|---|---|
| `GRU` | 目前表現最好也最穩定（08） | `hidden_size` 可在 `model_params.GRU` 設定（預設仍等於特徵數，舊結果可重現）；單層時不再傳 dropout 給 RNN |
| `LSTM` | 經典基準 | 同 GRU（`model_params.LSTM.hidden_size`） |
| `TCN` | 卷積類代表，因果卷積適合時間序列 | **修正 bug**：原本沒有把輸入轉成 (batch, 特徵, 時間)，卷積實際上沿「特徵順序」進行，現在改為沿時間軸 |
| `TransformerEncoderPE` | 取代 `TransformerModel`：有位置編碼、只用 encoder（08 中 AUC 0.631 > 0.603） | — |

## 2. 新增的模型（2023–2024 時間序列文獻中常用的強基準）

| 模型 | 出處 | 想法 | 為什麼適合這個問題 |
|---|---|---|---|
| `DLinear` | Zeng et al., AAAI 2023, *Are Transformers Effective for Time Series Forecasting?* | 移動平均分解成趨勢 + 殘差，各接一個沿時間軸的線性層 | 幾乎是線性模型。該論文證明它常常贏過 Transformer，可以檢驗「深度模型是否真的必要」 |
| `PatchTST` | Nie et al., ICLR 2023 | 每個特徵切成重疊的 patch 當作 token，channel-independent 的 Transformer | 目前最常被引用的強基準；patch 能降低雜訊、減少 token 數 |
| `iTransformer` | Liu et al., ICLR 2024 | 把「每個特徵的整段序列」當作一個 token，attention 學習特徵之間的關係 | 本研究有 32 個技術指標，重點可能在指標之間的組合，而不在時間位置 |
| `TSMixer` | Chen et al. (Google), TMLR 2023 | 全 MLP，交替沿時間與沿特徵混合 | 參數少、訓練快，在小資料上通常比 Transformer 穩 |

另外加入 **LightGBM**（每個預測天數各訓練一個分類器，輸入為視窗的最後一天、5 日平均、64 日平均與 5 日變化）作為非深度學習的強基準。2025–2026 年的研究指出，在日報酬預測上，boosting 模型常常勝過大型預訓練模型（見 §4）。

這 4 個新模型都是依論文精神寫的精簡實作（約 2–10 萬參數，配合本研究的 64×32 輸入與 16 天二元輸出），沒有完整重現原論文的所有細節，例如 RevIN 正規化和 PatchTST 的自監督預訓練。

## 3. 其他模型：只記錄問題，暫不修正

程式碼保留原樣，舊實驗仍可重現。

| 模型 | 問題 |
|---|---|
| `TransformerModel`、`TransformerModel_many_to_many` | 沒有位置編碼，模型分不出時間先後；把 `nn.Transformer` 的 encoder-decoder 當 encoder 用（`transformer(x, x)`，decoder 沒有 causal mask）；`d_model` 直接等於 32 個原始特徵，沒有投影；`dim_feedforward` 綁定 `look_back`；`nhead` 必須整除 32 |
| `SelfAttention` | 沒有位置編碼；只有單層 attention，沒有殘差、LayerNorm 和 FFN；取最後一步，等於以「最後一天」為 query 對 64 天做加權平均 |
| `SelfAttention_many_to_many` | 自製 Q/K/V 投影到 `look_back` 維（64），維度與時間長度耦合；同樣沒有位置編碼 |
| `BiLSTM`、`BiLSTM_many_to_many` | 取 `out[:, -1, :]`：反向那一半在最後一步只看過最後 1 天，幾乎沒用。應改用兩個方向的最終 hidden state `h_n`。`AttentionBiLSTM` 用 attention 池化，沒有這個問題 |
| `CNN`、`CNN_many_to_many` | depthwise 卷積（`groups=32`），特徵之間完全不交互；只有 1 層、kernel 3，感受野 3 天；本質上接近線性模型 |
| `CNN_LSTM`、`CNN_LSTM_many_to_many` | 卷積同樣是 depthwise，只有 3 天感受野；其餘結構合理 |
| `AttentionLSTM`、`AttentionBiLSTM`（含 `_many_to_many`） | 結構合理（attention 池化）；只有 hidden size 綁定特徵數、單層 dropout 無效等共通問題 |
| `LSTM_many_to_many`、`GRU_many_to_many` | 名稱誤導：不是 seq2seq，只是把 64 個時間步攤平接 FC；FC 有 2,048 個輸入，容易過擬合 |
| `LSTM_many2many` | 與 `LSTM_many_to_many` 重複，且沒有註冊在 factory |
| `TCN_many_to_many` | 有正確 permute ✅；攤平所有時間步接 FC（45k 參數） |
| 共通（所有原始 RNN） | hidden size 固定等於特徵數（32），只有 1 層；`dropout` 參數對單層 RNN 無效（PyTorch 會警告） |

## 4. 比架構更根本的問題：輸出頭和標籤結構不匹配

`filter_reverse_trend_train_test` 讓每個 16 天標籤都只有 `[a, a, …, a, b, …, b]` 這種形狀。也就是只帶「目前趨勢」和「第幾天反轉（或不反轉）」兩個資訊，共 32 種可能。所有模型卻都輸出 16 個獨立 logit，用逐格 BCE 訓練：

- 模型可以輸出標籤中不存在的形狀，例如 `[0,1,0,1,…]`，後處理只取第一個變化點。
- 「趨勢延續」的格子主導 loss，反轉那一格幾乎沒有權重。這和模型只學到短期動能的結果一致（08）。

更貼合的輸出設計：「目前趨勢（2 類）＋反轉日（16 類，含不反轉）」的分類頭，或用 hazard / 存活分析預測每天發生反轉的機率。**這可能比換架構更有幫助**，建議作為下一步。

## 5. 關於用 LLM 或時間序列基礎模型

| 方案 | 文獻證據 | 對本研究的風險 |
|---|---|---|
| **通用 LLM 直接看數字預測**（把價格序列寫成文字給 GPT / Claude / LLaMA） | Tan et al.（NeurIPS 2024）[*Are Language Models Actually Useful for Time Series Forecasting?*](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6ed5bf446f59e2c6646d23058c86424b-Abstract.html) 的消融實驗顯示：把 LLM 拿掉或換成簡單 attention，預測通常不變、甚至更好。2025 年的研究也發現 LLM 對漲跌**方向**的預測很差（[ICBBEM 2025](https://www.atlantis-press.com/proceedings/icbbem-25/126011834)） | ⚠ **資料污染**：測試期 2019–2023 的 S&P 500 走勢就在 LLM 的訓練資料裡，LLM 可能「記得」答案，結果無法當作預測能力的證據。另外成本高、速度慢 |
| **LLM 處理新聞 / 財報文字，產生情緒特徵**，再和技術指標一起餵給模型 | 有研究指出加入新聞情緒能改善預測（例如 [PICBE 2025 的研究](https://reference-global.com/article/10.2478/picbe-2025-0043)），但多半只和 ARIMA 這類弱基準比較 | 需要另外蒐集帶時間戳記的新聞資料；同樣要嚴格避免使用晚於預測日的資訊 |
| **時間序列基礎模型**（Chronos-2、TimesFM 2.5、Moirai 2.0、MOMENT） | 2025–2026 的金融基準研究顯示，零樣本 TSFM 對日報酬的預測力很弱：方向準確率接近 50%，且輸給 CatBoost / LightGBM（[Re(Visiting) TSFMs in Finance](https://arxiv.org/html/2511.18578v1)、[Pretrained TSFMs for Financial Return Forecasting, 2026](https://arxiv.org/pdf/2606.27100)） | 部分模型的預訓練語料可能也包含股價；需要從 Hugging Face 下載權重，目前環境的網路權限可能要再開放 |

**建議**：不要用 LLM 直接取代這裡的模型。資料污染問題會讓任何好結果都無法採信，而且文獻證據也不支持。比較值得嘗試的是：

1. 先完成 §4 的輸出頭改造和輸入正規化，這是目前的主要瓶頸。
2. 若想引入「語言模型」，最合理的是 **LLM 產生新聞情緒特徵**（需要新聞資料，並以發布時間嚴格對齊）。
3. 若想試基礎模型，可以拿 **Chronos-2 或 TimesFM 2.5 做零樣本**：預測未來 16 天價格，再用相同的局部極值規則換算成趨勢，和 GRU、LightGBM 放在同一張表比較。測試期最好用 2024 年以後的資料，以降低污染風險。

其他參考：[Is Mamba Effective for Time Series Forecasting?](https://arxiv.org/pdf/2403.11144)、[Deep Learning for Financial Time Series: A Large-Scale Benchmark (2026)](https://arxiv.org/pdf/2603.01820)、[A Survey of Deep Learning and Foundation Models for Time Series Forecasting](https://arxiv.org/pdf/2401.13912)。
