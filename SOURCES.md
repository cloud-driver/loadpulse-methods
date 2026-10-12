# 程式與方法來源對照

## 核對版本

**B1**：Giacomini, R., & Rossi, B. (2009). Detecting and Predicting Forecast Breakdowns. The Review of Economic Studies, 76(2), 669–705。

實際公式核對：ECB Working Paper 638，June 2006，49 頁 PDF。以下頁碼是印刷頁，不是期刊頁碼。官方全文：https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp638.pdf 。正式書目入口：https://www.jstor.org/stable/40247617 。

**B2**：Hansen, P. R., Lunde, A., & Nason, J. M. (2011). The Model Confidence Set. Econometrica, 79(2), 453–497。DOI：10.3982/ECTA5771。

實際公式核對：CREATES Research Paper 2010-76，文內 March 2010，40 頁 PDF。以下頁碼是工作論文印刷頁。全文：https://repec.econ.au.dk/repec/creates/rp/10/rp10_76.pdf 。期刊入口：https://www.econometricsociety.org/publications/econometrica/2011/03/01/model-confidence-set 。

**NW1987**：Newey, W. K., & West, K. D. (1987). A Simple, Positive Semi-definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix. Econometrica, 55(3), 703–708。Bartlett HAC 的命名來源；本實作具體 estimator 依 B1 式 7 至 9 及 B2 §3.1.1。

**PR1994**：Politis, D. N., & Romano, J. P. (1994). The Stationary Bootstrap. Journal of the American Statistical Association, 89(428), 1303–1313。Geometric-block stationary bootstrap 的來源。B2 使用 block bootstrap 家族，本工具提供 stationary、circular、moving 三種明確可選的實作。這些選項不表示原文所有實驗都使用同一方法或本版預設 block length。

**LP**：本次提供的《LoadPulse｜第一次論文閱讀與分享筆記_B.md》（保留於 [docs/READING_NOTES_B.md](docs/READING_NOTES_B.md)）內「LoadPulse 可採用的操作定義草案」，及本對話第 11 頁公式解釋。它是研究設計提案，不是已完成的丹麥資料驗證。多視窗 max-statistic bootstrap、稀疏日曆組處理、事件合併及包裝接口為本工具工程／方法延伸。

## 每個檔案的來源

| 檔案 | 方法或資料来源 | 原文定位與界線 |
|---|---|---|
| `loadpulse_stats/losses.py` | B1、B2 | B1 §3.1 印刷頁 8–9；B2 §2 印刷頁 5，§3.1.2 頁 11。error 與 loss、配對差定義；介面檢查自行設計 |
| `loadpulse_stats/resampling.py` | B1、B2、NW1987、PR1994 | B1 §3.5 式 7–9 印刷頁 14；B2 §3.1、§3.2.3 印刷頁 10–16。Bartlett estimator、joint block bootstrap；batching 自行設計 |
| `loadpulse_stats/forecasting.py` | B1 與工程實作 | B1 §3.1、Algorithm 1 印刷頁 8–9、12–13。三種 estimation scheme；target-aligned OLS helper、可用性 embargo 與 array 容器自行設計 |
| `loadpulse_stats/forecast_breakdown.py` | B1 | §3.3 式 1–4 頁 10；Algorithm 1 式 5–9 頁 12–14；Corollaries 3–4 式 10 頁 14–15；§5 式 13–14 頁 17–18；§6 式 15–16 頁 18–19 |
| `loadpulse_stats/model_confidence_set.py` | B2 | §2.3 Definition 4、Theorem 3 頁 8–9；§3.1.2 和 Proposition 1 頁 11–12。R/eR、max/emax、MCS p-values；+1 Monte Carlo 修正與完全重複欄位合併为明示工程延伸 |
| `loadpulse_stats/regression_mcs.py` | B2 | §3.1.1 頁 10–11 quadratic EPA；§3.2.1–3.2.4 頁 13–17 Gaussian Q、B3 effective df、式 3 null bootstrap、KLIC/AIC/BIC/star criteria。限定線性回歸 |
| `loadpulse_stats/local_failure.py` | LP，受 B1、B2 啟發 | u 與 A 的計算、calendar baseline、条件 bootstrap max scan、共識與事件合併为 LoadPulse 延伸；不是 B1 原單尾檢定，也不是 MCS |
| `loadpulse_stats/_validation.py` | 本工具工程設計 | array shape、finite 值、參數範圍檢查，无論文實驗數據 |
| `loadpulse_stats/__init__.py` | 本工具工程設計 | Public API 匯出與版本資訊，无論文算法 |
| `examples/demo.py` | 本工具合成資料設計 | 全部由固定 seed 隨機產生，用於走完四種分析；不是丹麥實際資料或論文 Tables |
| `examples/__init__.py` | 本工具工程設計 | Python package 標記，无研究資料 |
| `examples/read_csv_example.py` | 本工具工程設計 | 標準庫 CSV 讀取介面，僅使用使用者預測資料，不下載任何資料 |
| `tests/test_methods.py` | 手算與合成檢查 | 公式來自 B1/B2；測試數字自行設計。部分路徑例子受 B2 Table 1 累積 p-value 想法啟發，不是 Table 1 的數字複刻 |
| `tests/__init__.py` | 本工具工程設計 | Python test package 標記 |
| `tests/statistical_check.py` | 本工具 Monte Carlo 校準示例 | 已知 null、known alternative 的合成資料；設定為本次驗證設計，不是原論文 Monte Carlo 配置 |
| `pyproject.toml` | 本工具工程設計 | 安裝、版本與套件 metadata |
| `requirements.txt` | 本工具工程設計 | NumPy、SciPy 執行依賴 |
| `README.md` | B1、B2、LP 與本工具說明 | 操作流程、method 邊界、最小呼叫範例。引用對應以上核對版本 |
| `SOURCES.md` | B1、B2、LP 與本工具實作 | 本來源對照及版本界線，不是另一份原始資料集 |
| `API_REFERENCE.md` | 本工具程式與 docstring | 自動彙整公開函式 signature、參數與回傳容器，再依本來源表理解來源 |
| `VALIDATION.md` | 本次實際測試結果 | 實際執行測試與例子的紀錄，不能代替 Denmark empirical validity |
| `docs/METHODS_AND_SOURCES.md` | 原 Word 說明文件；README、API_REFERENCE、SOURCES | Word 轉 Markdown，保留表格、範例與逐檔來源；更新 repository 路徑 |
| `docs/READING_NOTES_B.md` | 使用者提供的原始閱讀筆記 | 原樣保留，非程式執行結果；其中研究提案與已實作能力須分開理解 |
| `.gitignore` | 本工具工程設計 | 排除 Python 暫存、建置產物、虛擬環境與本機資料 |

## 為什麼不能宣稱所有分支都自動完成推論

B1 的一般 influence representation 需要模型專屬 scores／導數／B 矩陣，本工具提供組裝運算。Proposition 7 的公開版 S_zz 記號：B3 写未加 lag 的二階矩，式 16 後的 estimator 說明含 HAC lag 项。這一分支因此採明確 moment 輸入；正式使用時對照期刊版並確認假設。

B2 TQ 有自己的 closed-testing 要求，本工具提供 standalone EPA 而不配任意淘汰。回歸版本按文中 Gaussian linear-model branch，不延伸為任意深度網路的 likelihood MCS。LP scan 是 conditional on frozen baseline 的近似方法，没有宣称吸收全部估計不確定性。

本工具沒有包含或再散布原文 PDF、作者程式或原始資料集。來源連結與章節為追溯依據。
