# LoadPulse 論文方法 Python 工具包

版本 1.0.0。Python 3.10 以上。依據 Giacomini 與 Rossi 的 Forecast Breakdown，以及 Hansen、Lunde 與 Nason 的 Model Confidence Set，整理為可 import 的 Python 模組。所有程式為獨立實作，並非作者官方程式。逐檔原文定位見 [SOURCES.md](SOURCES.md)，完整參數與回傳值見 [API_REFERENCE.md](API_REFERENCE.md)。原 Word 說明文件已轉為 [方法與來源說明](docs/METHODS_AND_SOURCES.md)，原始閱讀筆記保留於 [READING_NOTES_B.md](docs/READING_NOTES_B.md)。

這個工具包用來評估你們已經產生的預測和損失，也提供線性 OLS 的 Fixed、Rolling、Recursive 示範。它不下載 DK1／DK2、氣象或論文原始資料，不會替你們選出最佳預測期距、視窗或 block length。示範中的數字都是合成資料。

## 1 安裝與 import

先取得 repository，再於根目錄執行：

```bash
git clone https://github.com/cloud-driver/loadpulse-methods.git
cd loadpulse-methods
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell 改用 .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
python -m examples.demo
python -m unittest discover -s tests -v
```

也可以不安裝套件，把完整 `loadpulse_stats` 資料夾放到你的 Python 專案中。它的內部相對 import 需要保留整個資料夾，不能只抽出單一 `.py` 檔。正式環境可用 `python -m pip install .` 一般安裝。

```python
from loadpulse_stats.forecast_breakdown import surprise_loss, forecast_breakdown_test
from loadpulse_stats.model_confidence_set import model_confidence_set
from loadpulse_stats.local_failure import window_excess_loss
# 也可 from loadpulse_stats import model_confidence_set
```

執行 `help(函式名稱)` 可以直接查看參數說明與適用條件。實測環境為 NumPy 2.3.5、SciPy 1.17.0；最低版本範圍未逐一測試。

## 2 我們實際能借用什麼

| 工作 | 對應方法 | 在 LoadPulse 中的用途 |
|---|---|---|
| 定義預測 error 和 loss | B1 §3.1、B2 §2 | 保留高估／低估方向，明確指定平方或絕對損失 |
| 指定預測方案 | B1 §3.1 | 按資訊可用時間產生樣本外預測 |
| 定義 surprise loss | B1 式 1 至 4 | 比較同一模型的樣本外損失與對應樣本內平均 |
| 處理估計方式與序列相依 | B1 Algorithm 1、Theorem 2、Corollary 3 | 算正確的檢定變異數，而非一般標準差 |
| 簡化變異數估計 | B1 Corollary 4 | 假設成立時使用依估計方案調整的 HAC |
| 線性模型過擬合修正 | B1 §5，式 13、14 | 降低有限樣本 train–test gap 對判斷的干擾 |
| 預測未來模型損失 | B1 §6，式 15、16 | 對 surprise loss 做額外回歸，含低階 Wald 工具 |
| 保留模型選擇不確定性 | B2 §2.3、§3.1.2 | 用 coherent R 或 max 檢定淘汰相對差模型 |
| 比較線性回歸的樣本內準則 | B2 §3.2 | KLIC、AIC、BIC 與有效自由度版本 |
| 掃描局部失效、合併事件 | LoadPulse 延伸 | 對事先設定的多模型、多視窗做條件 bootstrap 掃描 |

MCS 回答相對模型比較，不能取代共同失效檢定。所有模型一起多出相同損失時，模型間的損失差不會改變。資料不足時留在 MCS，也不能說該模型絕對品質良好。

## 3 檔案與函式

| 模組 | 公開函式 |
|---|---|
| `losses.py` | `forecast_error`、`squared_loss`、`absolute_loss`、`loss_matrix`、`loss_differentials` |
| `resampling.py` | `hac_long_run_covariance`、`block_bootstrap_indices`、`bootstrap_mean_samples` |
| `forecasting.py` | `forecasting_splits`、`rolling_origin_ols` |
| `forecast_breakdown.py` | `surprise_loss`、`forecast_weights`、`weighted_breakdown_variance`、`scheme_variance_factor`、`stationary_breakdown_variance`、`linear_overfit_correction`、`iid_overfit_correction`、`forecast_breakdown_test`、`fit_surprise_regression`、`predict_surprise_regression`、`predictive_wald_test`、`predictive_cross_factor`、`predictive_covariance_from_moments`、`surprise_regression_lower_bound` |
| `model_confidence_set.py` | `mcs_statistics`、`model_confidence_set` |
| `regression_mcs.py` | `equal_predictive_ability_test`、`gaussian_deviance`、`effective_degrees_of_freedom`、`regression_model_confidence_set` |
| `local_failure.py` | `fit_calendar_baseline`、`predict_calendar_baseline`、`excess_loss`、`window_excess_loss`、`joint_window_scan`、`consensus_failure`、`merge_failure_events` |

總計 37 個公開函式。`MCSResult` 等 dataclass 是回傳容器，不計入函式數量。`_validation.py` 為內部輸入檢查，不是論文方法。

## 4 資料介面與時間對齊

單一模型序列使用 `(n,)`。多模型資料使用 `(n,k)`，每列是一個共同預測目標、每欄是一個模型。模型欄位順序在預測、loss、baseline、bootstrap 與最終評估中必須一致。不同期距要分開執行。

本版接受 NumPy 陣列或可轉成陣列的 list。你們先在資料層完成時間戳記 join，移除重複時間、处理缺值、夏令時間與資料修訂。函式不會猜測時間對齊，也不會默默刪除某模型的缺值。若資料不連續，block length 代表「資料列數」而不是小時數；應先解決時間軸問題。

error 的正值代表實際值高於預測，也就是低估用電。loss 只代表代價，通常没有正負方向。DK1 和 DK2 的原始平方損失單位相同但規模不同，不能直接拿數值判斷哪一區比較容易失效。需要用事前固定的規模標準化或分區評估。

## 5 第一篇的直接使用方式

### OLS 加平方損失

`features[t]` 必須是為 `actual[t]` 設計、而且在 `t-horizon` 已經可取得的特徵。例如小時、星期、已知假日以及適當落後的負載。不能把目標時點事後觀測到的天氣或負載放進去。

```python
from loadpulse_stats import (
    rolling_origin_ols, surprise_loss, weighted_breakdown_variance,
    forecast_breakdown_test,
)

# actual:(N,), features:(N,p)，自行準備與按時間排列
fit = rolling_origin_ols(actual, features, 1000, scheme="rolling", horizon=1)
sl = surprise_loss(fit.out_of_sample_loss, fit.in_sample_mean_loss)
sigma2 = weighted_breakdown_variance(
    fit.full_losses, m=1000, n=len(sl), scheme="rolling", max_lag=24,
    estimation_irrelevant=True,
)
result = forecast_breakdown_test(
    sl, asymptotic_variance=sigma2,
    variance_source="B1 Algorithm 1 / Corollary 3; OLS squared loss",
)
print(result.statistic, result.pvalue, result.reject)
```

例中的 1000、24 是呼叫示意，不是丹麥資料的建議最佳值。`estimation_irrelevant=True` 代表你有依據使用 Theorem 2(a)／Corollary 3。它不是一個可任意打開以略過計算的開關。

`fit.full_losses` 依 Algorithm 1 排列：最初 m 個樣本內 loss、期距造成的 h−1 個中間項、最後 n 個樣本外 loss。Fixed 的中間項依原文註腳填零。`m` 在這個 helper 中是已可估計的 target-aligned 列數，和建立落後項前的原始時間列數不一定相同。

### 一般估計不確定性

如果模型、估計損失和評價損失不符合簡化條件，`weighted_breakdown_variance` 會要求：完整 score 序列 `(T,q)`、每次 surprise loss 的參數導數 D `(n,k)`、以及估計量 influence 矩陣 B `(n,k,q)`。末 h 列 score 須填零；Fixed 只保留初始 m 列 score。

```python
sigma2 = weighted_breakdown_variance(
    full_losses, m, n, scheme="recursive", horizon=h, max_lag=lag,
    scores=score_series,
    surprise_gradients=D,
    influence_matrices=B,
)
```

這些輸入需要模型本身的統計推導，不能從任意神經網路的訓練紀錄猜出來。本版提供原文的通用運算，不宣稱提供所有機器學習模型的 A3 influence representation。

### Covariance-stationary 捷徑

`stationary_breakdown_variance` 使用 Corollary 4。它對樣本外 loss 算 Bartlett HAC，再乘上估計方案的 factor。它額外要求 H0 下的 loss covariance stationarity，以及估計不確定性可忽略。

```python
sigma2 = stationary_breakdown_variance(
    oos_losses, m=1000, scheme="rolling", max_lag=24,
    assumptions_confirmed=True,
)
```

預設會拒絕未確認假設的呼叫。此函式不是自動的平穩性檢查。

### 過擬合修正

`linear_overfit_correction` 實作式 13 的 trace 版本。它需要平均 X′X 與 full-sample parameter estimate 的漸近 covariance，後者是 `sqrt(T)*beta` 的 covariance，不是 `beta` 的普通 covariance。

`iid_overfit_correction` 是式 14 的特殊情形，需要誤差 variance 和 k 個參數（包含截距）。回傳的 c 是從 `sqrt(n)*mean(SL)` 扣除的量；不要直接從每個 SL 扣 c。

```python
c = iid_overfit_correction(
    n=len(sl), m=1000, scheme="rolling",
    error_variance=sigma_error_squared, n_parameters=k,
)
result = forecast_breakdown_test(
    sl, asymptotic_variance=sigma2, correction=c,
    variance_source="B1 Corollary 3 + equation 14 iid special case",
)
```

不能把線性／平方損失的修正無条件用在 MAE、樹模型或深度網路。

### 預測未來 surprise loss

`fit_surprise_regression` 和 `predict_surprise_regression` 對應式 15。預測變數必須在 forecast origin 已知。此回歸预测的是模型表現，不是原始用電。

`predictive_cross_factor` 和 `predictive_covariance_from_moments` 提供 Proposition 7 式 16 的低階 covariance 組裝。`predictive_wald_test` 再算 Wald，`surprise_regression_lower_bound` 提供逐點的條件平均單尾信賴下界。它不是個別未來觀測的 prediction interval，也不是同時信賴帶。

公開工作論文對 S_zz 在 B3 寫成樣本二階矩，式 16 後的說明卻又加入 HAC lag 項。此版本差異會影響一般自相關 predictor 的 estimator。因此本版要求使用者明確提供 moment，不會自動選擇一套並冒稱完整復刻。要做該分支的正式推論，先對照期刊定稿與你的模型推導；一般的 regression-residual HAC 不能直接冒充原文修正。

## 6 第二篇的直接使用方式

### 預測模型 MCS

```python
from loadpulse_stats import loss_matrix, model_confidence_set

losses = loss_matrix(actual_targets, prediction_matrix, loss="squared")
result = model_confidence_set(
    losses,
    model_names=["calendar_ols", "seasonal_naive", "load_lag_model"],
    alpha=0.05, statistic="R", n_resamples=5000,
    block_length=24, bootstrap_method="stationary", random_state=42,
)
print(result.included)
print(result.excluded)
print(result.pvalues)
```

`alpha=.05` 表示 95% MCS。`statistic="R"` 對應 range statistic 與 `argmax_i max_j t_ij`；`statistic="max"` 對應各模型相對集合平均的最大 t 與相同方向的淘汰規則。不能把它們改成任意刪掉最大 RMSE。

重抽樣使用共同時間索引；所有模型的相關結構一起保留。每個 subset 都重算該集合的相對平均。程序繼續走到最後一個模型以計算完整 MCS p-values，再根據 requested alpha 決定 included。`history` 中後面的淘汰步驟只是完整 p-value 路徑，不表示它們在此次 alpha 下都被剔除。

每個 model p-value 是沿路 EPA p-value 的累積最大值。最後留下的代表模型 p-value 為 1。有限次重抽樣使用 `(1+count)/(B+1)`，所以不回報完全為零的 p-value。這是明確標示的有限 Monte Carlo 工程選擇。

完全一致的 loss 欄位會先合併成一個候選，完成後把結果還原到所有同值模型。這是為處理退化輸入的工程延伸，等於先對候選集合去重；尤其 Tmax 的平均相對基準會跟「保留重複候選的原始集合」不同。非同值但固定差異的欄位，可能沒有正的 loss-differential variance，函式會报錯。不要透過添加極小亂數把不符合 Assumption 2 的資料硬塞進去。

`equal_predictive_ability_test` 是 §3.1.1 的單次 quadratic-form EPA。它不是 MCS；因為該統計量没有直接提供 coherent elimination，不能自行配一個刪最大 loss 的迴圈。

### 回歸模型的 likelihood MCS

```python
from loadpulse_stats import regression_model_confidence_set

result = regression_model_confidence_set(
    y, [X_calendar, X_calendar_and_lags],
    model_names=["calendar", "calendar_lags"],
    criterion="AIC_star", alpha=0.05,
    n_resamples=2000, block_length=24, random_state=42,
)
print(result.included, result.effective_df)
```

此處是同一樣本上的線性 Gaussian quasi-likelihood，預設自行加截距，design 不能再重複帶截距。每次 moving-block bootstrap 都對同一組 y／X 重新估計所有回歸，並算 B3 effective degrees of freedom，含誤差 variance 參數。

可選 `KLIC`、`AIC`、`BIC`、`AIC_star`、`BIC_star`。KLIC 使用原文 Q 加 overfit correction。星號準則用估出的有效自由度，普通準則用參數數量。null bootstrap 按 §3.2.3 式 3 建立，不是對固定 loss matrix 再套一次 forecasting MCS。

這個版本處理的是样本內回歸規格選擇，不是局部用電預測失效。Nested-model KLIC 的 bootstrap 可能較保守，原文 §3.2.4 已說明。某些重抽樣導致 design rank deficient 或完全擬合時，會报錯而不偷偷補 ridge。

## 7 第 11 頁公式與局部失效延伸

第 11 頁的兩個式子，對應 `excess_loss` 和 `window_excess_loss`。它們是 LoadPulse 設計的量，不是 B1 原檢定。要用歷史 rolling-origin 样本外 loss 建立日曆基準，而不是任意模型的訓練 loss。

至少分開四種資料角色：模型訓練、baseline fitting、null calibration、final evaluation。Baseline fitting 與 null calibration 都在 final evaluation 之前。一般多模型可在同一歷史資料切分中訓練，但不能把這些模型視為獨立證據。

```python
from loadpulse_stats import (
    fit_calendar_baseline, predict_calendar_baseline, excess_loss,
    joint_window_scan, consensus_failure, merge_failure_events,
)

baseline = fit_calendar_baseline(baseline_losses, baseline_calendar_groups, min_count=30)
null_mu = predict_calendar_baseline(baseline, null_calendar_groups)
eval_mu = predict_calendar_baseline(baseline, evaluation_calendar_groups)
null_u = excess_loss(null_losses, null_mu)
eval_u = excess_loss(evaluation_losses, eval_mu)

scan = joint_window_scan(
    null_u, eval_u, window_sizes=[24, 72], block_length=24,
    practical_threshold=delta_per_model, n_resamples=5000,
    model_names=names, alpha=0.05, random_state=42,
    residual_stationarity_confirmed=True,
)
selected, fraction = consensus_failure(scan.failure_mask, minimum_fraction=2/3)
events = merge_failure_events(scan.windows, selected, max_gap=0)
```

`predict_calendar_baseline` 預設对未見或樣本不足的日曆組报錯；只有明確選 `fallback="global"` 才用總體平均，分析時需交代這個近似。

`joint_window_scan` 用 null calibration 的中心化 excess-loss 序列，對整段 final evaluation 的長度聯合重抽樣。每次取所有視窗、模型與尺度中的最大正方向統計量，再給各視窗 simultaneous max-statistic adjusted p-values。還必須超過事前固定的實務門檻。

這個 bootstrap 是**給定已凍結 baseline 的條件、近似推論**。它沒有把 baseline fitting error、資料修訂或前期模型挑選的不確定性算進來。若條件殘差的 variance、季節結構或相依性仍在改變，不能只把確認旗標改成 True 就使用。要主張包含全部估計不確定性的 Robust Temporal Failure，需再做整條程序的聯合重抽樣／模擬校準；本版不會把這個較強結論自動寫進結果。

`consensus_failure` 只是一個描述性比例門檻，不假定模型獨立，也不是另一個 p-value。當底層 scan 已對全部模型／視窗調整時，consensus 是額外子集規則。它仍受限於 scan 的条件性質。`merge_failure_events` 合併交疊或相鄰窗口，回傳 `[start,stop)` 的列索引範圍。它不是確認的物理事件起訖。

## 8 正式研究前要固定的選擇

預測期距、模型家族、loss、估計方案、視窗長度與步長、calendar grouping、稀疏組處理、實務門檻、significance level、block length、resample 次數，以及事件合併規則，都应在檢視最終結果前決定。

先做品質檢查與失效校準，再凍結標籤，最後分析氣象。對未來預測使用當時可取得的天氣預報；事後觀測到的未來天氣只能做理想資訊／回顧分析。共同錯誤或顯著關聯都不能直接推導天氣因果。

没有檢出 failure 代表目前設計與資料未提供充分證據，不等於統計上證明不存在 failure。若檢定力不足，還需報告偵測能力，而不能只宣布 H1 為假。

## 9 實作邊界與驗證

這套程式没有逐行複製作者 Matlab／其他第三方套件，也没有宣稱重現通膨與 Taylor rule Tables 的數字。參考版本是 2006 ECB 工作論文與 2010 CREATES 工作論文；正式書目保留 2009／2011 期刊版。

Tests 包含：HAC 手算、三種 weights 與多期距恆等式、一般 influence variance、過擬合修正的 iid 特例、資訊 embargo、改未來資料不影響過去預測、正確淘汰規則、MCS 累積 p-value、共同衝擊抵消、零 variance 邊界、likelihood 聯合 refit、第 11 頁數值例子，以及重疊窗口合併。這些是數學／工程驗證，不能代替 DK1／DK2 的 empirical validation。

所有 API 的完整型態、參數與回傳容器欄位見 `API_REFERENCE.md`。所有檔案逐一來源見 `SOURCES.md`。

## Repository 文件導覽

- [方法與來源說明](docs/METHODS_AND_SOURCES.md)：由原 Word 說明文件轉換，保留 37 個函式的用途、表格、範例與逐檔來源。
- [API_REFERENCE.md](API_REFERENCE.md)：完整函式簽章、參數與回傳值。
- [SOURCES.md](SOURCES.md)：兩篇原文的章節、公式、頁碼與工程延伸界線。
- [VALIDATION.md](VALIDATION.md)：29 項測試與有限樣本模擬結果。
- [READING_NOTES_B.md](docs/READING_NOTES_B.md)：原始論文閱讀筆記，為研究提案背景；其中的未來設計要求不代表目前程式已實作。

本 repository 僅包含工具程式、文件、合成範例與測試。實際研究資料由使用者自行提供；目前尚未發布授權條款。
