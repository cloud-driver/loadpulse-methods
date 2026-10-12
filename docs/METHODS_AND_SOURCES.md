# LoadPulse 方法與來源說明

Python 工具包 1.0.0    2026 年 10 月 12 日

本文件說明兩篇論文可轉用的方法、37 個公開函式、資料介面、呼叫範例與逐檔來源。工具包提供 Forecast Breakdown、Model Confidence Set，以及另外標示的 LoadPulse 局部失效延伸。程式是獨立實作，尚未用 DK1／DK2 實際資料驗證。

## 快速使用

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python -m examples.demo
python -m unittest discover -s tests -v
```

在 repository 根目錄執行。Python 3.10 以上，執行依賴只有 NumPy 與 SciPy。完整 signature 和 docstring 位於 API_REFERENCE.md。可用 help(函式) 看即時說明。

```python
from loadpulse_stats import model_confidence_set
from loadpulse_stats.forecast_breakdown import surprise_loss
```

## 資料必須先對齊

單一序列為 (n,)，多模型陣列為 (n,k)。每一列必須對應相同目標時間及預測期距。所有模型的欄位順序必須一致。函式不猜測時間對齊、不默默刪除缺值，也不下載電力或氣象資料。缺值、重複時間、UTC／夏令時間、資料版本先在資料層處理。

error 正值表示實際值高於預測，也就是低估。loss 是評價代價。Block length 和 event boundaries 都以資料列衡量，只有完整規則的小時資料才可當成小時。

## 最重要的研究界線

B1 評估同一模型是否比自己的歷史表現惡化；B2 比較同時期候選模型的相對品質。MCS 不能證明所有模型共同失效，更不能證明天氣因果。先固定失效定義，再分析天氣。沒有檢出失效表示證據不足，也不能直接證明不存在失效。

## 誤差和損失

檔案 loadpulse_stats/losses.py。來源 B1 §3.1；B2 §2、§3.1.2。

| 函式 | 用途 |
| --- | --- |
| forecast_error | 實際值減預測值。正值代表低估，負值代表高估。 |
| squared_loss | 平方誤差，較強地處罰大偏差；單位為目標單位的平方。 |
| absolute_loss | 絕對誤差；單位與原目標相同。 |
| loss_matrix | 將共同目標和多模型預測轉成時間乘模型的 loss matrix。 |
| loss_differentials | 計算模型 i 與 j 在同一觀測下的損失差。 |

## 時間相依推論

檔案 loadpulse_stats/resampling.py。來源 B1 式 7 至 9；B2 §3；Bartlett HAC／stationary bootstrap。

| 函式 | 用途 |
| --- | --- |
| hac_long_run_covariance | Bartlett HAC long-run covariance；lag 由使用者事前指定。 |
| block_bootstrap_indices | 產生 stationary、circular 或 moving block 的共同抽樣索引。 |
| bootstrap_mean_samples | 以共同區塊索引對所有模型聯合抽樣，回傳每次平均值。 |

## 預測資訊與估計方案

檔案 loadpulse_stats/forecasting.py。來源 B1 §3.1、Algorithm 1；target-aligned helper 是工程實作。

| 函式 | 用途 |
| --- | --- |
| forecasting_splits | 產生 Fixed、Rolling、Recursive 的訓練索引和預測目標，包含期距 embargo。 |
| rolling_origin_ols | 線性 OLS 加平方損失的可用範例，回傳每次樣本外 loss 及同參數樣本內平均。 |

## 第一篇檢定與預測回歸

檔案 loadpulse_stats/forecast_breakdown.py。來源 B1 §3.3、§3.5、§5、§6；式 1 至 16。

| 函式 | 用途 |
| --- | --- |
| surprise_loss | 樣本外損失減去該次相應的樣本內平均，對應 B1 式 1。 |
| forecast_weights | 依預測方案建立 Algorithm 1 的完整 loss 和 score 權重。 |
| weighted_breakdown_variance | 組合 weighted loss 和估計 influence，再用 HAC 算 B1 檢定的 variance。 |
| scheme_variance_factor | Corollary 4 的 Fixed、Rolling、Recursive factor。 |
| stationary_breakdown_variance | 當額外的 covariance-stationarity 條件成立時，使用 Corollary 4 捷徑。 |
| linear_overfit_correction | 式 13 的線性模型 trace 修正，需要 full-sample parameter 漸近 covariance。 |
| iid_overfit_correction | 式 14 的 iid 特例，使用 error variance 和參數數量，截距也計入。 |
| forecast_breakdown_test | 式 4 或式 13 的單尾檢定。使用者必須提供正確 variance 與來源說明。 |
| fit_surprise_regression | 式 15 的 OLS loss regression point estimate，預測變數須在 origin 已知。 |
| predict_surprise_regression | 用已估計的 loss regression 預測額外損失。 |
| predictive_wald_test | 對 loss regression 做 Wald；Omega 必須另依 Proposition 7 提供。 |
| predictive_cross_factor | Proposition 7 依 estimation scheme 調整交叉 covariance 的 Lambda。 |
| predictive_covariance_from_moments | 式 16 的低階 Omega 組裝，明確提供各個 moments，避免來源記號歧義。 |
| surprise_regression_lower_bound | fitted conditional mean 的逐點單尾信賴下界，不是個別 observation 的預測區間。 |

## 第二篇預測模型信賴集合

檔案 loadpulse_stats/model_confidence_set.py。來源 B2 §2.3、§3.1.2；Definition 4、Proposition 1。

| 函式 | 用途 |
| --- | --- |
| mcs_statistics | 計算目前集合的 R 或 max 統計量、null bootstrap 統計量與相容淘汰位置。 |
| model_confidence_set | 完整 forecasting MCS、累積 model p-values、included、excluded 和完整淘汰歷史。 |

## 第二篇回歸模型準則

檔案 loadpulse_stats/regression_mcs.py。來源 B2 §3.1.1、§3.2.1 至 §3.2.4。

| 函式 | 用途 |
| --- | --- |
| equal_predictive_ability_test | Standalone quadratic-form EPA；此結果沒有自動提供 MCS 淘汰規則。 |
| gaussian_deviance | 線性 Gaussian quasi-likelihood 的 Q，省略所有模型共同常數。 |
| effective_degrees_of_freedom | 依每次共同 bootstrap 的原 fit 和 refit 差異估計 B3 effective df。 |
| regression_model_confidence_set | 線性回歸樣本內 KLIC、AIC、BIC、AIC star、BIC star MCS，重抽樣時聯合 refit。 |

## LoadPulse 局部失效延伸

檔案 loadpulse_stats/local_failure.py。來源 LoadPulse 筆記與第 11 頁；B1/B2 提供靈感，非原論文檢定。

| 函式 | 用途 |
| --- | --- |
| fit_calendar_baseline | 從獨立歷史樣本外 loss，依日曆組估計期待 loss。 |
| predict_calendar_baseline | 對新日曆組查詢期待 loss；未見或稀疏組預設報錯。 |
| excess_loss | 第 11 頁的 u，實際 loss 減相似日曆條件的 baseline。 |
| window_excess_loss | 第 11 頁的 A，指定局部視窗內平均額外損失，不自行給 p-value。 |
| joint_window_scan | 給定已固定 baseline，對全部模型和視窗做近似 max-statistic bootstrap 調整。 |
| consensus_failure | 事前指定比例的描述性共識，沒有把模型當成獨立投票。 |
| merge_failure_events | 合併交疊或相鄰命中視窗，回傳列索引的半開區間。 |

## 第一篇最小呼叫

```python
fit = rolling_origin_ols(actual, features, 1000,
    scheme='rolling', horizon=1)
sl = surprise_loss(fit.out_of_sample_loss,
    fit.in_sample_mean_loss)
v = weighted_breakdown_variance(fit.full_losses,
    1000, len(sl), scheme='rolling', max_lag=24,
    estimation_irrelevant=True)
result = forecast_breakdown_test(sl,
    asymptotic_variance=v,
    variance_source='B1 Corollary 3; OLS squared loss')
```

1000、24 是示意，正式分析前固定。Features[t] 必須在 t-horizon 已可得。OLS 加平方損失才有這裡的簡化依據；一般模型需提供 score、surprise gradient 與 B influence matrices。Variance 不是 surprise loss 的普通 variance，也不是一般 sample standard deviation。

線性過擬合 c 是從 sqrt(n) times mean(SL) 扣除的 numerator correction。不要直接從每筆 SL 扣 c，不能任意套在 MAE 或深度網路。完整規則與 code example 見 README 第 5 節。

B1 的公開版 Proposition 7 對 S_zz 的 B3 定義與式 16 後的 estimator 說明不同。本工具提供明確 moments 的低階組裝，而不猜測一般自相關 predictor 的 estimator。正式使用該進階 covariance 分支前應核對期刊版。

## 第二篇最小呼叫

```python
losses = loss_matrix(actual, predictions, loss='squared')
result = model_confidence_set(losses,
    model_names=names, alpha=0.05, statistic='R',
    n_resamples=5000, block_length=24, random_state=42)
print(result.included, result.pvalues)
```

R 配 eR，max 配 emax。每次 bootstrap 使用所有模型共同時間索引；完整路徑產生每個模型的累積最大 p-value。Alpha .05 是 95% MCS。程式加入有限 Monte Carlo 的 +1 修正。完全重複欄位去重後再還原，其他零 variance 的固定差異直接報錯。

Regression MCS 使用共同 y 與每個 design，聯合 moving-block refit，估有效 df，支援 KLIC、AIC、BIC 及 star 版本。這是樣本內回歸規格選擇，不能包裝成用電的樣本外局部失效。

## 局部失效的資料角色與回傳值

分開模型訓練、baseline fitting、null calibration、final evaluation。用既有樣本外預測的歷史 loss 建立日曆 baseline。先對 null calibration 與 final evaluation 算 excess loss，再掃描事前指定的模型、視窗、尺度。

```python
baseline = fit_calendar_baseline(baseline_losses, groups)
null_u = excess_loss(null_losses, null_expected_losses)
eval_u = excess_loss(eval_losses, eval_expected_losses)
scan = joint_window_scan(null_u, eval_u,
    window_sizes=[24, 72], block_length=24,
    practical_threshold=delta_per_model,
    residual_stationarity_confirmed=True,
    n_resamples=5000, random_state=42)
```

scan 包含 windows、mean_excess_loss、statistics、adjusted_pvalues、failure_mask，以及 inference_scope。只有統計和實務門檻都符合才命中。Consensus 是描述性規則，event merge 只是視窗覆蓋範圍，不是已證實的物理起訖。

Inference scope 明確為 conditional_on_frozen_baseline。此版不包含 baseline-fitting uncertainty、revision uncertainty 或模型選擇不確定性。Residual 的 joint null 分布仍須穩定；把旗標設為 True 不會驗證該假設。整套 Robust Temporal Failure 還需要依實際資料做聯合模擬校準。

## 每個檔案的資料與方法來源

### loadpulse_stats/losses.py

來源：B1、B2。定位與範圍：B1 §3.1 印刷頁 8–9；B2 §2 印刷頁 5，§3.1.2 頁 11。error 與 loss、配對差定義；介面檢查自行設計。

### loadpulse_stats/resampling.py

來源：B1、B2、NW1987、PR1994。定位與範圍：B1 §3.5 式 7–9 印刷頁 14；B2 §3.1、§3.2.3 印刷頁 10–16。Bartlett estimator、joint block bootstrap；batching 自行設計。

### loadpulse_stats/forecasting.py

來源：B1 與工程實作。定位與範圍：B1 §3.1、Algorithm 1 印刷頁 8–9、12–13。三種 estimation scheme；target-aligned OLS helper、可用性 embargo 與 array 容器自行設計。

### loadpulse_stats/forecast_breakdown.py

來源：B1。定位與範圍：§3.3 式 1–4 頁 10；Algorithm 1 式 5–9 頁 12–14；Corollaries 3–4 式 10 頁 14–15；§5 式 13–14 頁 17–18；§6 式 15–16 頁 18–19。

### loadpulse_stats/model_confidence_set.py

來源：B2。定位與範圍：§2.3 Definition 4、Theorem 3 頁 8–9；§3.1.2 和 Proposition 1 頁 11–12。R/eR、max/emax、MCS p-values；+1 Monte Carlo 修正與完全重複欄位合併為明示工程延伸。

### loadpulse_stats/regression_mcs.py

來源：B2。定位與範圍：§3.1.1 頁 10–11 quadratic EPA；§3.2.1–3.2.4 頁 13–17 Gaussian Q、B3 effective df、式 3 null bootstrap、KLIC/AIC/BIC/star criteria。限定線性回歸。

### loadpulse_stats/local_failure.py

來源：LP，受 B1、B2 啟發。定位與範圍：u 與 A 的計算、calendar baseline、條件 bootstrap max scan、共識與事件合併為 LoadPulse 延伸；不是 B1 原單尾檢定，也不是 MCS。

### loadpulse_stats/_validation.py

來源：本工具工程設計。定位與範圍：array shape、finite 值、參數範圍檢查，無論文實驗數據。

### loadpulse_stats/__init__.py

來源：本工具工程設計。定位與範圍：Public API 匯出與版本資訊，無論文算法。

### examples/demo.py

來源：本工具合成資料設計。定位與範圍：全部由固定 seed 隨機產生，用於走完四種分析；不是丹麥實際資料或論文 Tables。

### examples/__init__.py

來源：本工具工程設計。定位與範圍：Python package 標記，無研究資料。

### examples/read_csv_example.py

來源：本工具工程設計。定位與範圍：標準庫 CSV 讀取介面，僅使用使用者預測資料，不下載任何資料。

### tests/test_methods.py

來源：手算與合成檢查。定位與範圍：公式來自 B1/B2；測試數字自行設計。部分路徑例子受 B2 Table 1 累積 p-value 想法啟發，不是 Table 1 的數字複刻。

### tests/__init__.py

來源：本工具工程設計。定位與範圍：Python test package 標記。

### tests/statistical_check.py

來源：本工具 Monte Carlo 校準示例。定位與範圍：已知 null、known alternative 的合成資料；設定為本次驗證設計，不是原論文 Monte Carlo 配置。

### pyproject.toml

來源：本工具工程設計。定位與範圍：安裝、版本與套件 metadata。

### requirements.txt

來源：本工具工程設計。定位與範圍：NumPy、SciPy 執行依賴。

### README.md

來源：B1、B2、LP 與本工具說明。定位與範圍：操作流程、method 邊界、最小呼叫範例。引用對應以上核對版本。

### SOURCES.md

來源：B1、B2、LP 與本工具實作。定位與範圍：本來源對照及版本界線，不是另一份原始資料集。

### API_REFERENCE.md

來源：本工具程式與 docstring。定位與範圍：自動彙整公開函式 signature、參數與回傳容器，再依本來源表理解來源。

### VALIDATION.md

來源：本次實際測試結果。定位與範圍：實際執行測試與例子的紀錄，不能代替 Denmark empirical validity。

### docs/METHODS_AND_SOURCES.md

來源：原始《LoadPulse_方法與來源說明.docx》，其內容依 README、API_REFERENCE、SOURCES 整理。此檔保留原說明文件的表格與範例，並更新 GitHub 路徑。

## 原始文獻與核對版本

B1 正式版：Giacomini, R., & Rossi, B. (2009). Detecting and Predicting Forecast Breakdowns. The Review of Economic Studies, 76(2), 669–705。實際公式核對使用 ECB Working Paper 638，June 2006。文中頁碼是工作論文的印刷頁，不是期刊頁碼。

[官方全文](https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp638.pdf)

B2 正式版：Hansen, P. R., Lunde, A., & Nason, J. M. (2011). The Model Confidence Set. Econometrica, 79(2), 453–497。DOI 10.3982/ECTA5771。實際核對使用 CREATES Research Paper 2010-76，文內 March 2010。

[全文](https://repec.econ.au.dk/repec/creates/rp/10/rp10_76.pdf)

另依 B1/B2 estimator 使用 Newey & West (1987) Bartlett HAC 與 Politis & Romano (1994) stationary bootstrap。來源及樣本分割延伸定位完整記在 SOURCES.md。沒有包含原文 PDF 或作者程式。

## 實際驗證與正式使用限制

29 個單元檢查涵蓋手算、scheme weights、估計 influence、資訊 embargo、MCS 淘汰規則、累積 p-values、共同衝擊抵消、零 variance 邊界、回歸聯合 refit，以及局部視窗合併。另以固定 seed 做 100 次已知 null／alternative 的有限樣本診斷，結果與 Wilson 區間詳列 VALIDATION.md。

有限樣本失真與 block-length 敏感度需要記錄。這些測試不能保證任意資料都達名義 5% 誤報率，尤其短且具相依性的序列。正式 LoadPulse 研究需要依 DK1/DK2 的實際季節、缺值和模型相依再校準，不應直接接受函式輸出作為氣象原因證據。
