# 驗證紀錄

日期：2026-10-12。執行環境 Python 3.12、NumPy 2.3.5、SciPy 1.17.0。

## 程式與公式檢查

`python -m unittest discover -s tests -v`：29 個測試全部通過。

涵蓋 HAC 手算、Fixed／Rolling／Recursive weights、多期距 loss 恆等式、general influence variance、equal-loss variance、iid overfit 特例、Prop 7 moments 組裝、mean-loss 信賴下界、資訊 embargo、未來資料不能改變過去預測、兩種 coherent MCS 淘汰規則、MCS p-value 累積最大、共同衝擊抵消、完全相同模型、固定差異零 variance、Gaussian Q、effective df、聯合 refit、日曆 sparse group、第 11 頁例子、max scan 和事件合併。

`python -m examples.demo` 完成 B1、forecast MCS、regression MCS 與 LoadPulse scan。已知 variance 增加的 B1 示範 p-value 為 0.0002664；forecast MCS 淘汰刻意偏差的 weak_model；regression MCS 保留 signal；LP 回傳 conditional_on_frozen_baseline 與窗口合併範圍 [21,65)。此範圍不是人工加入改變的精確邊界，因為窗口覆蓋會延伸到兩側。

`pip install --no-deps --no-build-isolation --target <隔離目錄> .` 成功建立並安裝 wheel。Import 與示範不需要 pytest、pandas、arch 或 scikit-learn。相容範圍最低版本尚未逐一跑過。

## 100 次有限樣本診斷

重現：`python -m tests.statistical_check`。固定 seed 20261012，每項 100 次，alpha=.05，MCS bootstrap B=499。以下是本次實際結果，保留不利的結果，不包裝成通用 5% 保證。

| 合成設定 | 命中或誤報 | 比率 | Wilson 95% 區間 |
|---|---:|---:|---:|
| MCS R：3 個相同期望 iid 平方 Gaussian loss，n=240，block=5；任何排除算誤報 | 7/100 | 7% | 3.43%–13.75% |
| MCS max：同一 null 設定 | 5/100 | 5% | 2.15%–11.18% |
| MCS R：第 3 模型加 1.5 損失，檢查有沒有淘汰它 | 100/100 | 100% | 96.30%–100% |
| MCS R：AR(1) rho=.5、burn-in=100 的平方 loss，n=240，block=8；任何排除算誤報 | 12/100 | 12% | 7.00%–19.81% |
| B1 Fixed OLS：T=480、m=n=240、iid error、平方損失、HAC lag=5、iid correction；H0 rejection | 4/100 | 4% | 1.57%–9.84% |

在 AR1 短樣本配置中，observed rejection 高於名義 5%，Wilson 區間也未包含 5%。這是需要認真處理的有限樣本結果；不能只說「block bootstrap 就解決自相關」。

追加敏感度檢查：seed=84，順序跑 (n,block)=(240,8)、(240,24)、(1000,12)，每配置 150 次，B=999。實測任何排除率依序為 13/150=8.67%、17/150=11.33%、10/150=6.67%；Wilson 區間依序為 5.13%–14.26%、7.20%–17.40%、3.66%–11.84%。這些比較使用不同模擬抽樣，不能把差異全歸因於 block length。

結論是需要資料專屬的樣本量與 block 敏感度／誤報校準，而不是任意選一個 block 即聲稱控制 5%。MCS 原文的保證建立在漸近條件；本版照公式運算不會自動給所有有限樣本相同保證。

## 明確未驗證的部分

沒有重現論文通膨、Taylor rule 的原始 Tables；没有 DK1／DK2 實證資料校準；未自動推導任意 ML 模型的 influence representation。LP scan 沒有吸收 frozen baseline 的估計不確定性。Proposition 7 的公開版 moment 記號疑義保留為明確輸入。

完整操作與來源：README.md、API_REFERENCE.md、SOURCES.md。正式研究應先在可代表資料季節、相依性、缺值與共同模型錯誤的合成／歷史穩定期校準，再開啟最終評估。
