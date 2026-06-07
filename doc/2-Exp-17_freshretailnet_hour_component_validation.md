# 2-Exp-17: FreshRetailNet Hour Component Validation

## 目的

2-Exp-16 では、FreshRetailNet でも target と粒度を選ぶと残差補正が改善することが分かった。

特に良かった条件は次の 2 つである。

- 店舗商品単位の `series_mean_all`
- 店舗第三カテゴリ単位の `store_third_category_series_mean_repro_top`

2-Exp-17 では、この改善が偶然ではないかを 5 seed で確認し、さらに hour 成分が本当に補正に効いているかを検証する。

## 仮説

FreshRetailNet の残差補正では、曜日よりも時間帯の構造が効いている。

数式で書くと、残差を

```text
r = g + d + h + i + eps
```

と分けたとき、FreshRetailNet では `h` の寄与が大きい。

そのため、良い結果では次が起きる。

- `b + r_hat` が `b` のみより MAE を下げる
- 高残差 top10 でも MAE を下げる
- `hour_component` を消すと MAE が悪化する
- residual の hour profile と予測 hour profile が相関する
- `hour_component` の heatmap が時間帯ごとに解釈可能な形を持つ

## 比較条件

| scenario | 役割 | 期待 |
| --- | --- | --- |
| `series_mean_all` | 2-Exp-16 の主成功条件 | 全体 MAE、高残差 top10、hour ablation が改善する |
| `store_third_category_series_mean_repro_top` | 集約粒度の主成功条件 | 店舗商品単位より残差構造が見えやすい |
| `same_hour_recent_mean_d7_all` | 強い baseline 対照 | 改善は小さいが、強い baseline 下の限界を示す |
| `weekday_same_hour_mean_all` | 負例 | baseline が構造を消し、補正余地が小さい |

## 実験モデル

| model | 目的 |
| --- | --- |
| `output_decomp_centered` | 出力成分を直接分ける基本モデル |
| `output_decomp_bias_loss_calibrated` | bias 制御と validation calibration を入れた安定化モデル |

2-Exp-16 では、`output_decomp_centered` の方が素直に改善する条件と、`output_decomp_bias_loss_calibrated` の方が高残差で安定する条件が混在していた。そのため 2-Exp-17 では両方を残す。

## 追加指標

2-Exp-17 から次の指標を追加した。

| metric | 意味 |
| --- | --- |
| `residual_hour_profile_corr` | residual の時間帯平均と `r_hat` の時間帯平均の相関 |
| `residual_hour_profile_mae` | residual の時間帯平均と `r_hat` の時間帯平均の MAE |
| `hour_component_residual_profile_corr` | residual の時間帯平均と `hour_component` の時間帯平均の相関 |
| `hour_component_profile_mean_abs` | hour 成分の平均的な大きさ |
| `hour_component_cell_abs_mean` | cell 単位で見た hour 成分の大きさ |

これにより、単に MAE が改善したかだけでなく、hour 成分が本当に hour profile を拾っているかを見る。

## 成功条件

主成功:

- `series_mean_all` で corrected MAE が baseline MAE より安定して低い
- `series_mean_all` で high residual top10 が改善する
- `component_ablation_without_hour_mae_delta_mean` が正で、day / interaction より大きい
- `residual_hour_profile_corr_mean` または `hour_component_residual_profile_corr_mean` が正になる

補助成功:

- `store_third_category_series_mean_repro_top` でも同じ傾向が出る
- `same_hour_recent_mean_d7_all` では改善が小さく、強い baseline の限界を示せる
- `weekday_same_hour_mean_all` では補正が効かず、構造を消す baseline の負例になる

失敗条件:

- seed を増やすと `series_mean_all` の改善が消える
- MAE は改善するが hour ablation がほぼ 0 になる
- hour profile 指標が低く、global 補正だけで説明できる
- `store_third_category_series_mean_repro_top` の改善が不安定になる

## 実行コマンド

まず smoke を実行する。

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-17_smoke.json
```

本実験を実行する。

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-17_freshretailnet_hour_component_validation.json
```

## 出力の確認

集計結果:

```text
runs/2-Exp-17_freshretailnet_hour_component_validation/aggregate.csv
```

全 run:

```text
runs/2-Exp-17_freshretailnet_hour_component_validation/all_results.csv
```

hour heatmap:

```text
runs/2-Exp-17_freshretailnet_hour_component_validation/<scenario>/seed_<seed>/<model>/z_hour_heatmap.csv
```

component array:

```text
runs/2-Exp-17_freshretailnet_hour_component_validation/<scenario>/seed_<seed>/<model>/hour_component.npy
```

## 結果の読み方

`series_mean_all` で全体 MAE と high residual top10 が改善し、かつ `without_hour` の delta が大きければ、FreshRetailNet では hour 構造が残差補正の主要因だと言える。

一方、`same_hour_recent_mean_d7_all` や `weekday_same_hour_mean_all` で改善が小さい場合は、強い baseline が day/hour 構造を先に吸収してしまい、分離表現の学習余地が小さくなるという 2-Exp-16 の解釈を補強する。

論文では、FreshRetailNet を主張の中心ではなく、外部妥当性と適用条件の実験として置く。2-Exp-17 はその中で「どの target なら実データでも効果が出るか」を示す本実験になる。

## 結果メモ

5 seed の平均では、`series_mean_all` と `store_third_category_series_mean_repro_top` の両方で安定した改善が出た。

| scenario | model | baseline MAE | corrected MAE | high residual baseline MAE | high residual corrected MAE | residual R2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `series_mean_all` | `output_decomp_centered` | 0.0697 | 0.0506 | 0.2788 | 0.2061 | 0.3767 |
| `series_mean_all` | `output_decomp_bias_loss_calibrated` | 0.0697 | 0.0507 | 0.2788 | 0.2000 | 0.3894 |
| `store_third_category_series_mean_repro_top` | `output_decomp_centered` | 0.0811 | 0.0643 | 0.3138 | 0.2509 | 0.2600 |
| `store_third_category_series_mean_repro_top` | `output_decomp_bias_loss_calibrated` | 0.0811 | 0.0650 | 0.3138 | 0.2545 | 0.2380 |
| `same_hour_recent_mean_d7_all` | `output_decomp_centered` | 0.0580 | 0.0561 | 0.2534 | 0.2403 | 0.0862 |
| `weekday_same_hour_mean_all` | `output_decomp_centered` | 0.0420 | 0.0424 | 0.1955 | 0.1955 | 0.0021 |

解釈:

- `series_mean_all` は 5 seed でも最も安定して改善した。
- 店舗第三カテゴリ集約でも改善し、店舗商品より粗い粒度で残差構造が見えるという 2-Exp-16 の結果を補強した。
- `same_hour_recent_mean_d7_all` は改善するが小さい。強い baseline が残差構造を先に吸収している。
- `weekday_same_hour_mean_all` はほぼ改善せず、負例として使える。

hour 成分の寄与も明確だった。

| scenario | model | without global delta | without day delta | without hour delta | without interaction delta | hour profile corr |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `series_mean_all` | `output_decomp_centered` | 0.0049 | 0.0013 | 0.0099 | 0.0007 | 0.9934 |
| `series_mean_all` | `output_decomp_bias_loss_calibrated` | 0.0042 | 0.0021 | 0.0115 | 0.0013 | 0.9901 |
| `store_third_category_series_mean_repro_top` | `output_decomp_centered` | 0.0033 | 0.0002 | 0.0096 | 0.0000 | 0.9857 |
| `store_third_category_series_mean_repro_top` | `output_decomp_bias_loss_calibrated` | 0.0032 | 0.0001 | 0.0094 | 0.0000 | 0.9857 |
| `same_hour_recent_mean_d7_all` | `output_decomp_centered` | 0.0010 | 0.0022 | 0.0006 | 0.0017 | -0.8772 |
| `weekday_same_hour_mean_all` | `output_decomp_centered` | -0.0003 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

解釈:

- `series_mean_all` と店舗第三カテゴリでは、hour 成分を消すと MAE が最も悪化する。
- hour component と residual hour profile の相関も約 0.99 で、単なる偶然の改善ではない。
- `same_hour_recent_mean_d7_all` では hour profile 相関が負になっており、基準成分が hour 構造を吸収した後の残差には、同じ意味の hour 成分が残っていない。

注意点:

- raw の `corrected_cell_bias` は大きく負に寄る。
- `series_mean_all / output_decomp_centered` で約 -0.257、`output_decomp_bias_loss_calibrated` でも約 -0.182。
- 既存の validation bias calibration は bias を縮めるが、全体 MAE は raw より悪化することがある。

現時点の結論:

FreshRetailNet でも、`series_mean` target なら hour 残差構造を学習できる。これは 2-Exp-17 の主成功である。

ただし、予測補正として使うには raw の `r_hat` をそのまま足すだけでは bias が大きい。次は `2-Exp-18` として、validation 上で `r_hat` の大きさと bias を調整する calibration / shrinkage を検証する。
