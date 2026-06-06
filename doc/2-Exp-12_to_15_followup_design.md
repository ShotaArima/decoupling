# 2-Exp-12 to 2-Exp-15: Follow-up Experiments After Output Decomposition

## 背景

2-Exp-11 では、真の residual 成分を持つ synthetic で `output_decomp_centered` が強く機能した。

一方で FreshRetailNet subset では、`b + r_hat` が `b` を上回らず、特に補正値の bias が大きく崩れた。

このため、次の検証では次の 2 点を切り分ける。

1. FreshRetailNet のどの subset なら residual 補正が意味を持つか。
2. 実データで悪化している原因が bias なのか、構造不足なのか、モデル設計なのか。

## 追加したコード

### subset filter の拡張

FreshRetailNet subset を切り分けるため、`subset_filter` で使える指標を追加した。

- `residual_abs_mean`
- `residual_hour_eta`
- `residual_weekday_eta`
- `discount_std`
- `residual_structure_score`

`residual_hour_eta` は、残差のうち hour ごとの平均差で説明できる割合を表す。

`residual_weekday_eta` は、残差のうち weekday ごとの平均差で説明できる割合を表す。

`residual_structure_score` は、hour / weekday / discount 由来の構造が強い系列を拾うための簡易スコア。

### sweep runner の拡張

`residual-sweep` に `sweep.scenarios` を追加した。

これにより、同じ seed / model 設定のまま、dataset や loss の条件だけを変えて比較できる。

出力は従来通り次のファイルに保存される。

- `all_results.csv`
- `aggregate.csv`
- `summary.json`

scenario を使った場合は、`aggregate.csv` が `scenario` と `name` ごとに集計される。

### bias control の追加

loss に次の項を追加した。

- `residual_bias_weight`
- `series_residual_bias_weight`

さらに validation set で推定した residual bias を test prediction から引く post-hoc calibration も追加した。

calibrated 指標は `calibrated_*` prefix で出力される。

## 2-Exp-12: FreshRetailNet subset 条件比較

目的は、FreshRetailNet のどの subset で residual 補正が意味を持つかを確認すること。

実行:

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-12_freshretailnet_subset_conditions.json
```

比較する scenario:

| scenario | 意味 |
|---|---|
| `residual_std_top` | 残差分散が大きい系列 |
| `hour_structure_top` | hour ごとの残差構造が強い系列 |
| `weekday_structure_top` | weekday ごとの残差構造が強い系列 |
| `discount_variable_top` | discount の変動がある系列 |
| `combined_structure_top` | hour / weekday / discount をまとめた構造スコア上位 |
| `low_structure_negative_control` | 構造スコア下位の負例 |

見る指標:

- `baseline_cell_mae_mean`
- `corrected_cell_mae_mean`
- `high_residual_top10_corrected_mae_mean`
- `residual_r2_mean`
- `component_ablation_without_day_mae_delta_mean`
- `component_ablation_without_hour_mae_delta_mean`
- `component_ablation_without_interaction_mae_delta_mean`

成功条件:

- `combined_structure_top` や `hour_structure_top` で `corrected_cell_mae_mean < baseline_cell_mae_mean` になる。
- `low_structure_negative_control` では改善しない。
- 構造が強い subset ほど ablation delta が大きくなる。

## 2-Exp-13: bias control

目的は、FreshRetailNet で `corrected_cell_bias` が崩れる問題を抑えること。

実行:

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-13_freshretailnet_bias_control.json
```

比較する model:

| model | 意味 |
|---|---|
| `output_decomp_centered` | 2-Exp-11 の中心モデル |
| `output_decomp_bias_loss` | 全体の residual bias penalty を追加 |
| `output_decomp_series_bias_loss` | 系列単位の residual bias penalty も追加 |
| `output_decomp_validation_calibrated` | validation bias を test 補正から引く |
| `output_decomp_bias_loss_calibrated` | bias loss と validation calibration を併用 |

見る指標:

- `corrected_cell_bias_mean`
- `corrected_cell_mae_mean`
- `calibrated_corrected_cell_bias_mean`
- `calibrated_corrected_cell_mae_mean`
- `high_residual_top10_corrected_mae_mean`
- `calibrated_high_residual_top10_corrected_mae_mean`

成功条件:

- calibrated 指標で bias が縮む。
- bias を抑えても MAE が大きく悪化しない。
- 可能なら high residual top 10% で改善する。

## 2-Exp-14: synthetic difficulty sweep

目的は、どの条件なら成分分離が可能かを synthetic で確認すること。

実行:

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-14_synthetic_difficulty_sweep.json
```

比較する scenario:

| scenario | 意味 |
|---|---|
| `base` | 標準設定 |
| `low_interaction` | interaction が弱い |
| `high_interaction` | interaction が強い |
| `high_noise` | noise が大きい |
| `short_history` | 観測日数が短い |
| `small_sample` | 系列数が少ない |
| `high_stockout` | 欠品が多い |

見る指標:

- `component_global_corr_mean`
- `component_day_corr_mean`
- `component_hour_corr_mean`
- `component_interaction_corr_mean`
- `residual_mae_mean`
- `residual_r2_mean`
- `component_ablation_without_*_mae_delta_mean`

成功条件:

- base では 2-Exp-11 と同様に高い成分相関が出る。
- high_noise / short_history / small_sample で相関が落ちる。
- low_interaction では interaction の ablation delta が小さくなる。
- high_interaction では interaction の ablation delta が大きくなる。

2-Exp-14 では clean component は保持しつつ、学習対象は `noisy_true_residual` にする。これにより、noise を増やしたときに「合計 residual の再構成はできても、clean な成分回収がどの程度崩れるか」を確認できる。

## 2-Exp-15: paper table 用の最終比較

目的は、論文に載せる最終比較表を作ること。

synthetic:

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-15_final_synthetic.json
```

FreshRetailNet:

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-15_final_freshretailnet.json
```

最終的に整理する表:

| dataset | scenario | model | baseline MAE | corrected MAE | calibrated corrected MAE | residual R2 | component recovery |
|---|---|---|---:|---:|---:|---:|---|

論文上の主張は次の形にする。

> 真の residual 成分が存在し、平均ゼロ制約が妥当な場合、出力分解モデルは成分を高精度に回収できる。実データでは、残差構造が強い subset と bias 制御が必要であり、全系列で無条件に改善するものではない。

## 実験結果提出時に見る順番

1. 2-Exp-12 の `aggregate.csv` で、どの scenario が改善しているかを見る。
2. 2-Exp-13 の calibrated 指標で、bias が縮んだかを見る。
3. 2-Exp-14 で、synthetic の分離可能条件が仮説通りかを見る。
4. 2-Exp-15 は、2-Exp-12〜14 の結果を踏まえて論文用の表として使う。
