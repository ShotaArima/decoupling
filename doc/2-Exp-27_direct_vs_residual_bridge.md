# 2-Exp-27: Direct Target vs Residual Target Bridge

## 背景

2-Exp-26 では、元論文に近い `global + local` 分解を通常の future total sales forecasting に移植し、local を `day/hour/interaction` に分けた。

FreshRetailNet では、`global + day + hour` は `global + local` より小幅に改善した。

```text
paper_global_local WAPE = 0.6233
four_factor_global_day_hour WAPE = 0.6133
four_factor_global_day_hour_interaction WAPE = 0.6267
```

この結果は、day/hour 分割が小売データに合う導入であることを示す。一方で、interaction まで含む 4 成分版は通常予測では安定した改善を示さなかった。

したがって、次に確認すべきことは、同じ表現構造を residual target に移したときに、direct target よりも成分分解の効果が明確になるかである。

## 目的

目的は 3 つ。

1. Exp-26 の direct forecasting 結果と対応する residual target 側の最小比較を作る。
2. `global/local` reference と day/hour split の差が residual target で大きくなるか確認する。
3. 結果に応じて proposal の論調を調整する。

## 比較設計

direct target 側は 2-Exp-26 の結果を使う。

residual target 側は 2-Exp-27 で新しく実行する。

```text
r = y - b
b = series_mean
```

比較する model family は 2-Exp-26 と揃える。

| target | model | 実装上の variant |
|---|---|---|
| direct `y` | `global + local` | `paper_global_local` |
| direct `y` | `global + day + hour` | `four_factor_global_day_hour` |
| direct `y` | `global + day + hour + interaction` | `four_factor_global_day_hour_interaction` |
| residual `y - b` | `global + local` | `paper_global_local_residual` |
| residual `y - b` | `global + day + hour` | `four_factor_global_day_hour_residual` |
| residual `y - b` | `global + day + hour + interaction` | `four_factor_global_day_hour_interaction_residual` |

## なぜ `series_mean` residual に絞るか

この実験は、residual target が常に良いことを示すためではない。

まずは、これまでの実験で residual 構造が残ることが分かっている `series_mean` に絞り、direct target と residual target の対照を作る。

`same_hour_recent_mean` はすでに hour 構造を強く吸収するため、改善幅が小さく、hour component も解釈しにくいことが分かっている。これは後続で「baseline と residual structure に依存する」ことを示す条件として扱う。

## 成功条件

強い結果:

- residual target で `four_factor_global_day_hour` または `four_factor_global_day_hour_interaction` が `paper_global_local_residual` を明確に上回る。
- direct target での改善幅より residual target での改善幅が大きい。
- high residual top10 や corrected cell MAE で改善が出る。

この場合、proposal は次のように強められる。

```text
通常の売上全体を直接分解する場合、day/hour split の利点は小さい。
一方、series_mean baseline で水準成分を除いた residual では、day/hour/interaction の帰納バイアスがより明確に効く。
したがって、本研究の主対象は raw demand forecasting ではなく residual structure learning である。
```

弱い結果:

- residual target でも `paper_global_local_residual` が最も良い。
- day/hour/interaction の改善が direct target と同程度か、それ以下である。

この場合は、proposal を次のように抑える。

```text
残差なら常に成分分解が有利というわけではない。
有効性は baseline が何を取り除き、残差にどの構造が残るかに依存する。
本研究は、残差構造が残る条件を診断し、その条件下で解釈可能な補正を行う方法である。
```

## 実行コマンド

smoke:

```bash
uv run decoupled-ts residual-experiment --config configs/2-Exp-27_direct_vs_residual_bridge_smoke.json
```

FreshRetailNet:

```bash
uv run decoupled-ts residual-experiment --config configs/2-Exp-27_direct_vs_residual_bridge_freshretailnet.json
```

## 出力

```text
runs/2-Exp-27_direct_vs_residual_bridge_smoke/summary.csv
runs/2-Exp-27_direct_vs_residual_bridge_freshretailnet/summary.csv
```

各 variant 配下には次が出る。

```text
history.jsonl
metrics.json
hour_profile.csv
```

## 結果の読み方

まず、Exp-26 の direct target 結果を固定する。

| target | model | WAPE |
|---|---|---:|
| direct | `global + local` | 0.6233 |
| direct | `global + day + hour` | 0.6133 |
| direct | `global + day + hour + interaction` | 0.6267 |

次に、Exp-27 の residual target 結果で、`paper_global_local_residual` に対する改善幅を見る。

特に見る指標:

- `corrected_cell_mae`
- `high_residual_top10_corrected_mae`
- `residual_mae`
- `residual_r2`
- `component_ablation_*` が出る場合はその delta

direct target の改善幅が小さい一方で residual target の改善幅が大きければ、「残差成分から表現学習する方がよい」という論調を強められる。

逆に residual target でも改善が小さい場合は、「残差 target は必要条件ではなく、baseline によって残る residual structure が重要」と整理する。

## smoke 確認

smoke は実行済み。

```bash
uv run decoupled-ts residual-experiment --config configs/2-Exp-27_direct_vs_residual_bridge_smoke.json
```

smoke 結果:

| model | residual MAE | corrected cell MAE | baseline cell MAE | high residual top10 corrected MAE | residual hour corr |
|---|---:|---:|---:|---:|---:|
| `paper_global_local_residual` | 1.5537 | 1.5537 | 1.5631 | 4.6472 | -0.0094 |
| `four_factor_global_day_hour_residual` | 1.5465 | 1.5465 | 1.5631 | 4.7719 | -0.4372 |
| `four_factor_global_day_hour_interaction_residual` | 1.5890 | 1.5890 | 1.5631 | 4.5673 | 0.8638 |

これは smoke 用の小規模 synthetic なので、性能差の解釈には使わない。確認したい点は、3 つの residual variant が同じ config で学習・評価でき、`corrected_cell_mae`、high residual 指標、hour profile 指標が出ることである。

本番の FreshRetailNet 実行後は、次の観点でこの節に結果を追記する。

- `paper_global_local_residual` に対する `four_factor_*` の corrected MAE / WAPE 改善幅。
- Exp-26 direct target の改善幅と、Exp-27 residual target の改善幅の比較。
- high residual top10 で day/hour/interaction が効いているか。
- 結果が弱い場合は、残差 target 一般ではなく `series_mean` など residual structure が残る baseline に依存する、と整理する。
