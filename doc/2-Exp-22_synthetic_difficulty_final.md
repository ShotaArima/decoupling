# 2-Exp-22: Synthetic Difficulty Final

## 背景

FreshRetailNet 側では、2-Exp-17 から 2-Exp-21 により次が分かった。

- `series_mean` residual では hour 成分が残る。
- 提案モデルは hour component を安定して拾う。
- `same_hour_recent_mean` residual では hour 構造が薄く、改善余地が小さい。
- bias 制約つき calibration は high residual top10 と bias の trade-off を改善する。

一方、論文としては「なぜ成分分解が可能なのか」「どの条件で失敗するのか」を synthetic で明確に示す必要がある。

2-Exp-22 は、真の `global/day/hour/interaction` 成分を持つ synthetic で、成分回復と失敗条件を最終表として固定する。

## 目的

目的は 3 つ。

1. `output_decomp_centered` が、真の成分を持つ synthetic で成分回復できることを示す。
2. noise / interaction / history / sample / stockout の変化で、どこから失敗するかを見る。
3. interaction component が必要な条件で、`no_interaction` model が不利になることを確認する。

## 比較 scenario

| scenario | 目的 |
| --- | --- |
| `base` | 標準条件 |
| `low_interaction` | interaction が弱いとき、interaction component の寄与が小さくなるか |
| `high_interaction` | interaction が強いとき、interaction component が必要になるか |
| `high_noise` | noisy target で clean component recovery が崩れるか |
| `short_history` | 観測日数が短いと分解が不安定になるか |
| `small_sample` | 系列数が少ないと成分回復が落ちるか |
| `high_stockout` | 欠品が多いと residual learning が崩れるか |
| `low_hour_signal` | hour signal が弱いと hour ablation delta が小さくなるか |

## 比較 model

| model | 目的 |
| --- | --- |
| `latent_concat_interaction` | 成分を明示的に分けない latent concat baseline |
| `output_decomp_no_center` | 出力分解はするが centering なし |
| `output_decomp_centered` | 提案モデル |
| `output_decomp_centered_no_interaction` | interaction を消した負例 |

## 成功条件

主成功:

- `base` で `output_decomp_centered` の component corr が高い。
- `output_decomp_centered` が `output_decomp_no_center` より成分解釈性で良い。
- `high_interaction` で `output_decomp_centered_no_interaction` が不利になる。
- `low_interaction` で interaction ablation delta が小さくなる。
- `low_hour_signal` で hour ablation delta が小さくなる。

補助成功:

- `high_noise`, `short_history`, `small_sample`, `high_stockout` で component recovery が落ちる。
- 落ち方が `residual_mae` だけでなく component corr / ablation delta にも出る。

悪い結果:

- `output_decomp_centered` と `output_decomp_no_center` の差がない。
- interaction を消しても `high_interaction` で性能が落ちない。
- difficulty を上げても component metrics がほとんど変わらない。

## 見る指標

- `component_global_corr_mean`
- `component_day_corr_mean`
- `component_hour_corr_mean`
- `component_interaction_corr_mean`
- `component_total_true_residual_mae_mean`
- `residual_mae_mean`
- `residual_r2_mean`
- `component_ablation_without_global_mae_delta_mean`
- `component_ablation_without_day_mae_delta_mean`
- `component_ablation_without_hour_mae_delta_mean`
- `component_ablation_without_interaction_mae_delta_mean`

## 実行コマンド

smoke:

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-22_synthetic_difficulty_smoke.json
```

本実験:

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-22_synthetic_difficulty_final.json
```

## 出力

```text
runs/2-Exp-22_synthetic_difficulty_final/all_results.csv
runs/2-Exp-22_synthetic_difficulty_final/aggregate.csv
runs/2-Exp-22_synthetic_difficulty_final/summary.json
```

## 論文上の位置づけ

FreshRetailNet は外部妥当性と限界検証、synthetic は同定可能性と成立条件の主実験として扱う。

2-Exp-22 の表が良ければ、論文では次の流れにできる。

1. Synthetic で、真の residual 成分がある場合は成分回復できる。
2. Centering と interaction component が分離性に効く。
3. Noise / stockout / short history では component recovery が落ちる。
4. FreshRetailNet では、残差構造が残る target でだけ効果が出る。
