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
| `output_decomp_no_center` | 4変数への分解はするが centering なし |
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

## 結果

実行:

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-22_synthetic_difficulty_final.json
```

seed:

```text
17, 23, 31, 47, 59
```

### 提案モデルの難易度別結果

`output_decomp_centered` の集計。

| scenario | residual MAE | residual R2 | total component MAE | global corr | day corr | hour corr | interaction corr | no global delta | no day delta | no hour delta | no interaction delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 0.0965 | 0.9764 | 0.0311 | 0.9995 | 0.9979 | 0.9991 | 0.9654 | 0.3132 | 0.1930 | 0.5033 | 0.0228 |
| low_interaction | 0.0959 | 0.9765 | 0.0300 | 0.9995 | 0.9977 | 0.9991 | 0.8211 | 0.3142 | 0.1941 | 0.5042 | 0.0029 |
| high_interaction | 0.0969 | 0.9767 | 0.0315 | 0.9994 | 0.9982 | 0.9990 | 0.9890 | 0.3129 | 0.1927 | 0.5029 | 0.0625 |
| high_noise | 0.2700 | 0.8428 | 0.0738 | 0.9990 | 0.9795 | 0.9948 | 0.8334 | 0.2776 | 0.1560 | 0.4609 | 0.0094 |
| short_history | 0.0959 | 0.9764 | 0.0345 | 0.9993 | 0.9978 | 0.9985 | 0.9547 | 0.3045 | 0.1879 | 0.5000 | 0.0207 |
| small_sample | 0.1461 | 0.9446 | 0.1129 | 0.9912 | 0.9801 | 0.9902 | 0.4252 | 0.2414 | 0.1321 | 0.4221 | 0.0012 |
| high_stockout | 0.0983 | 0.9755 | 0.0359 | 0.9992 | 0.9975 | 0.9985 | 0.9637 | 0.3091 | 0.1879 | 0.4978 | 0.0207 |
| low_hour_signal | 0.0951 | 0.9519 | 0.0260 | 0.9994 | 0.9983 | 0.9895 | 0.9800 | 0.3179 | 0.1973 | 0.1084 | 0.0256 |

読み取り:

- `base` では global/day/hour はほぼ完全に回復し、interaction も 0.9654 まで回復した。
- `high_interaction` では interaction corr が 0.9890、interaction を消したときの MAE delta が 0.0625 まで上がった。interaction component が必要な条件で必要性が表に出ている。
- `low_interaction` では interaction delta が 0.0029 まで落ちた。interaction が弱い条件では、interaction component の寄与が小さくなる。
- `low_hour_signal` では hour delta が 0.1084 まで落ちた。hour signal を弱くすると hour component の寄与も小さくなる。
- `high_noise` と `small_sample` では成分回復が落ちる。特に `small_sample` は interaction corr が 0.4252 まで落ち、相互作用成分はサンプル数に敏感である。

### Centering と interaction component の比較

| scenario | model | residual MAE | total component MAE | global corr | day corr | hour corr | interaction corr | no interaction delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | output_decomp_no_center | 0.1069 | 0.0495 | -0.8976 | 0.9597 | 0.8244 | 0.0262 | 0.1830 |
| base | output_decomp_centered | 0.0965 | 0.0311 | 0.9995 | 0.9979 | 0.9991 | 0.9654 | 0.0228 |
| base | output_decomp_centered_no_interaction | 0.1070 | 0.0509 | 0.9995 | 0.9972 | 0.9990 | 0.0000 | 0.0000 |
| low_interaction | output_decomp_no_center | 0.0986 | 0.0351 | -0.9178 | 0.9602 | 0.8241 | 0.0055 | 0.1857 |
| low_interaction | output_decomp_centered | 0.0959 | 0.0300 | 0.9995 | 0.9977 | 0.9991 | 0.8211 | 0.0029 |
| low_interaction | output_decomp_centered_no_interaction | 0.0967 | 0.0317 | 0.9995 | 0.9977 | 0.9991 | 0.0000 | 0.0000 |
| high_interaction | output_decomp_no_center | 0.1237 | 0.0716 | -0.5793 | 0.9589 | 0.8249 | 0.0861 | 0.1836 |
| high_interaction | output_decomp_centered | 0.0969 | 0.0315 | 0.9994 | 0.9982 | 0.9990 | 0.9890 | 0.0625 |
| high_interaction | output_decomp_centered_no_interaction | 0.1305 | 0.0840 | 0.9995 | 0.9910 | 0.9978 | 0.0000 | 0.0000 |
| low_hour_signal | output_decomp_no_center | 0.1054 | 0.0467 | 0.9554 | 0.8800 | 0.4513 | 0.1428 | 0.0858 |
| low_hour_signal | output_decomp_centered | 0.0951 | 0.0260 | 0.9994 | 0.9983 | 0.9895 | 0.9800 | 0.0256 |
| low_hour_signal | output_decomp_centered_no_interaction | 0.1063 | 0.0487 | 0.9995 | 0.9978 | 0.9883 | 0.0000 | 0.0000 |

読み取り:

- `output_decomp_no_center` は residual MAE だけを見ると極端に悪くないが、global corr が負になり、interaction corr もほぼ崩れる。これは「当てること」と「成分として読めること」が別であることを示している。
- `output_decomp_centered` は residual MAE と component MAE の両方で良く、成分 corr も高い。
- `output_decomp_centered_no_interaction` は `low_interaction` では大きく悪化しないが、`base` と `high_interaction` では residual MAE と component MAE が悪化する。interaction がある条件では interaction component が必要である。

## 結論

2-Exp-22 は主成功条件を満たした。

- true component がある synthetic では、`output_decomp_centered` が成分を回復できる。
- centering は global/day/hour/interaction の解釈性に強く効く。
- interaction component は、interaction が十分にある条件で必要になる。
- low interaction / low hour signal では、対応する ablation delta が小さくなる。
- high noise / small sample では成分回復が落ち、成立条件と失敗条件を説明できる。

この結果により、論文では synthetic を「同定可能性と成立条件の主実験」として置ける。

## 論文上の位置づけ

FreshRetailNet は外部妥当性と限界検証、synthetic は同定可能性と成立条件の主実験として扱う。

2-Exp-22 の表が良ければ、論文では次の流れにできる。

1. Synthetic で、真の residual 成分がある場合は成分回復できる。
2. Centering と interaction component が分離性に効く。
3. Noise / stockout / short history では component recovery が落ちる。
4. FreshRetailNet では、残差構造が残る target でだけ効果が出る。
