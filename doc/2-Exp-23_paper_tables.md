# 2-Exp-23: Paper Table Aggregation

## 背景

2-Exp-19 から 2-Exp-22 で、論文に必要な結果はほぼ揃った。

- FreshRetailNet では `series_mean_all` で予測補正が効く。
- 2-Exp-20 で seed-level の統計確認を行った。
- 2-Exp-21 で hour component の可視化を確認した。
- 2-Exp-22 で synthetic の成分回復と失敗条件を確認した。

2-Exp-23 は追加学習ではなく、これらの結果を論文用の表として固定する集約実験である。

## 目的

目的は 3 つ。

1. synthetic の成分回復表を固定する。
2. FreshRetailNet の補正性能と calibration trade-off を固定する。
3. 統計検証の CI 表を固定する。

これにより、論文本文と appendix の表を同じコマンドで再生成できるようにする。

## 入力

| 入力 | 内容 |
| --- | --- |
| `runs/2-Exp-22_synthetic_difficulty_final/summary.json` | synthetic difficulty の最終結果 |
| `runs/2-Exp-19_freshretailnet_bias_constrained_calibration/summary.json` | FreshRetailNet 補正と calibration |
| `runs/2-Exp-20_statistical_validation/summary.json` | seed-level paired bootstrap |

## 出力

| 出力 | 内容 |
| --- | --- |
| `synthetic_component_recovery.csv` / `.md` | synthetic の成分回復表 |
| `freshretail_correction.csv` / `.md` | FreshRetailNet の補正・bias・hour component 表 |
| `statistical_validation.csv` / `.md` | bootstrap CI と paired comparison 表 |
| `summary.json` | 生成された表の manifest |

## 成功条件

良い結果:

- 3 つの表が同じ出力ディレクトリに生成される。
- 表が `scenario` と `name` で比較できる。
- 本文に必要な主要指標が欠けていない。

悪い結果:

- 入力 summary の path が環境ごとにずれて再生成できない。
- 表に本文で必要な指標が不足している。
- synthetic と FreshRetailNet の役割分担が表から読み取れない。

## 実行コマンド

smoke:

```bash
uv run decoupled-ts paper-tables --config configs/2-Exp-23_paper_tables_smoke.json
```

本番:

```bash
uv run decoupled-ts paper-tables --config configs/2-Exp-23_paper_tables.json
```

## 論文での使い方

本文では次の対応にする。

| 論文上の表 | 2-Exp-23 の出力 |
| --- | --- |
| Synthetic main table | `synthetic_component_recovery.md` |
| FreshRetailNet correction table | `freshretail_correction.md` |
| Statistical validation table | `statistical_validation.md` |

2-Exp-23 が完了すれば、追加実験よりも執筆と図表整形に比重を移せる。

## 実行結果

実行:

```bash
uv run decoupled-ts paper-tables --config configs/2-Exp-23_paper_tables.json
```

出力:

| table | rows | 解釈 |
| --- | ---: | --- |
| `synthetic_component_recovery` | 24 | 8 scenario × 3 model の比較表 |
| `freshretail_correction` | 4 | `series_mean_all` と `same_hour_recent_mean_d7_all` の主比較 |
| `statistical_validation` | 56 | baseline 改善と model 間比較の CI |

注: 初回 config では `bias_constrained_0025` を `bias_constrained_002` と書いていたため、`freshretail_correction` は 4 行になった。config は `bias_constrained_0025` に修正済みなので、再実行すると該当 model も表に含まれる。

## 結果の読み取り

### Synthetic

`synthetic_component_recovery` では、2-Exp-22 と同じ結論が表として固定できた。

- `output_decomp_centered` は `base` で global/day/hour をほぼ完全に回復し、interaction も高く回復した。
- `output_decomp_no_center` は residual MAE だけなら極端に悪くないが、global corr が負になり、interaction corr が崩れる。
- `output_decomp_centered_no_interaction` は `low_interaction` では大きく不利ではないが、`base` と `high_interaction` では component MAE と residual MAE が悪化する。
- `small_sample` では interaction corr が大きく落ち、相互作用成分はサンプル数に敏感である。

論文上は、synthetic を「成分分解が成立する条件と失敗する条件」の主実験として使える。

### FreshRetailNet

`freshretail_correction` では、`series_mean_all` と `same_hour_recent_mean_d7_all` の差が明確に出ている。

| scenario | model | baseline MAE | corrected MAE | calibrated corrected MAE | high residual baseline | calibrated high residual corrected | hour profile corr |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `series_mean_all` | `bias_constrained_001` | 0.0697 | 0.0501 | 0.0534 | 0.2788 | 0.1914 | 0.9938 |
| `series_mean_all` | `mae_grid_reference` | 0.0697 | 0.0508 | 0.0509 | 0.2788 | 0.2127 | 0.9897 |
| `same_hour_recent_mean_d7_all` | `bias_constrained_001` | 0.0580 | 0.0565 | 0.0584 | 0.2534 | 0.2480 | -0.8766 |
| `same_hour_recent_mean_d7_all` | `mae_grid_reference` | 0.0580 | 0.0564 | 0.0561 | 0.2534 | 0.2404 | -0.8684 |

読み取り:

- `series_mean_all` では、全体 MAE と high residual top10 の両方で改善が大きい。
- `series_mean_all` の hour profile corr は約 0.99 で、hour component が residual の時間帯構造を拾っている。
- `bias_constrained_001` は calibrated bias をほぼ 0 に近づけつつ、high residual top10 を最も強く改善している。
- `same_hour_recent_mean_d7_all` は全体 MAE の改善が小さく、hour profile corr も負で、hour component を主張するには弱い。

FreshRetailNet では、提案法が常に強いというより、残差に hour 構造が残る target で効果が出る、という主張が妥当である。

### Statistical validation

`statistical_validation` では、`series_mean_all` の改善は 5 seed で一貫している。

主な結果:

- `series_mean_all` の `bias_constrained_001` は `corrected_cell_mae` で baseline より `-0.0196` 改善し、CI は `[-0.0201, -0.0189]`。
- 同じ model の `calibrated_high_residual_top10_corrected_mae` は `-0.0874` 改善し、CI は `[-0.0905, -0.0842]`。
- `same_hour_recent_mean_d7_all` でも改善はあるが、全体 MAE の改善はおよそ `-0.001` から `-0.002` 程度で小さい。
- `same_hour_recent_mean_d7_all` の calibration 後全体 MAE は、一部 model で悪化している。

このため、本文では `series_mean_all` を主成功例、`same_hour_recent_mean_d7_all` を強い baseline 下の限界例として出すのが良い。

## 次の判断

2-Exp-23 の結果により、追加実験よりも表の整形と執筆に移る段階に入れる。

本文の主張は次の形にする。

```text
Synthetic では、成分が存在する条件で 4変数への分解 が成分を回復できる。
FreshRetailNet では、残差に hour 構造が残る target で、予測補正と high residual 改善が確認できる。
一方、強い same-hour baseline 後の残差では構造が薄く、改善は小さい。
```
