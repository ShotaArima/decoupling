# 2-Exp-28: Latent Split vs Output Decomposition

## 目的

2-Exp-28 は、元論文型の latent split と、本研究の主提案である output decomposition を同じ `series_mean` residual 条件で直接比較する実験である。

2-Exp-26 では、direct target に対して `global/local` から `global/day/hour/(interaction)` へ拡張した。
2-Exp-27 では、同じ比較を residual target に移した。
その結果、residual target は baseline 改善に有効だったが、4 成分 latent split が常に `global/local` residual を上回るわけではなかった。

したがって次に確認すべき問いは、次である。

```text
latent を細かく分けるだけでは不十分だとして、
output decomposition + centering constraints は
同じ residual target 上で何を改善するのか。
```

この実験は、性能をさらに上げるための探索ではなく、proposal の論理を閉じるための比較実験である。

## 比較するモデル

| name | type | 位置づけ |
|---|---|---|
| `paper_global_local_residual` | `multigrain_ae` | 元論文に近い `global/local` residual 版 |
| `four_factor_global_day_hour_residual` | `multigrain_ae` | local を day/hour latent に分けた版 |
| `four_factor_global_day_hour_interaction_residual` | `multigrain_ae` | day/hour/interaction latent 版 |
| `output_decomp_no_center` | `output_decomposition` | 出力分解のみ。centering なしの負例 |
| `output_decomp_centered_no_interaction` | `output_decomposition` | centering あり、interaction なし |
| `output_decomp_centered` | `output_decomposition` | 主提案の最小形 |

主比較では calibration を入れない。
理由は、この実験の目的が補正性能を最大化することではなく、同じ residual target 上で latent split と output decomposition の違いを見ることだからである。

必要であれば、後続実験または appendix で `mae_grid_reference` や `bias_constrained_001` を追加する。

## 実行コマンド

Smoke:

```bash
uv run decoupled-ts residual-experiment --config configs/2-Exp-28_latent_vs_output_decomposition_smoke.json
```

FreshRetailNet:

```bash
uv run decoupled-ts residual-experiment --config configs/2-Exp-28_latent_vs_output_decomposition_freshretailnet.json
```

## Smoke 実行確認

次のコマンドで smoke は完走した。

```bash
uv run decoupled-ts residual-experiment --config configs/2-Exp-28_latent_vs_output_decomposition_smoke.json
```

実行確認済みの variant:

| name | 実行状態 |
|---|---|
| `paper_global_local_residual` | 完走 |
| `four_factor_global_day_hour_residual` | 完走 |
| `four_factor_global_day_hour_interaction_residual` | 完走 |
| `output_decomp_no_center` | 完走 |
| `output_decomp_centered_no_interaction` | 完走 |
| `output_decomp_centered` | 完走 |

Smoke は 1 epoch の synthetic 実行であるため、性能比較には使わない。
本実験の判断には FreshRetailNet config の結果を使う。

## FreshRetailNet 結果

実行コマンド:

```bash
uv run decoupled-ts residual-experiment --config configs/2-Exp-28_latent_vs_output_decomposition_freshretailnet.json
```

### 予測補正

`series_mean` baseline の cell MAE は 0.072146 だった。
全モデルが baseline より低い corrected MAE を出したが、最も良かったのは output decomposition 系だった。

| model | corrected MAE | corrected WAPE | corrected bias | residual R2 |
|---|---:|---:|---:|---:|
| `paper_global_local_residual` | 0.060919 | 0.984469 | -0.3624 | 0.0351 |
| `four_factor_global_day_hour_residual` | 0.061955 | 1.001208 | -0.3920 | -0.0091 |
| `four_factor_global_day_hour_interaction_residual` | 0.061394 | 0.992142 | -0.4559 | -0.0241 |
| `output_decomp_no_center` | 0.061685 | 0.996847 | -0.4024 | 0.0173 |
| `output_decomp_centered_no_interaction` | **0.057189** | **0.924185** | -0.3425 | **0.1984** |
| `output_decomp_centered` | 0.057247 | 0.925127 | -0.4275 | 0.1855 |

この結果は、Exp-27 の読み方を強める。
Exp-27 では、4 成分 latent split は `global/local` residual を安定して上回らなかった。
Exp-28 では、同じ residual target 上で output decomposition + centering が latent split 系より良い corrected MAE を示した。

特に `output_decomp_centered_no_interaction` は `paper_global_local_residual` より corrected MAE を 0.00373 下げた。
これは baseline MAE 0.072146 から見ると、baseline との差分改善をさらに大きくする結果である。

### 高残差 top10

高残差上位 10% では、latent split 系は baseline と同等または悪化した。
一方、centered output decomposition は改善した。

| model | high residual baseline MAE | high residual corrected MAE | 読み方 |
|---|---:|---:|---|
| `paper_global_local_residual` | 0.292300 | 0.292751 | ほぼ同等、わずかに悪化 |
| `four_factor_global_day_hour_residual` | 0.292300 | 0.300422 | 悪化 |
| `four_factor_global_day_hour_interaction_residual` | 0.292300 | 0.303175 | 悪化 |
| `output_decomp_no_center` | 0.292300 | 0.295369 | 悪化 |
| `output_decomp_centered_no_interaction` | 0.292300 | 0.268087 | 改善 |
| `output_decomp_centered` | 0.292300 | **0.265624** | 最良 |

この結果は、output decomposition の価値を「平均 MAE」だけでなく「外れケース補正」としても説明できることを示す。

### hour profile と成分 ablation

Hour profile でも、centered output decomposition が最も安定した。

| model | residual hour profile corr | hour component residual profile corr |
|---|---:|---:|
| `paper_global_local_residual` | 0.8893 | n/a |
| `four_factor_global_day_hour_residual` | 0.8677 | n/a |
| `four_factor_global_day_hour_interaction_residual` | 0.8622 | n/a |
| `output_decomp_no_center` | 0.8574 | 0.8469 |
| `output_decomp_centered_no_interaction` | **0.9962** | **0.9944** |
| `output_decomp_centered` | 0.9923 | 0.9919 |

成分 ablation では、centered output decomposition の hour 成分が最も明確に効いていた。

| model | without global delta | without day delta | without hour delta | without interaction delta |
|---|---:|---:|---:|---:|
| `output_decomp_no_center` | 0.00954 | 0.01012 | 0.02785 | 0.01895 |
| `output_decomp_centered_no_interaction` | 0.00774 | 0.00067 | 0.00507 | 0.00000 |
| `output_decomp_centered` | 0.00703 | 0.00005 | 0.00499 | 0.00000 |

`output_decomp_no_center` では ablation delta は大きいが、component mean が 0 から外れており、主効果と interaction が混ざっている。
実際、`component_interaction_day_mean_abs` と `component_interaction_hour_mean_abs` は 0.0411 で、centering ありのモデルではほぼ 0 だった。
したがって、no-center の大きな interaction delta は「意味のある interaction」ではなく、主効果の混入として読むべきである。

centered モデルでは、day と interaction の寄与は小さく、hour の寄与が安定している。
これはこれまでの FreshRetailNet 結果と整合する。
つまり、実データでは interaction 成分を強く主張するより、`series_mean` residual に残る hour structure を output-level に分解できることを主張する方が妥当である。

### bias の残り方

一方で、centered output decomposition でも corrected bias は残った。

| model | corrected bias |
|---|---:|
| `output_decomp_centered_no_interaction` | -0.3425 |
| `output_decomp_centered` | -0.4275 |

したがって、Exp-28 は raw model の比較として使う。
実運用上の補正性能を議論する場合は、2-Exp-18/19 の calibration と bias 制約の結果を併せて使う。

## 評価指標

主に見る指標は以下である。

| 指標 | 見る理由 |
|---|---|
| `corrected_cell_mae` / `corrected_cell_wape` | baseline correction として効くか |
| `high_residual_top10_corrected_mae` | 外れケース補正に効くか |
| `corrected_cell_bias` | 補正に一方向の偏りがないか |
| `residual_hour_profile_corr` | residual の時間帯 profile を捉えているか |
| `hour_component_residual_profile_corr` | output decomposition の hour component が residual hour profile と対応するか |
| `component_ablation_without_*_mae_delta` | 出力成分を消したときに性能が落ちるか |
| `ablation_zero_z_*_residual_mae_delta` | latent を消したときに性能が落ちるか |

latent split 系では、`ablation_zero_z_*_residual_mae_delta` を見る。
output decomposition 系では、`component_ablation_without_*_mae_delta` と component profile を見る。

## 期待される読み方

FreshRetailNet 結果から、次のように書ける。

```text
Exp-26/27 では、latent を 4 成分に分けるだけでは安定した改善や解釈性は保証されなかった。
一方、output decomposition + centering は、出力成分そのものに構造を課すため、
補正性能と成分解釈を同時に評価できる。
```

今回の結果は、この読み方を支持する。
ただし、次の制約も残る。

```text
Output decomposition は raw model の段階で latent split より良い補正と hour profile 対応を示した。
ただし bias は残るため、最終的な予測補正では calibration や bias 制約を併用する必要がある。
また、FreshRetailNet では interaction 成分の寄与は小さいため、実データの主張は hour structure と high-residual correction を中心に置く。
```

## Proposal への接続

この実験で確認したい論理は、次の 3 段階である。

1. 元論文の `global/local` 表現分離は出発点として自然である。
2. 小売需要では local を day/hour/interaction に分けたいが、latent split だけでは成分の意味が安定しない。
3. そのため、残差の出力値そのものを `series/day/hour/interaction` に分け、centering constraints で読み方を固定する。

したがって、2-Exp-28 は新しい主張を増やす実験ではない。
既存の Exp-26/27 と proposal の主提案を直接つなぐための、最小比較実験である。
