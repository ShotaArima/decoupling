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

結果が良い場合、次のように書ける。

```text
Exp-26/27 では、latent を 4 成分に分けるだけでは安定した改善や解釈性は保証されなかった。
一方、output decomposition + centering は、出力成分そのものに構造を課すため、
補正性能と成分解釈を同時に評価できる。
```

結果が弱い場合でも、次のように書ける。

```text
Output decomposition は万能な補正器ではない。
ただし、latent split と違い、失敗時にもどの出力成分が寄与していないかを ablation と profile で診断できる。
このため、本研究の貢献は単純な精度改善ではなく、baseline 後の residual structure を診断可能な形に分解する点にある。
```

## Proposal への接続

この実験で確認したい論理は、次の 3 段階である。

1. 元論文の `global/local` 表現分離は出発点として自然である。
2. 小売需要では local を day/hour/interaction に分けたいが、latent split だけでは成分の意味が安定しない。
3. そのため、残差の出力値そのものを `series/day/hour/interaction` に分け、centering constraints で読み方を固定する。

したがって、2-Exp-28 は新しい主張を増やす実験ではない。
既存の Exp-26/27 と proposal の主提案を直接つなぐための、最小比較実験である。
