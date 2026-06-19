# 2-Exp-26: Global/Local から Four-Factor Split への段階的拡張

## 背景

これまでの主実験は、直近結果に依存する baseline を置き、その残差 `r = y - b` に注目して `global/day/hour/interaction` 成分を学習する流れだった。

この流れは、baseline が強く、かつ残差に再現可能な構造が残る場合には説明しやすい。一方で、論文上はいきなり「残差に注目し、さらに residual を 4 成分に分ける」と提案すると、元論からの距離が大きく見える。

2-Exp-26 では、まず残差ターゲットを使わず、元論の `global + local` 分解を出発点にする。その上で、local を retail data の構造に合わせて `day/hour/interaction` に分けるだけで何が変わるかを見る。

## 目的

目的は 3 つ。

1. 元論準拠の `global + local` 2 成分モデルを retail forecasting 実験に追加する。
2. 同じ通常予測タスクで、`global + local` と `global + day + hour + interaction` を比較する。
3. 両方に同じ反実仮想 global 正則化を入れ、差分を「反実仮想正則化の有無」ではなく「local 表現の分割方法」として読めるようにする。

## 実験の位置づけ

この実験は、最終提案である residual correction 実験の前段に置く。

論文上の流れは次のようにできる。

```text
元論: one global latent + one local latent sequence
  ↓
本研究の第一段階: retail 構造に合わせて local を day/hour/interaction に分割
  ↓
本研究の第二段階: 強い baseline で説明済みの部分を除き、残差成分に同じ分解を適用
```

この位置づけにより、残差実験に入る前に「local を細かく分けること自体」の意味を示せる。

## 比較 variant

| variant | latent 構造 | 目的 |
|---|---|---|
| `paper_global_local` | `z_global + z_local` | 元論準拠の 2 成分 reference |
| `four_factor_global_day_hour` | `z_global + z_day + z_hour` | local を day/hour に分ける中間条件 |
| `four_factor_global_day_hour_interaction` | `z_global + z_day + z_hour + z_interaction` | 提案の 4 成分 split |

smoke config には動作確認用に `baseline_flatten_mlp` も含める。

## 追加したモデル要素

通常の `RetailMultiGrainModel` に `z_local` を追加した。

```text
z_global: 系列全体の静的な傾向
z_local: day-hour cell ごとの local latent
z_day: 日単位の local latent
z_hour: 時間帯単位の local latent
z_interaction: day x hour の相互作用 latent
```

`paper_global_local` では、decoder の入力は次になる。

```text
[z_global, z_local(d, h)]
```

`four_factor_global_day_hour_interaction` では、decoder の入力は次になる。

```text
[z_global, z_day(d), z_hour(h), z_interaction(d, h)]
```

未来予測では、`z_day` と `z_local` は履歴末尾の直近 7 日平均を使う。`z_hour` は履歴全体から推定した時間帯表現をそのまま使う。

## 反実仮想 global 正則化

Exp-26 では、元論準拠を強めるため、両方の比較モデルに同じ counterfactual global regularization を入れる。

処理は次の通り。

1. batch 内で別系列の `z_global` を選ぶ。
2. 元の local 側成分と差し替えた `z_global_cf` で履歴系列を decode する。
3. decode した反実仮想系列を global encoder に戻す。
4. 再 encode された global が、元の `z_global` よりも `z_global_cf` に近くなるように penalize する。

実装上の loss は距離ベースの近似である。

```text
loss_cf = softplus(
  distance(reencode_global(x_cf), z_global_cf)
  - distance(reencode_global(x_cf), z_global_original)
  + margin
)
```

config では次で制御する。

```json
{
  "counterfactual_regularization": {
    "enabled": true,
    "weight": 0.05,
    "margin": 0.0
  }
}
```

学習ログには次が出る。

```text
train_loss_counterfactual_global
valid_loss_counterfactual_global
train_loss_counterfactual
valid_loss_counterfactual
```

## 元論との差分

元論と現在の実装は完全には同じではない。主な差分は次の通り。

| 項目 | 元論 | Exp-26 実装 |
|---|---|---|
| 学習形式 | VAE / probabilistic decoder | deterministic forecasting model |
| local latent | GP prior を持つ window latent sequence | cell-level `z_local` または `z_day/z_hour/z_interaction` |
| global latent | posterior distribution `q(z_g | X)` | deterministic encoder output |
| reconstruction | decoder likelihood | reconstruction loss + history sales loss |
| counterfactual | `z_g*` で生成し、global encoder の likelihood preference を penalize | `z_g*` で生成し、再 encode global が `z_g*` に近くなる距離 loss |
| regularization | KL local/global + counterfactual | covariance decouple + counterfactual distance penalty |
| forecasting | GP conditional local prediction | recent local/day average + hour latent reuse |
| target | time-series reconstruction / forecasting | future total sales forecasting |

重要なのは、Exp-26 は「元論そのものの完全再現」ではなく、元論の主要な構造的仮定を retail forecasting に移植した比較実験である、という点である。

移植している要素:

- `global` と `local` の分離
- local sequence を持つ reference model
- global を差し替えた counterfactual regularization
- 表現を分けた上で forecasting を行う構成

移植していない要素:

- VAE の posterior sampling
- GP prior / GP posterior による local latent の厳密な時系列外挿
- decoder likelihood に基づく反実仮想 preference loss

## 比較で見る指標

主指標:

- `mae`
- `rmse`
- `wape`
- `bias`

補助指標:

- `probe_z_global_subgroup_accuracy`
- `probe_z_day_weekday_accuracy`
- `probe_z_day_holiday_accuracy`
- `probe_z_day_discount_mae`

`paper_global_local` は active latent が `z_global` と `z_local` のため、day probe は基本的に出ない。`four_factor_*` では `z_day` が active なので day probe が出る。

## 良い結果

次の結果なら、4 成分 split を残差実験へ進める導入として使いやすい。

- `four_factor_global_day_hour_interaction` が `paper_global_local` より forecast 指標で同等以上。
- `z_day` probe が weekday / holiday / discount を拾う。
- `z_global` subgroup probe が維持される。
- counterfactual loss が学習ログに入り、訓練が破綻しない。

この場合、論文では次のように説明できる。

```text
元論の global/local 分解を retail data に移植した上で、
local を day/hour/interaction に分けると、予測性能を保ちつつ、
retail 固有の局所要因をより明示的に表現できる。
```

## 悪い結果

次の結果なら、主張を弱める必要がある。

- `paper_global_local` が明確に強く、4 成分 split が悪化する。
- day/hour probe が意味のある情報を拾わない。
- counterfactual loss を入れると forecast が大きく悪化する。

この場合は、4 成分 split は通常予測ではなく residual correction のための inductive bias として位置づける。

```text
通常の future total forecasting では、細分化した local 表現が常に有利とは限らない。
一方、強い baseline で説明済みの成分を除いた residual target では、
day/hour/interaction の帰納バイアスが有効になる。
```

## 実行コマンド

smoke:

```bash
uv run decoupled-ts retail-experiment --config configs/2-Exp-26_global_local_to_four_factor_smoke.json
```

synthetic:

```bash
uv run decoupled-ts retail-experiment --config configs/2-Exp-26_global_local_to_four_factor_synthetic.json
```

FreshRetailNet:

```bash
uv run decoupled-ts retail-experiment --config configs/2-Exp-26_global_local_to_four_factor_freshretailnet.json
```

## 出力

```text
runs/2-Exp-26_global_local_to_four_factor_smoke/summary.csv
runs/2-Exp-26_global_local_to_four_factor_synthetic/summary.csv
runs/2-Exp-26_global_local_to_four_factor_freshretailnet/summary.csv
```

各 variant 配下には次が出る。

```text
history.jsonl
metrics.json
predictions.csv
z_global.npy
z_local.npy
z_day.npy
z_hour.npy
z_interaction.npy
```

保存される latent は active なものだけである。

## smoke 確認

smoke は実行済み。

```bash
uv run decoupled-ts retail-experiment --config configs/2-Exp-26_global_local_to_four_factor_smoke.json
```

確認したこと:

- `paper_global_local` が最後まで学習・評価できる。
- `four_factor_global_day_hour_interaction` が最後まで学習・評価できる。
- `train_loss_counterfactual_global` と `valid_loss_counterfactual_global` が `history.jsonl` に記録される。
- active latent だけが保存される。

smoke は 1 epoch / 小規模データなので性能差の解釈には使わない。構文と学習ループの確認用である。

## 論文上の使い方

この実験は、提案の導入を滑らかにするために使う。

本文では大きく次のように書ける。

```text
We first instantiate the original global/local decomposition in the retail forecasting setting.
Then, keeping the same counterfactual global regularization, we replace the undifferentiated local latent with day, hour, and day-hour interaction latents.
This isolates the effect of the proposed retail-specific local factorization before moving to residual correction.
```

日本語では次の整理になる。

```text
まず元論の global/local 分解を通常予測タスクで再現する。
その上で、反実仮想 global 正則化は同じまま、local 表現だけを day/hour/interaction に分ける。
これにより、残差補正に進む前に、local 表現の細分化そのものの効果を切り出して検証する。
```
