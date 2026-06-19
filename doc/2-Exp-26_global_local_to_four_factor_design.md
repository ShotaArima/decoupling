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

## FreshRetailNet 結果

実行コマンド:

```bash
uv run decoupled-ts retail-experiment --config configs/2-Exp-26_global_local_to_four_factor_freshretailnet.json
```

結果:

| model | best epoch | valid loss | MAE | RMSE | WAPE | bias | z_global subgroup acc | z_day weekday acc | z_day holiday acc | z_day discount MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `paper_global_local` | 6 | 8.1246 | 1.4850 | 4.3703 | 0.6233 | -0.4676 | 0.0000 | - | - | - |
| `four_factor_global_day_hour` | 11 | 7.8172 | 1.4611 | 4.3013 | 0.6133 | -0.3787 | 0.0000 | 0.1575 | 0.5458 | 0.1014 |
| `four_factor_global_day_hour_interaction` | 4 | 7.8440 | 1.4931 | 4.2815 | 0.6267 | -0.2662 | 0.0000 | 0.2124 | 0.5266 | 0.1038 |

`paper_global_local` に対する差分:

| model | MAE delta | RMSE delta | WAPE delta | bias delta |
|---|---:|---:|---:|---:|
| `four_factor_global_day_hour` | -0.0239 | -0.0691 | -0.0101 | +0.0889 |
| `four_factor_global_day_hour_interaction` | +0.0080 | -0.0889 | +0.0034 | +0.2014 |

`four_factor_global_day_hour` は `paper_global_local` より MAE / RMSE / WAPE / valid loss のすべてで少し良い。WAPE の改善は 0.6233 から 0.6133 で、絶対値では 0.0101、相対では約 1.6% の改善である。bias も -0.4676 から -0.3787 に改善している。

一方、`four_factor_global_day_hour_interaction` は RMSE と bias は改善するが、MAE と WAPE は `paper_global_local` より悪い。したがって、この通常予測設定では interaction まで足すことが安定した改善にはつながっていない。

## FreshRetailNet の読み取り

主に言えることは次の 3 点である。

1. `global/local` から `global/day/hour` への分割は、通常の future total forecasting でも小さな改善を示した。
2. `day/hour/interaction` まで入れると、RMSE と bias は改善するが、MAE/WAPE は悪化した。
3. probe は弱く、特に `z_global` は subgroup を識別できていない。

### day/hour 分割について

`four_factor_global_day_hour` は、元論準拠の `paper_global_local` より少し良い。これは、local を完全に自由な cell-level latent として持つより、retail data の構造に合わせて day と hour に分けた方が、通常予測でも帰納バイアスとして効く可能性を示す。

ただし改善幅は大きくない。したがって、論文では次の程度の控えめな主張が妥当である。

```text
通常予測タスクでも、local 表現を day/hour に分けることで global/local reference をわずかに改善した。
これは、retail demand の局所変動を日単位要因と時間帯要因に分ける設計が、元論の local latent を単に細分化したものとして自然に導入できることを示す。
```

### interaction について

interaction 付きモデルは valid loss では `paper_global_local` より良いが、test MAE/WAPE では悪い。これは、通常の future total sales という集約ターゲットでは、day x hour interaction の細かい cell-level 情報が過剰になっている可能性がある。

一方で RMSE と bias は改善している。大きな誤差や系統的な過小予測には interaction が効いている可能性があるが、平均絶対誤差ではその複雑さが不利に出ている。

この結果は、interaction 成分を「通常予測で常に有利な追加要素」として主張するには弱い。むしろ、次のように residual 実験への橋渡しとして使うのが自然である。

```text
day/hour split は通常予測でも小さく有効だったが、interaction は集約予測では安定した改善を示さなかった。
そのため、interaction の価値は、強い baseline で主効果を除いた後に残る cell-level residual 構造で検証する必要がある。
```

### probe について

`probe_z_global_subgroup_accuracy` はすべて 0.0 である。train には 9 subgroup、test には 3 subgroup しかなく、overlap も 3 であるため、subgroup probe 自体がかなり厳しい設定になっている。majority accuracy は 0.582 であり、probe が subgroup をうまく拾えていないことは明確である。

これは 2 つの可能性を示す。

- `z_global` が city/subgroup よりも store-product 固有の需要水準や販売量情報を優先している。
- FreshRetailNet の train/test subgroup 分布が probe に向いておらず、global 表現の評価として city classification が不安定である。

したがって、Exp-26 の FreshRetailNet 結果では、`z_global` の解釈性を subgroup probe で強く主張しない方がよい。

`z_day` probe も強くはない。weekday accuracy は 0.1575 / 0.2124 で、7 クラスのランダム水準 0.1429 よりは上だが高くない。holiday accuracy も 0.526 前後で、class imbalance を考えると強い証拠とは言いにくい。discount MAE は 0.10 程度で、販促強度をある程度反映している可能性はあるが、単独で強い主張にはしない。

## FreshRetailNet 結果からの結論

この結果は、Exp-26 の目的に対して部分的に成功している。

成功している点:

- 元論準拠の `global + local` reference を実装し、同じ反実仮想正則化込みで比較できた。
- `global + day + hour` は `global + local` より少し良く、local を retail 構造に沿って分ける導入として使える。
- 残差ターゲットに進む前に、通常予測で day/hour 分割の小さな有効性を示せた。

弱い点:

- interaction まで入れた 4 成分版は、通常予測の MAE/WAPE では改善していない。
- probe は全体に弱く、表現解釈性の主張はこの実験だけでは不十分である。
- `z_global` subgroup accuracy が 0.0 なので、global 表現が city/subgroup を表すとは言えない。

したがって、論文上の整理は次が妥当である。

```text
Exp-26 は、元論の global/local 分解から本研究の day/hour 分割へ進むための橋渡しである。
FreshRetailNet の通常予測では、day/hour 分割は global/local reference を小幅に改善した。
一方、interaction 成分と表現解釈性は通常予測だけでは明確に支持されない。
このため、以降の residual 実験では、強い baseline で説明済みの主効果を除いた後に、
hour/interaction 成分が残差補正と component analysis で意味を持つかを検証する。
```

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
