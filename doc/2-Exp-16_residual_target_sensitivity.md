# 2-Exp-16: Residual Target Sensitivity on FreshRetailNet

## 目的

FreshRetailNet では `same_hour_recent_mean` が強く、残差が「学習すべき構造」ではなく、ほとんど説明し終えた後の小さい誤差になっている可能性がある。

この実験では、残差を作る基準成分 `b` を変え、day / hour / interaction が残りやすい target を探す。

論文内での位置づけは次のように整理する。

- synthetic: 分離表現が成立することを示す主実験
- synthetic difficulty: どの条件で成立し、どの条件で失敗するかを示す実験
- FreshRetailNet: 強い baseline と実データノイズの下で、どこまで外部妥当性があるかを見る限界検証

FreshRetailNet で常に改善しない場合でも、それは失敗だけではなく「baseline が残差構造を消すと分離表現の余地が小さくなる」という適用条件の証拠として扱う。

## 比較する residual target

| target | 目的 | 良い結果 | 悪い結果 |
| --- | --- | --- | --- |
| `series_mean` | 系列平均だけを引き、曜日・時間帯の構造を残す | residual probe R2 と `z_day/z_hour` probe が上がる | 残差が大きすぎて補正後 MAE が悪化する |
| `weekday_same_hour_mean` | 曜日同時間の平均を引き、短期変動や相互作用を残す | interaction probe や high residual top10 が改善する | day/hour の構造まで消えて probe が低い |
| `same_hour_recent_mean_d1` | 直近 1 日だけで基準を作り、短期変動への過適合を確認する | 短期の外れケースで補正が効く | baseline が不安定で残差がノイズ化する |
| `same_hour_recent_mean_d3` | 直近 3 日で基準を作り、d1 と d7 の中間を見る | probe と補正のバランスがよい | d7 と同じく構造が残らない |
| `same_hour_recent_mean_d7` | これまでの主 baseline を対照として置く | 他 target と同水準なら強い対照として使える | 以前と同じく補正余地が小さい |
| `log1p_same_hour_recent_mean_d7` | 売上スケール差を圧縮し、低売上系列の構造を見やすくする | WAPE や high residual top10 が安定する | 大きな外れ値を圧縮しすぎて補正力が落ちる |

## 新しい subset 選別

これまでの `residual_structure_score` は、残差分散や固定効果が大きい系列を拾っていた。しかし、train で大きく見える構造が validation でも再現するとは限らない。

今回の `residual_repro_score` は、系列ごとに履歴期間を前半 train / 後半 validation に分け、次が同じ向きに再現するかを見る。

- residual の hour 固定効果の再現性
- residual の weekday 固定効果の再現性
- discount 有無による residual 差の再現性
- stockout 近傍と通常時の residual 差の再現性

良い subset は、単に荒い系列ではなく「同じ構造が履歴内で繰り返し出る系列」である。

## 実験構成

### 16-A: target sensitivity

Config:

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-16_freshretailnet_target_sensitivity.json
```

比較対象:

- all 系列での target 差
- `residual_repro_score` 上位 subset での target 差

見る指標:

- `residual_mae_mean`
- `baseline_cell_mae_mean`
- `corrected_cell_mae_mean`
- `calibrated_corrected_cell_mae_mean`
- `high_residual_top10_baseline_mae_mean`
- `high_residual_top10_corrected_mae_mean`
- `probe_z_day_weekday_accuracy_mean`
- `probe_z_hour_hour_accuracy_mean`
- `component_ablation_without_day_mae_delta_mean`
- `component_ablation_without_hour_mae_delta_mean`
- `component_ablation_without_interaction_mae_delta_mean`

### 16-B: aggregation sensitivity

Config:

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-16_freshretailnet_aggregation_sensitivity.json
```

比較対象:

- 店舗商品: `["store_id", "product_id"]`
- 店舗第三カテゴリ: `["store_id", "third_category_id"]`
- 店舗第二カテゴリ: `["store_id", "second_category_id"]`
- 店舗全体: `["store_id"]`

目的は、店舗商品単位ではゼロ売上や欠品の影響が強すぎる場合に、少し集約した粒度で day/hour/interaction が見えるかを確認すること。

良い結果は、カテゴリまたは店舗全体に集約したときに residual probe や high residual top10 の改善が出ること。悪い結果は、集約しても probe が上がらず、補正も改善しないこと。この場合は、FreshRetailNet の残差分離にはより明示的な需要・在庫・価格モデルが必要だと考える。

## 成功条件

主成功:

- 少なくとも一部 target で residual probe R2 が上がる
- `z_day` または `z_hour` probe が上がる
- calibrated high residual top10 で baseline を上回る

補助成功:

- `residual_repro_score` 上位 subset が all 系列より安定して改善する
- 集約系列で店舗商品単位より改善する
- 改善がなくても、強い baseline や過度な集約が残差構造を消すことを示せる

## 実行順

まず smoke で実装を確認する。

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-16_smoke.json
```

次に FreshRetailNet の target 感度を見る。

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-16_freshretailnet_target_sensitivity.json
```

最後に集約粒度を変える。

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-16_freshretailnet_aggregation_sensitivity.json
```

## 結果の読み方

`corrected_cell_mae` が baseline より少し悪くても、probe と ablation が上がるなら「同水準の予測性能で分離表現を作れた」と読める。

一方で、probe も ablation も上がらず、calibration 後も high residual top10 が改善しない場合は、その residual target は分離表現の学習対象として弱い。

FreshRetailNet で一番重要なのは、全体平均の MAE だけではない。どの target で構造が残り、どの target で消えるかを示すことで、提案法の適用条件を狭くても明確にする。
