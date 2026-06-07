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

## 結果メモ

### 16-A: target sensitivity

3 seed の平均では、`series_mean_all` が最も明確に改善した。

| scenario | model | baseline MAE | corrected MAE | high residual baseline MAE | high residual corrected MAE | residual R2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `series_mean_all` | `output_decomp_bias_loss_calibrated` | 0.0697 | 0.0519 | 0.2788 | 0.2026 | 0.3840 |
| `series_mean_all` | `output_decomp_centered` | 0.0697 | 0.0524 | 0.2788 | 0.2124 | 0.3549 |
| `same_hour_recent_mean_d7_all` | `output_decomp_centered` | 0.0580 | 0.0564 | 0.2534 | 0.2438 | 0.0688 |
| `log1p_same_hour_recent_mean_d7_all` | `output_decomp_centered` | 0.0571 | 0.0545 | 0.2553 | 0.2534 | 0.0139 |
| `weekday_same_hour_mean_all` | `output_decomp_centered` | 0.0420 | 0.0427 | 0.1955 | 0.1954 | 0.0032 |

解釈:

- `series_mean` は強すぎない基準成分として機能しており、残差に学習可能な day/hour/interaction 構造を残している。
- `weekday_same_hour_mean` は baseline として強く、残差がほぼ小さい誤差になっている。補正しても baseline を超えにくい。
- `same_hour_recent_mean_d7` は以前よりは改善しているが、`series_mean` ほどの余地はない。
- `log1p_same_hour_recent_mean_d7` は全体 MAE を改善するが、高残差 top10 では改善が弱い。大きい外れ値を圧縮しすぎている可能性がある。
- `residual_repro_score` 上位 subset は、今回の設計では改善 subset を安定して選べていない。特に `same_hour_recent_mean_d3_repro_top` は全体 MAE が悪化した。

成分 ablation では、`series_mean_all` の `output_decomp_bias_loss_calibrated` で hour 成分の寄与が大きい。

| scenario | model | day delta | hour delta | interaction delta |
| --- | --- | ---: | ---: | ---: |
| `series_mean_all` | `output_decomp_bias_loss_calibrated` | 0.0017 | 0.0102 | 0.0010 |
| `series_mean_all` | `output_decomp_centered` | 0.0009 | 0.0083 | 0.0006 |
| `same_hour_recent_mean_d7_all` | `output_decomp_centered` | 0.0015 | 0.0009 | 0.0013 |

これは、FreshRetailNet のこの設定では「曜日」よりも「時間帯」構造が残差補正に効いていることを示す。

### 16-B: aggregation sensitivity

集約粒度では、店舗第三カテゴリ + `series_mean` が最も良い。

| scenario | model | baseline MAE | corrected MAE | high residual baseline MAE | high residual corrected MAE | residual R2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `store_third_category_series_mean_repro_top` | `output_decomp_centered` | 0.0811 | 0.0678 | 0.3138 | 0.2704 | 0.1951 |
| `store_third_category_series_mean_repro_top` | `output_decomp_bias_loss_calibrated` | 0.0811 | 0.0681 | 0.3138 | 0.2624 | 0.2050 |
| `store_product_same_hour_d7` | `output_decomp_centered` | 0.0580 | 0.0567 | 0.2534 | 0.2434 | 0.0703 |
| `store_total_log1p_same_hour_d7_repro_top` | `output_decomp_centered` | 0.7375 | 0.7154 | 2.4463 | 2.3317 | 0.0378 |
| `store_second_category_weekday_repro_top` | `output_decomp_centered` | 0.0724 | 0.0750 | 0.3447 | 0.3459 | -0.0099 |

解釈:

- 店舗第三カテゴリ集約では、店舗商品単位よりも残差に安定した構造が残る。
- 店舗第二カテゴリ + weekday baseline は悪化しており、baseline が構造を消しすぎているか、集約粒度が粗すぎる可能性がある。
- 店舗全体 + log1p は改善するが、スケールが大きく、主実験にするには別枠で扱う必要がある。

## 現時点の結論

2-Exp-16 の結果は、FreshRetailNet を「全 target でうまくいかない外部データ」ではなく、「baseline と集約粒度を選ぶと残差分離の余地が出る外部データ」として扱えることを示している。

論文上の主張としては、次が最も筋がよい。

1. `same_hour_recent_mean` は強すぎるため、残差分離の主実験 target には向きにくい。
2. `series_mean` では residual R2、補正 MAE、高残差 top10 が明確に改善する。
3. 成分 ablation では hour 成分の寄与が最も大きく、FreshRetailNet では時間帯構造の残差補正が中心になる。
4. 店舗第三カテゴリへ集約すると、店舗商品単位より残差構造が見えやすくなる。
5. ただし `residual_repro_score` はまだ selector として弱く、改善系列を安定して選ぶには再設計が必要である。

## 次に行うべき実験

次は `2-Exp-17` として、`series_mean_all` と `store_third_category_series_mean_repro_top` に絞った本実験に進む。

目的:

- 2-Exp-16 で見えた改善が偶然ではないことを確認する
- `output_decomp_centered` と `output_decomp_bias_loss_calibrated` のどちらを論文の FreshRetailNet 主モデルにするか決める
- hour 成分が本当に効いているかを可視化と ablation で確認する

推奨比較:

- `series_mean_all`
- `store_third_category_series_mean_repro_top`
- 対照として `same_hour_recent_mean_d7_all`
- 負例として `weekday_same_hour_mean_all`

見るべきもの:

- 5 seed 以上の平均と標準偏差
- corrected MAE / RMSE / WAPE
- high residual top10 MAE
- `without_hour` ablation delta
- hour component heatmap
- residual hour profile と predicted hour component の相関
- calibration 前後の bias

成功条件:

- `series_mean_all` で corrected MAE が baseline より安定して改善する
- high residual top10 でも改善する
- `without_hour` ablation delta が一貫して正になる
- heatmap で時間帯成分が解釈可能な形を持つ

失敗条件:

- seed を増やすと改善が消える
- 改善はあるが bias が大きすぎる
- hour ablation が効かず、単なる global 補正で説明できる
- 集約粒度を変えると結果が不安定になる
