# EXP-003: 多粒度分離仮説の検証実験設計

## 目的

添付メモで整理された以下の仮説を、予測性能と表現解釈性の両面から検証します。

```text
小売需要時系列では、時間変動要因を単一の local 表現にまとめるよりも、
日付単位の変動 z_day と時間帯単位の反復パターン z_hour に分けて学習する方が、
需要予測または潜在表現の解釈性において有利である。
```

この実験では、教授から指摘されやすい以下の問いに答えることを重視します。

- 何を解く研究なのか
- day/hour の2軸に分ける必然性はあるのか
- 単なる曜日・時間帯特徴量追加ではないのか
- 潜在表現が本当に役割分担しているのか
- 欠品がある売上を需要として扱ってよいのか

## 実験番号

この実験を `EXP-003` とします。出力先・設定ファイル・ドキュメント名も `EXP-003` で揃えます。

## 追加したコード

`EXP-003` は既存の `retail-experiment` ランナーを拡張して実行します。

主な追加点は以下です。

| 追加機能 | ファイル | 目的 |
|---|---|---|
| naive baseline | `src/decoupled_ts/retail_experiments.py` | 単純予測に勝てているか確認 |
| `valid_loss_forecast` による checkpoint 選択 | `src/decoupled_ts/retail_experiments.py` | 再構成損失ではなく需要予測指標でモデル選択 |
| `predictions.csv` 保存 | `src/decoupled_ts/retail_experiments.py` | 過小予測・需要帯別誤差を分析 |
| `data_audit.json` 保存 | `src/decoupled_ts/retail_experiments.py` | ゼロ率・観測率・target 分布を確認 |
| probe 診断の追加 | `src/decoupled_ts/retail_experiments.py` | class overlap と majority baseline を併記 |
| variant ごとの loss override | `src/decoupled_ts/retail_experiments.py` | decouple loss あり/なしを同じ config 内で比較 |

## 設定ファイル

| config | データ | 目的 |
|---|---|---|
| `configs/EXP-003_hypothesis_smoke.json` | synthetic small | 実装の動作確認 |
| `configs/EXP-003_hypothesis_synthetic.json` | synthetic retail | 仮説が成り立つ条件での検証 |
| `configs/EXP-003_hypothesis_freshretailnet.json` | FreshRetailNet | 実データでの検証 |

## 実行コマンド

```bash
uv run decoupled-ts retail-experiment --config configs/EXP-003_hypothesis_smoke.json
uv run decoupled-ts retail-experiment --config configs/EXP-003_hypothesis_synthetic.json
uv run decoupled-ts retail-experiment --config configs/EXP-003_hypothesis_freshretailnet.json
```

## 比較するモデル

| model | 目的 |
|---|---|
| `naive_last_day` | 直近1日の売上を将来にコピーする単純 baseline |
| `naive_recent_mean` | 直近 `recent_days` の日平均を使う baseline |
| `naive_same_hour_recent_mean` | 同じ時間帯の直近平均を使う baseline |
| `feature_flatten_mlp` | 曜日・時間帯などを特徴量として入れた MLP |
| `global_only` | 系列固有成分だけでどこまで説明できるか |
| `global_day` | 日付方向 local の寄与を見る |
| `global_hour` | 時間帯方向 local の寄与を見る |
| `proposed_no_decouple` | `z_global + z_day + z_hour` の構造だけの効果を見る |
| `proposed_with_decouple` | decouple loss の追加効果を見る |
| `proposed_interaction` | 日付 x 時間帯の相互作用を追加した場合を見る |

## 検証したい仮説

### H1: 単純特徴量追加だけではないか

比較:

```text
feature_flatten_mlp
vs
proposed_no_decouple
vs
proposed_with_decouple
```

期待:

```text
提案モデルが feature_flatten_mlp と同等以上の予測性能を示し、
さらに probe や latent 可視化で表現の役割分担を示せる。
```

失敗した場合:

```text
day/hour 分離よりも、単純な特徴量追加で十分な可能性がある。
この場合は、予測性能ではなく解釈性・反実仮想操作で差を示す必要がある。
```

### H2: day と hour を分ける意味があるか

比較:

```text
global_only
global_day
global_hour
proposed_with_decouple
```

期待:

```text
proposed_with_decouple が global_only / global_day / global_hour より良い。
または予測性能が同等でも、z_day と z_hour の probe 結果が異なる。
```

失敗した場合:

```text
実データでは系列固有成分 z_global が支配的で、
day/hour の追加効果が限定的である可能性がある。
```

### H3: decouple loss は有効か

比較:

```text
proposed_no_decouple
vs
proposed_with_decouple
```

期待:

```text
decouple loss により、予測性能を大きく落とさず probe の役割分担が改善する。
```

失敗した場合:

```text
現在の covariance penalty が弱い、または予測タスクに対して過剰制約になっている可能性がある。
```

### H4: naive baseline に勝てているか

比較:

```text
naive_last_day
naive_recent_mean
naive_same_hour_recent_mean
feature_flatten_mlp
proposed_with_decouple
```

期待:

```text
提案モデルが naive baseline より WAPE / MAE を改善する。
```

失敗した場合:

```text
需要系列が短期平均で十分説明できる、またはモデルの future day 表現が弱い可能性がある。
```

### H5: 欠品・観測制約の影響はどの程度か

確認:

```text
data_audit.json
predictions.csv
bias
sales_observed_rate
sales_zero_rate
```

期待:

```text
欠品率やゼロ率が高い系列で、誤差や bias が悪化していないか確認できる。
```

今回の実装では、まず監査値と予測 CSV を保存する段階までを行います。次の段階で、欠品重みあり/なしを明示的な variant として追加します。

## 出力ファイル

各 run の root に以下を保存します。

| file | 内容 |
|---|---|
| `run.log` | 実験全体ログ |
| `resolved_config.json` | 実行時 config |
| `data_audit.json` | データ構造・ゼロ率・観測率・target 分布 |
| `summary.json` | 全 variant の結果 |
| `summary.csv` | 全 variant の結果表 |

各 variant ディレクトリに以下を保存します。

| file | 内容 |
|---|---|
| `config.json` | variant 固有設定 |
| `history.jsonl` | epoch ごとの train/valid loss |
| `metrics.json` | test metrics |
| `predictions.csv` | `y_true`, `y_pred`, `error`, `abs_error` |
| `best.pt` | 学習モデルの checkpoint |
| `z_global.npy` | test set の global latent |
| `z_day.npy` | test set の day latent |
| `z_hour.npy` | test set の hour latent |
| `subgroup.npy` | probe 用 label |

naive baseline は学習しないため、`history.jsonl`, `best.pt`, latent `.npy` は生成されません。

## Smoke 実行確認

以下のコマンドで実行確認済みです。

```bash
uv run decoupled-ts retail-experiment --config configs/EXP-003_hypothesis_smoke.json
```

生成を確認した主なファイル:

```text
runs/EXP-003_hypothesis_smoke/data_audit.json
runs/EXP-003_hypothesis_smoke/summary.csv
runs/EXP-003_hypothesis_smoke/summary.json
runs/EXP-003_hypothesis_smoke/*/predictions.csv
```

実行ログ上の時刻は以下です。

| 項目 | 時刻 |
|---|---|
| start | 2026-05-30 05:54:11 |
| complete | 2026-05-30 05:54:54 |

データ監査結果は以下です。

| item | value |
|---|---:|
| train examples | 90 |
| valid examples | 19 |
| test examples | 19 |
| input dim | 10 |
| days | 14 |
| hours | 24 |
| forecast days | 1 |
| sales zero rate | 0.1481 |
| sales observed rate | 0.9340 |
| target mean | 59.2444 |
| target std | 27.2386 |
| target p50 | 55.0000 |
| target p90 | 91.5000 |
| subgroup classes | 4 |

Smoke の結果は以下です。

| model | best valid loss | WAPE | MAE | RMSE | bias | probe acc | majority acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| `naive_last_day` | - | 0.2693 | 14.4737 | 18.6928 | 0.2341 | - | - |
| `naive_recent_mean` | - | **0.1447** | **7.7783** | **11.0444** | 0.1402 | - | - |
| `naive_same_hour_recent_mean` | - | 0.1482 | 7.9619 | 11.2194 | 0.1436 | - | - |
| `feature_flatten_mlp` | 1247.4939 | 0.9345 | 50.2147 | 54.2574 | -0.9345 | - | - |
| `proposed_no_decouple` | 906.2604 | 0.6757 | 36.3116 | 41.9582 | -0.6755 | 0.4211 | 0.3684 |
| `proposed_with_decouple` | 843.0789 | 0.6355 | 34.1495 | 39.6998 | -0.6265 | 0.4737 | 0.3684 |

Smoke は 1 epoch のため、学習モデルの性能評価には使いません。コードが通り、必要な成果物が生成されることを確認するための実験です。

ただし、動作確認段階でも次の傾向は見えています。

- `proposed_with_decouple` は `proposed_no_decouple` より WAPE / MAE / RMSE が良く、probe accuracy も majority baseline を上回りました。
- 1 epoch では学習モデルが naive baseline に大きく負けており、強い過小予測 bias が残っています。
- 本評価には `EXP-003_hypothesis_synthetic.json` または `EXP-003_hypothesis_freshretailnet.json` の複数 epoch 実行結果が必要です。

## 結果の読み方

主指標は以下です。

| 指標 | 意味 |
|---|---|
| WAPE | 需要予測の主指標 |
| MAE | 平均絶対誤差 |
| RMSE | 大きな外れ誤差の確認 |
| bias | 過大予測・過小予測の方向 |
| probe accuracy | latent が subgroup 情報を持つか |
| majority accuracy | probe の最低比較対象 |
| overlap classes | train/test で probe label が噛み合っているか |

特に FreshRetailNet では、WAPE だけでなく bias を必ず確認します。過去の `EXP-002` では latent model が WAPE を改善する一方で、負の bias が大きく、過小予測に寄っている可能性がありました。

## 期待される結論パターン

### 最も望ましい結果

```text
proposed_with_decouple が naive / feature_flatten_mlp / global_only を上回り、
probe でも z_global, z_day, z_hour の役割差が確認できる。
```

この場合、研究主張はかなり強くなります。

### 予測性能は同等だが probe が良い

```text
予測性能では大差がないが、
z_day は日付要因、z_hour は時間帯要因、z_global は系列固有要因をより明確に保持する。
```

この場合、研究の軸は「高精度予測」ではなく「予測性能を保った解釈可能な表現分解」に置きます。

### global_only が最良

```text
FreshRetailNet では系列固有性が支配的で、
今回の設定では day/hour 分離の追加効果は限定的。
```

この場合も失敗ではありません。次に、店舗×商品ではなく店舗×カテゴリへ集約する、未来日付の既知特徴を使う、欠品重みを強める、という改善方針につながります。

## 次に追加したい実験

1. 欠品重みあり/なしの明示的比較
2. 店舗×カテゴリ単位への集約
3. `z_day` から weekday / holiday / discount を予測する probe
4. `z_hour` の hour pattern heatmap
5. `z_day` swap / `z_hour` swap による反実仮想可視化
6. valid WAPE による checkpoint 選択
