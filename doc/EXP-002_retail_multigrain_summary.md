# EXP-002: 小売向け多粒度 Local 分離実験サマリー

## 実験目的

元論文の「1サンプル内の static / dynamic 分離」を、小売需要時系列向けに拡張し、動的成分を以下に分ける効果を確認しました。

- `z_global`: 店舗・商品に固有な静的成分
- `z_day`: 日付単位で変化する成分
- `z_hour`: 時間帯単位で繰り返される成分
- `z_interaction`: 日付 x 時間帯の相互作用成分

中心となる問いは次です。

```text
小売時系列における local 表現を単一の系列表現ではなく、
day-level local と hour-level local に分けることで、
需要予測性能と解釈性は向上するか？
```

## 実行コマンド

```bash
uv run decoupled-ts retail-experiment --config configs/retail_multigrain_smoke.json
uv run decoupled-ts retail-experiment --config configs/retail_multigrain.json
uv run decoupled-ts retail-experiment --config configs/retail_multigrain_freshretailnet.json
```

## 実験対象

| config | データ | 目的 |
|---|---|---|
| `configs/retail_multigrain_smoke.json` | synthetic small | 実装の動作確認 |
| `configs/retail_multigrain.json` | synthetic retail | 提案構造が効く条件での ablation |
| `configs/retail_multigrain_freshretailnet.json` | FreshRetailNet | 実データでの予測性能比較 |

比較した variant は以下です。

| variant | 内容 |
|---|---|
| `baseline_flatten_mlp` | `D x H` 構造を潰した MLP |
| `global_only` | `z_global` のみ |
| `global_day` | `z_global + z_day` |
| `global_hour` | `z_global + z_hour` |
| `global_day_hour` | `z_global + z_day + z_hour` |
| `global_day_hour_interaction` | `z_global + z_day + z_hour + z_interaction` |

## Smoke 結果

`configs/retail_multigrain_smoke.json` では、2 variant を 1 epoch だけ実行しました。

| model | WAPE | MAE | RMSE | bias |
|---|---:|---:|---:|---:|
| `baseline_flatten_mlp` | 0.9597 | 57.3091 | 62.5408 | -0.9597 |
| `global_day_hour` | 0.7103 | 42.4178 | 49.3815 | -0.7103 |

動作確認としては成功です。`global_day_hour` は baseline より WAPE / MAE が約 26% 改善しました。ただし 1 epoch のため、両モデルとも強い過小予測が残っています。

## Synthetic Retail 結果

`configs/retail_multigrain.json` の結果です。

| model | best valid loss | WAPE | MAE | RMSE | bias | probe acc |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_flatten_mlp` | 123.4978 | 0.1085 | 14.1193 | 20.6287 | -0.0007 | - |
| `global_only` | 122.8209 | 0.1119 | 14.5637 | 22.0946 | -0.0687 | 0.5000 |
| `global_day` | 128.8930 | 0.1082 | 14.0741 | 20.3597 | 0.0179 | 0.4833 |
| `global_hour` | 122.5169 | 0.1114 | 14.4910 | 22.1936 | -0.0497 | 0.4917 |
| `global_day_hour` | 109.3743 | **0.1027** | **13.3631** | 20.5762 | -0.0253 | 0.4917 |
| `global_day_hour_interaction` | 116.1037 | 0.1088 | 14.1635 | 21.2638 | -0.0533 | 0.5333 |

### Synthetic の所見

`global_day_hour` が WAPE / MAE で最良でした。baseline と比べると、WAPE は約 5.4%、MAE も約 5.4% 改善しています。

この結果は、分離構造を持つ synthetic retail data において、`z_day` と `z_hour` を同時に持たせる設計が有効であることを示しています。一方で、`z_interaction` を追加すると性能が悪化しました。現状の interaction は、データ量・正則化・予測ヘッドのいずれかに対して複雑すぎる可能性があります。

## FreshRetailNet 結果

`configs/retail_multigrain_freshretailnet.json` の結果です。

| model | best valid loss | WAPE | MAE | RMSE | bias | probe acc |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_flatten_mlp` | **7.1364** | 0.7173 | 1.7089 | **4.1212** | 0.0254 | - |
| `global_only` | 7.8342 | **0.6150** | **1.4651** | 4.3194 | -0.4063 | 0.0000 |
| `global_day` | 7.7815 | 0.6398 | 1.5243 | 4.2641 | -0.2130 | 0.2600 |
| `global_hour` | 8.4280 | 0.6347 | 1.5122 | 4.4035 | -0.4945 | 0.0000 |
| `global_day_hour` | 7.5820 | 0.6265 | 1.4927 | 4.2444 | -0.2634 | 0.2600 |
| `global_day_hour_interaction` | 7.5592 | 0.6711 | 1.5988 | 4.2452 | -0.1176 | 0.0000 |

### FreshRetailNet の所見

FreshRetailNet では、WAPE / MAE では `global_only` が最良でした。baseline と比較すると、WAPE は約 14.3%、MAE も約 14.3% 改善しています。

`global_day_hour` も baseline より WAPE を約 12.6% 改善しましたが、`global_only` には届きませんでした。したがって、実データでは「多粒度 local 分離が常に最良」とはまだ言えません。

現時点での解釈は以下です。

```text
FreshRetailNet では latent model 群は flatten baseline より有効。
ただし、今回の設定では系列固有の global 表現が支配的で、
day/hour 分離の追加効果は限定的だった。
```

## 学習ログから見えること

FreshRetailNet では、全モデルで train-valid gap が大きく出ています。

| model | epoch12 train forecast loss | epoch12 valid forecast loss |
|---|---:|---:|
| `baseline_flatten_mlp` | 1.6353 | 7.1364 |
| `global_only` | 1.9755 | 7.7243 |
| `global_day_hour` | 1.7270 | 7.4795 |
| `global_day_hour_interaction` | 1.6864 | 7.8287 |

これは単なる学習不足というより、過学習、train/eval 分布差、欠品、または target 作成方法の影響が強い可能性があります。

また、FreshRetailNet では latent model の bias が負に寄っています。特に `global_only` と `global_hour` は強い過小予測です。

| model | bias |
|---|---:|
| `baseline_flatten_mlp` | 0.0254 |
| `global_only` | -0.4063 |
| `global_day` | -0.2130 |
| `global_hour` | -0.4945 |
| `global_day_hour` | -0.2634 |
| `global_day_hour_interaction` | -0.1176 |

このため、WAPE / MAE が改善していても、需要水準を低めに寄せている可能性があります。

## Probe 評価について

`probe_z_global_subgroup_accuracy` は現状では参考値扱いです。FreshRetailNet で `0.0` が出ている variant があり、以下の問題が疑われます。

- train/test 間で subgroup label の overlap が少ない
- subgroup のクラス数や分布が probe に対して厳しい
- majority baseline を併記していない
- `z_global` 以外の latent に情報が漏れている可能性を評価できていない

次回は、probe 評価に以下を追加する必要があります。

- class 数
- train/test label overlap
- majority baseline accuracy
- `z_day`, `z_hour` からの subgroup probe
- 曜日・時間帯・日付特徴を対象にした probe

## まとめ

今回の実験から言えることは次です。

1. Synthetic retail data では、`global_day_hour` が最良であり、day-level local と hour-level local を分ける設計の有効性が確認できました。
2. FreshRetailNet では、latent model 群は flatten baseline より WAPE / MAE を改善しました。
3. FreshRetailNet では `global_only` が最良で、day/hour 分離の追加効果は限定的でした。
4. `z_interaction` は synthetic / FreshRetailNet ともに安定した改善を示していないため、現時点では主実験ではなく追加実験扱いが妥当です。
5. FreshRetailNet では train-valid gap と負の bias が大きく、次の改善対象です。

## 次アクション

優先度順に、以下を実装・確認します。

1. checkpoint 選択基準を `valid_loss` から `valid_forecast_loss` または valid WAPE に変更する。
2. 予測値と正解値を CSV 保存し、需要帯別の過小予測を確認する。
3. `last_day`, `recent_mean`, `same_hour_recent_mean` などの naive baseline を追加する。
4. probe 評価を修正し、majority baseline と label overlap を出す。
5. FreshRetailNet では未来日の既知特徴を `z_day` 予測に使える形へ拡張する。
6. `z_interaction` は正則化・低次元化・十分なデータ量で再評価する。
