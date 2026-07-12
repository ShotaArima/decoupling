# 2-Exp-12 to 2-Exp-15: Follow-up Experiments After Four-Factor Modeling

## 背景

2-Exp-11 では、真の residual 成分を持つ synthetic で `output_decomp_centered` が強く機能した。

一方で FreshRetailNet subset では、`b + r_hat` が `b` を上回らず、特に補正値の bias が大きく崩れた。

このため、次の検証では次の 2 点を切り分ける。

1. FreshRetailNet のどの subset なら residual 補正が意味を持つか。
2. 実データで悪化している原因が bias なのか、構造不足なのか、モデル設計なのか。

## 追加したコード

### subset filter の拡張

FreshRetailNet subset を切り分けるため、`subset_filter` で使える指標を追加した。

- `residual_abs_mean`
- `residual_hour_eta`
- `residual_weekday_eta`
- `discount_std`
- `residual_structure_score`

`residual_hour_eta` は、残差のうち hour ごとの平均差で説明できる割合を表す。

`residual_weekday_eta` は、残差のうち weekday ごとの平均差で説明できる割合を表す。

`residual_structure_score` は、hour / weekday / discount 由来の構造が強い系列を拾うための簡易スコア。

### sweep runner の拡張

`residual-sweep` に `sweep.scenarios` を追加した。

これにより、同じ seed / model 設定のまま、dataset や loss の条件だけを変えて比較できる。

出力は従来通り次のファイルに保存される。

- `all_results.csv`
- `aggregate.csv`
- `summary.json`

scenario を使った場合は、`aggregate.csv` が `scenario` と `name` ごとに集計される。

### bias control の追加

loss に次の項を追加した。

- `residual_bias_weight`
- `series_residual_bias_weight`

さらに validation set で推定した residual bias を test prediction から引く post-hoc calibration も追加した。

calibrated 指標は `calibrated_*` prefix で出力される。

## 2-Exp-12: FreshRetailNet subset 条件比較

目的は、FreshRetailNet のどの subset で residual 補正が意味を持つかを確認すること。

実行:

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-12_freshretailnet_subset_conditions.json
```

比較する scenario:

| scenario | 意味 |
|---|---|
| `residual_std_top` | 残差分散が大きい系列 |
| `hour_structure_top` | hour ごとの残差構造が強い系列 |
| `weekday_structure_top` | weekday ごとの残差構造が強い系列 |
| `discount_variable_top` | discount の変動がある系列 |
| `combined_structure_top` | hour / weekday / discount をまとめた構造スコア上位 |
| `low_structure_negative_control` | 構造スコア下位の負例 |

見る指標:

- `baseline_cell_mae_mean`
- `corrected_cell_mae_mean`
- `high_residual_top10_corrected_mae_mean`
- `residual_r2_mean`
- `component_ablation_without_day_mae_delta_mean`
- `component_ablation_without_hour_mae_delta_mean`
- `component_ablation_without_interaction_mae_delta_mean`

成功条件:

- `combined_structure_top` や `hour_structure_top` で `corrected_cell_mae_mean < baseline_cell_mae_mean` になる。
- `low_structure_negative_control` では改善しない。
- 構造が強い subset ほど ablation delta が大きくなる。

## 2-Exp-13: bias control

目的は、FreshRetailNet で `corrected_cell_bias` が崩れる問題を抑えること。

実行:

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-13_freshretailnet_bias_control.json
```

比較する model:

| model | 意味 |
|---|---|
| `output_decomp_centered` | 2-Exp-11 の中心モデル |
| `output_decomp_bias_loss` | 全体の residual bias penalty を追加 |
| `output_decomp_series_bias_loss` | 系列単位の residual bias penalty も追加 |
| `output_decomp_validation_calibrated` | validation bias を test 補正から引く |
| `output_decomp_bias_loss_calibrated` | bias loss と validation calibration を併用 |

見る指標:

- `corrected_cell_bias_mean`
- `corrected_cell_mae_mean`
- `calibrated_corrected_cell_bias_mean`
- `calibrated_corrected_cell_mae_mean`
- `high_residual_top10_corrected_mae_mean`
- `calibrated_high_residual_top10_corrected_mae_mean`

成功条件:

- calibrated 指標で bias が縮む。
- bias を抑えても MAE が大きく悪化しない。
- 可能なら high residual top 10% で改善する。

## 2-Exp-14: synthetic difficulty sweep

目的は、どの条件なら成分分離が可能かを synthetic で確認すること。

実行:

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-14_synthetic_difficulty_sweep.json
```

比較する scenario:

| scenario | 意味 |
|---|---|
| `base` | 標準設定 |
| `low_interaction` | interaction が弱い |
| `high_interaction` | interaction が強い |
| `high_noise` | noise が大きい |
| `short_history` | 観測日数が短い |
| `small_sample` | 系列数が少ない |
| `high_stockout` | 欠品が多い |

見る指標:

- `component_global_corr_mean`
- `component_day_corr_mean`
- `component_hour_corr_mean`
- `component_interaction_corr_mean`
- `residual_mae_mean`
- `residual_r2_mean`
- `component_ablation_without_*_mae_delta_mean`

成功条件:

- base では 2-Exp-11 と同様に高い成分相関が出る。
- high_noise / short_history / small_sample で相関が落ちる。
- low_interaction では interaction の ablation delta が小さくなる。
- high_interaction では interaction の ablation delta が大きくなる。

2-Exp-14 では clean component は保持しつつ、学習対象は `noisy_true_residual` にする。これにより、noise を増やしたときに「合計 residual の再構成はできても、clean な成分回収がどの程度崩れるか」を確認できる。

## 2-Exp-15: paper table 用の最終比較

目的は、論文に載せる最終比較表を作ること。

synthetic:

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-15_final_synthetic.json
```

FreshRetailNet:

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-15_final_freshretailnet.json
```

最終的に整理する表:

| dataset | scenario | model | baseline MAE | corrected MAE | calibrated corrected MAE | residual R2 | component recovery |
|---|---|---|---:|---:|---:|---:|---|

論文上の主張は次の形にする。

> 真の residual 成分が存在し、平均ゼロ制約が妥当な場合、4変数への分解モデルは成分を高精度に回収できる。実データでは、残差構造が強い subset と bias 制御が必要であり、全系列で無条件に改善するものではない。

## 実験結果提出時に見る順番

1. 2-Exp-12 の `aggregate.csv` で、どの scenario が改善しているかを見る。
2. 2-Exp-13 の calibrated 指標で、bias が縮んだかを見る。
3. 2-Exp-14 で、synthetic の分離可能条件が仮説通りかを見る。
4. 2-Exp-15 は、2-Exp-12〜14 の結果を踏まえて論文用の表として使う。

## 実行結果

実行日: 2026-06-07

対象コマンド:

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-12_freshretailnet_subset_conditions.json
uv run decoupled-ts residual-sweep --config configs/2-Exp-13_freshretailnet_bias_control.json
uv run decoupled-ts residual-sweep --config configs/2-Exp-14_synthetic_difficulty_sweep.json
uv run decoupled-ts residual-sweep --config configs/2-Exp-15_final_synthetic.json
uv run decoupled-ts residual-sweep --config configs/2-Exp-15_final_freshretailnet.json
```

### 2-Exp-12: FreshRetailNet subset 条件比較

どの subset 条件でも、`b + r_hat` は `b` を安定して上回らなかった。

| scenario | model | baseline MAE | corrected MAE | residual R2 | top10 corrected MAE | hour ablation delta |
|---|---|---:|---:|---:|---:|---:|
| residual_std_top | latent_concat_interaction | 0.0547 | 0.0566 | -0.0057 | 0.2444 | - |
| residual_std_top | output_decomp_centered | 0.0547 | 0.0631 | -0.1275 | 0.2529 | 0.0003 |
| combined_structure_top | latent_concat_interaction | 0.0547 | 0.0554 | -0.0136 | 0.2474 | - |
| combined_structure_top | output_decomp_centered | 0.0547 | 0.0565 | -0.0312 | 0.2485 | 0.0006 |
| hour_structure_top | latent_concat_interaction | 0.0547 | 0.0571 | -0.0325 | 0.2488 | - |
| hour_structure_top | output_decomp_centered | 0.0547 | 0.0569 | -0.0262 | 0.2476 | 0.0001 |
| weekday_structure_top | latent_concat_interaction | 0.0547 | 0.0555 | -0.0097 | 0.2466 | - |
| weekday_structure_top | output_decomp_centered | 0.0547 | 0.0577 | -0.0501 | 0.2493 | 0.0007 |
| discount_variable_top | latent_concat_interaction | 0.0548 | 0.0570 | -0.0326 | 0.2509 | - |
| discount_variable_top | output_decomp_centered | 0.0548 | 0.0563 | -0.0190 | 0.2494 | 0.0001 |
| low_structure_negative_control | latent_concat_interaction | 0.0547 | 0.0559 | -0.0235 | 0.2483 | - |
| low_structure_negative_control | output_decomp_centered | 0.0547 | 0.0573 | -0.0324 | 0.2479 | -0.0000 |

解釈:

- `residual_structure_score` や `hour/weekday/discount` による subset 切り分けでは、改善する領域を見つけられなかった。
- `component_ablation_without_hour_mae_delta` はほぼゼロで、4成分として出した day/hour/interaction が補正にほとんど効いていない。
- `low_structure_negative_control` と構造上位 subset の差も小さい。つまり、今回の subset score は「モデルが使える残差構造」を十分に選別できていない。
- FreshRetailNet の `same_hour_recent_mean` baseline が非常に強く、残差に残る信号がモデルの汎化可能な信号ではなくノイズに近い可能性が高い。

### 2-Exp-13: bias control

validation calibration は bias をほぼ 0 にできたが、MAE は改善しなかった。

| scenario | model | corrected MAE | calibrated MAE | bias | calibrated bias | top10 MAE | calibrated top10 MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| combined_structure_top | output_decomp_centered | 0.0559 | - | -0.1381 | - | 0.2463 | - |
| combined_structure_top | output_decomp_validation_calibrated | 0.0567 | 0.0578 | -0.1350 | 0.0000 | 0.2461 | 0.2441 |
| combined_structure_top | output_decomp_bias_loss_calibrated | 0.0565 | 0.0579 | -0.1298 | 0.0000 | 0.2461 | 0.2442 |
| residual_std_top | output_decomp_centered | 0.0561 | - | -0.2192 | - | 0.2473 | - |
| residual_std_top | output_decomp_validation_calibrated | 0.0560 | 0.0576 | -0.2562 | 0.0000 | 0.2482 | 0.2444 |
| residual_std_top | output_decomp_bias_loss_calibrated | 0.0572 | 0.0582 | -0.2029 | 0.0000 | 0.2470 | 0.2440 |

解釈:

- calibration により `corrected_cell_bias` はほぼ 0 になる。
- ただし `calibrated_corrected_cell_mae` は baseline MAE より悪い。
- high residual top 10% では calibrated top10 MAE が baseline とほぼ同水準まで戻るが、明確な改善ではない。
- したがって、FreshRetailNet での悪化理由は bias だけではない。bias を消しても MAE が改善しないため、`r_hat` の順位・形・符号に十分な予測力がない。

### 2-Exp-14: synthetic difficulty sweep

synthetic では、難易度に対する挙動は仮説と整合した。

| scenario | model | corrected MAE | residual R2 | global corr | day corr | hour corr | interaction corr | interaction ablation delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| base | output_decomp_centered | 0.1076 | 0.9699 | 0.9979 | 0.9962 | 0.9975 | 0.6805 | 0.0084 |
| high_interaction | output_decomp_centered | 0.1181 | 0.9613 | 0.9977 | 0.9946 | 0.9970 | 0.7729 | 0.0288 |
| low_interaction | output_decomp_centered | 0.1018 | 0.9733 | 0.9979 | 0.9962 | 0.9976 | 0.5487 | 0.0009 |
| high_noise | output_decomp_centered | 0.2764 | 0.8351 | 0.9972 | 0.9809 | 0.9936 | 0.6624 | 0.0062 |
| short_history | output_decomp_centered | 0.1066 | 0.9701 | 0.9977 | 0.9965 | 0.9974 | 0.6968 | 0.0085 |
| small_sample | output_decomp_centered | 0.2432 | 0.8389 | 0.9683 | 0.9409 | 0.9701 | 0.1578 | 0.0002 |

解釈:

- `global/day/hour` は多くの条件で高く回収できる。
- `interaction` は条件に敏感で、`high_interaction` では相関と ablation delta が上がり、`low_interaction` と `small_sample` では弱くなる。
- `high_noise` では residual R2 と MAE が悪化するが、clean component correlation は比較的保たれる。
- `small_sample` では interaction の同定が大きく崩れる。これは論文上、「interaction は十分なデータ量と効果量がないと同定しにくい」という重要な制限になる。

### 2-Exp-15: final synthetic

synthetic の最終比較では、`output_decomp_centered` が最も強い。

| model | corrected MAE | residual R2 | global corr | day corr | hour corr | interaction corr |
|---|---:|---:|---:|---:|---:|---:|
| latent_concat_interaction | 0.0312 | 0.9942 | - | - | - | - |
| output_decomp_no_center | 0.0396 | 0.9918 | -0.8734 | 0.9665 | 0.8258 | 0.0473 |
| output_decomp_centered | 0.0141 | 0.9994 | 0.9998 | 0.9995 | 0.9998 | 0.9938 |
| output_decomp_centered_bias_loss | 0.0153 | 0.9992 | 0.9997 | 0.9995 | 0.9998 | 0.9948 |

解釈:

- 4変数への分解 + 平均ゼロ制約は、controlled setting では非常に強い。
- `output_decomp_no_center` は総 residual の再構成はできるが、成分復元は崩れる。特に `global` の符号反転と `interaction` の低相関が大きい。
- これは「予測精度が高いこと」と「成分が意味を持って分離されること」は別であり、平均ゼロ制約が分離のために必要である、という論拠になる。

### 2-Exp-15: final FreshRetailNet

FreshRetailNet の最終比較では、どのモデルも baseline を安定して上回らなかった。

| scenario | model | baseline MAE | corrected MAE | calibrated MAE | R2 | calibrated R2 | top10 MAE | calibrated top10 MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| combined_structure_top | latent_concat_interaction | 0.0558 | 0.0567 | - | -0.0152 | - | 0.2497 | - |
| combined_structure_top | output_decomp_centered | 0.0558 | 0.0564 | - | -0.0115 | - | 0.2496 | - |
| combined_structure_top | output_decomp_validation_calibrated | 0.0558 | 0.0562 | 0.0583 | -0.0166 | -0.0025 | 0.2504 | 0.2469 |
| combined_structure_top | output_decomp_bias_loss_calibrated | 0.0558 | 0.0566 | 0.0580 | -0.0109 | -0.0039 | 0.2493 | 0.2469 |
| low_structure_negative_control | latent_concat_interaction | 0.0558 | 0.0559 | - | -0.0070 | - | 0.2492 | - |
| low_structure_negative_control | output_decomp_centered | 0.0558 | 0.0568 | - | -0.0197 | - | 0.2502 | - |
| low_structure_negative_control | output_decomp_validation_calibrated | 0.0558 | 0.0562 | 0.0578 | -0.0173 | -0.0008 | 0.2505 | 0.2470 |
| low_structure_negative_control | output_decomp_bias_loss_calibrated | 0.0558 | 0.0562 | 0.0576 | -0.0087 | -0.0009 | 0.2494 | 0.2468 |

解釈:

- `combined_structure_top` でも `low_structure_negative_control` でも傾向は大きく変わらない。
- calibrated R2 は 0 に近づくが、MAE は baseline より悪い。
- top10 MAE は calibration 後に baseline と同水準に近づくが、改善とは言えない。
- probe accuracy も低く、`z_day/z_hour` が FreshRetailNet の曜日・時間情報を十分に持っているとは言いにくい。
- 実データでは、今回の `same_hour_recent_mean` baseline が強く、残差は「分離表現を学習する対象」として弱い可能性が高い。

## 総合考察

今回の結果から、論文の主張は次のように収束させるのが妥当。

1. **理論・controlled experiment としては成立している。**  
   synthetic では、真の成分構造がある場合に `output_decomp_centered` が `global/day/hour/interaction` を高精度に回収した。

2. **分離には制約が必要。**  
   `output_decomp_no_center` は予測だけなら良いが、成分は混ざる。平均ゼロ制約がないと、成分の意味は保証されない。

3. **interaction は最も難しい。**  
   effect size が小さい、sample size が小さい、noise が大きい条件で interaction は崩れやすい。これは論文の limitation として明確に書ける。

4. **FreshRetailNet での下流改善は現時点では支持されない。**  
   subset 選別、bias loss、calibration を試しても、baseline を安定して上回らなかった。

5. **FreshRetailNet は「成功例」ではなく「限界・現実データ検証」として扱うべき。**  
   実データで無条件に改善するとは主張せず、強い baseline の残差には学習可能な成分構造が十分残らない場合がある、という負例として扱う方が安全。

## 次に行うべきこと

### 1. FreshRetailNet では target を変える

`same_hour_recent_mean` は強すぎて、残差が学習対象として弱い。

次は以下を試す。

- `series_mean` 残差
- `weekday_same_hour_mean` 残差
- `same_hour_recent_mean` から `recent_days` を短くした残差
- log1p sales 残差
- 店舗商品ではなく、店舗カテゴリや店舗全体に集約した残差

目的は「強すぎる baseline の残りカス」ではなく、まだ day/hour/interaction が残る residual を作ること。

### 2. 実データで oracle ではない成分構造スコアを作り直す

現在の `residual_structure_score` では、改善する subset を選べなかった。

次は系列単位で、train split 内だけから次を推定する。

- residual の hour 固定効果の validation 再現性
- residual の weekday 固定効果の validation 再現性
- discount 有無による residual 差の validation 再現性
- stockout 近傍の residual 差の validation 再現性

単に分散が大きい系列ではなく、train で見えた構造が validation でも再現する系列を選ぶ。

### 3. FreshRetailNet の論文内での位置づけを変更する

FreshRetailNet は最終主張を支える主実験ではなく、次のように置く。

- synthetic: 同定可能性と分離性の主実験
- synthetic difficulty: 成立条件と失敗条件
- FreshRetailNet: 強い baseline 下での外部妥当性・限界検証

これにより、FreshRetailNet で改善しないこと自体を論文の弱点ではなく、仮説の適用条件の議論にできる。

### 4. 次の実験番号

次に実装するなら `2-Exp-16` として、baseline sensitivity を行う。

実験名:

```text
2-Exp-16: Residual target sensitivity on FreshRetailNet
```

比較する residual target:

- `series_mean`
- `weekday_same_hour_mean`
- `same_hour_recent_mean_recent_days_1`
- `same_hour_recent_mean_recent_days_3`
- `same_hour_recent_mean_recent_days_7`
- `log1p_same_hour_recent_mean`

成功条件:

- 少なくとも一部 target で residual probe R2 が上がる。
- `z_day/z_hour` probe が上がる。
- calibrated high residual top10 で baseline を上回る。
- 改善しない場合でも、強い baseline によって残差構造が消えることを示せる。
