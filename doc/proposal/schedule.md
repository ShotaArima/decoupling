# Schedule: 論文化までの残り実験計画

## 現在地

更新日: 2026-06-09

当初の 12 週間計画のうち、実装と探索の多くは前倒しで進んでいる。

現時点で分かっていること:

- synthetic では、成分を持つ残差に対して output decomposition が有効に働く。
- FreshRetailNet では、`same_hour_recent_mean` のような強い baseline の残差は構造が薄い。
- `series_mean` residual では hour 構造が残り、hour component の寄与が安定して大きい。
- 2-Exp-17〜19 で、FreshRetailNet でも `series_mean_all` とカテゴリ集約条件では baseline より低い MAE が出た。
- 2-Exp-20 で、`series_mean_all` の baseline 改善は 5 seed の paired bootstrap でも 0 未満になった。
- 2-Exp-21 で、`series_mean_all` は hour component が強く、`same_hour_recent_mean_d7_all` は hour component が薄いことを確認した。
- 2-Exp-22 で、synthetic では `output_decomp_centered` が true component を高い相関で回復し、centering と interaction component の必要性も確認できた。
- 2-Exp-23 で、synthetic / FreshRetailNet / statistical validation の論文用表を CSV/Markdown として再生成可能にした。
- 2-Exp-24 で、FreshRetailNet の `series_mean` residual は 12000 train 系列まで増やしても改善が保たれ、hour component の対応も安定した。
- 2-Exp-24 では、`same_hour_recent_mean_d7` residual も MAE は少し改善したが、hour component の対応は負相関になり、解釈可能な分解としては不安定だった。
- 2-Exp-25 で、`series_mean` residual の改善は先頭系列ブロックだけでなく、開始位置をずらした 3 つの 6000 系列ブロックでも再現した。
- 2-Exp-25 でも、`same_hour_recent_mean_d7` residual は改善幅が小さく、hour component の対応は負相関であり、強い基準値下では解釈可能な残差構造が残りにくいことを確認した。
- bias 制約つき calibration は bias を抑え、高残差上位 10% の改善を強める一方、全体 MAE は `mae_grid_reference` より悪化する。

したがって、論文の主張は次に寄せるのが現実的である。

```text
提案する残差分解は、強い baseline を置き換える手法ではなく、
baseline 後の残差に残る day/hour 構造を分解し、
条件が合う場合に予測補正と外れケース改善を与える。
```

## あと必要な実験数

最低限の実験は完了。

| ID | 目的 | 必須度 |
| --- | --- | --- |
| `2-Exp-23` | 論文用 final table を 1 つの summary に統合 | 完了 |

今後の展望として、査読耐性と応用可能性を上げるための追加実験を行う。

| ID | 目的 | 必須度 |
| --- | --- | --- |
| `2-Exp-24` | FreshRetailNet の系列数感度確認 | 完了 |
| `2-Exp-25` | FreshRetailNet の系列ブロック頑健性確認 | 完了 |
| `2-Exp-26` | 基準値選択と target 設計の自動化に向けた診断 | 任意 |
| `2-Exp-27` | 実データの interaction 成分が出る条件の探索 | 任意 |
| `2-Exp-28` | 異常検知・運用支援への応用例整理 | 任意 |

つまり、論文 1 本の骨格に必要な実験は揃っている。2-Exp-24 と 2-Exp-25 により、「系列数を増やしても保たれるか」「系列ブロックを変えても保たれるか」は補強できた。次の実験は主張の中心を作るためではなく、限界として残っている「基準値選択に依存する」「実データ interaction が弱い」という点を補強するために行う。

## Revised Week 1: 統計検証と採用モデル決定

期間:

```text
2026-06-07 〜 2026-06-13
```

### 実験

- `2-Exp-20`: 2-Exp-19 の seed-level paired bootstrap

### 完了条件

- `series_mean_all` で baseline 改善の CI が確認できる。
- `bias_constrained_001` を主モデルにするか、`mae_grid_reference` を主モデルにするかを決める。
- 全体 MAE、bias、高残差 top10 の trade-off を表にできる。

状態:

```text
完了。2-Exp-20 で series_mean_all の baseline 改善 CI は 0 未満。
```

## Revised Week 2: 可視化と成功・失敗例

期間:

```text
2026-06-14 〜 2026-06-20
```

### 実験

- `2-Exp-21`: residual heatmap / hour profile / component profile

### 完了条件

- 成功例 3 件、失敗例 3 件を選べる。
- `series_mean_all` で hour component が residual hour profile と対応する図がある。
- `same_hour_recent_mean_d7_all` で残差構造が薄いことを示す図がある。

状態:

```text
完了。visualization CSV からの図生成は scripts/plot_2_exp_21_heatmaps.py で対応済み。
```

## Revised Week 3: Synthetic 最終表

期間:

```text
2026-06-08 〜 2026-06-14
```

### 実験

- `2-Exp-22`: synthetic difficulty final

### 完了条件

- noise / interaction / missing の条件別に、どこで成分分解が成立し、どこで失敗するかを表にできる。
- true component がある synthetic で、同定可能性の主張を支える。

状態:

```text
完了。output_decomp_centered は base で global/day/hour をほぼ完全に回復し、interaction も高く回復した。small_sample と high_noise では成分回復が落ち、失敗条件も表にできた。
```

## Revised Week 4: Final Table 統合

期間:

```text
2026-06-08 〜 2026-06-14
```

### 実験

- `2-Exp-23`: paper table aggregation

### 完了条件

- Synthetic main table
- FreshRetailNet correction table
- Calibration trade-off table
- Limitation table

を 1 つの出力ディレクトリに固定する。

状態:

```text
完了。2-Exp-23 で synthetic_component_recovery、freshretail_correction、statistical_validation を生成した。
```

主な読み取り:

- Synthetic は成分回復と centering の必要性を支える主表として使える。
- FreshRetailNet は `series_mean_all` を主成功例、`same_hour_recent_mean_d7_all` を限界例として使える。
- `statistical_validation` は本文には大きすぎるため、本文用の小表と appendix 用の詳細表に分ける。

## Revised Week 5: 本文表の選定と図表整形

期間:

```text
2026-06-09 〜 2026-06-15
```

### 作業

- `synthetic_component_recovery` から本文用の 6〜9 行を選ぶ。
- `freshretail_correction` から `series_mean_all` と `same_hour_recent_mean_d7_all` の主比較を作る。
- `statistical_validation` から本文用の CI 行だけを抜き出す。
- 2-Exp-21 の heatmap / hour profile 図を本文候補として選ぶ。
- appendix に回す表を決める。

### 完了条件

- 本文 Table 1〜3 の行数と指標が決まっている。
- Figure 1〜3 の候補が決まっている。
- 「本文に載せる結果」と「appendix に回す結果」が分かれている。

## Revised Week 6: 追加検証 1 - 系列数感度

期間:

```text
2026-06-16 〜 2026-06-22
```

### 実験

- `2-Exp-24`: FreshRetailNet scale sensitivity

### 目的

現在の FreshRetailNet 主結果は、最大 6000 train 系列、最大 1500 eval 系列での結果である。

2-Exp-24 では、系列数を 2000 / 6000 / 12000 に変えて、`series_mean` residual の改善が保たれるかを確認する。

### 完了条件

- `series_mean_12k` でも corrected MAE が baseline MAE より低い。
- `series_mean_12k` でも high residual top10 が改善する。
- `series_mean_12k` でも hour component が残差の時間帯 profile と対応する。
- `same_hour_recent_mean_d7_12k` では改善が小さく、強い baseline 下の限界例として扱える。

### 実行コマンド

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-24_freshretailnet_scale_sensitivity_smoke.json
uv run decoupled-ts residual-sweep --config configs/2-Exp-24_freshretailnet_scale_sensitivity.json
```

状態:

```text
完了。series_mean residual は 2k / 6k / 12k すべてで baseline を改善した。
series_mean_12k では baseline MAE 0.0694 に対し、corrected MAE は 0.0487〜0.0494。
hour component residual profile corr も 0.9822〜0.9920 と高く、系列数を増やしても hour 成分の対応は崩れなかった。
same_hour_recent_mean_d7 は MAE では小さく改善したが、hour corr が負になり、解釈可能な分解としては弱い。
```

## Revised Week 6.5: 追加検証 2 - 系列ブロック頑健性

期間:

```text
2026-06-20 〜 2026-06-27
```

### 実験

- `2-Exp-25`: FreshRetailNet block robustness

### 目的

2-Exp-24 は、先頭から取る系列数を増やす実験だった。

2-Exp-25 では、系列の開始位置をずらし、先頭ブロック以外でも `series_mean` residual の改善と hour component の対応が保たれるかを確認する。

### 完了条件

- `series_mean_block1_6k` と `series_mean_block2_6k` でも corrected MAE が baseline MAE より低い。
- high residual top10 でも改善する。
- hour component residual profile corr が正で高い。
- `same_hour_recent_mean_d7` では、MAE 改善と解釈可能性が分離するという限界解釈が保たれる。

### 実行コマンド

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-25_freshretailnet_block_robustness_smoke.json
uv run decoupled-ts residual-sweep --config configs/2-Exp-25_freshretailnet_block_robustness.json
```

状態:

```text
完了。series_mean residual は block0 / block1 / block2 のすべてで baseline を改善した。
MAE 改善幅は 0.0173〜0.0190 程度で、先頭系列だけに依存する結果ではなかった。
hour component residual profile corr も mae_grid_reference で 0.9754〜0.9957 と高く、時間帯成分の対応は保たれた。
same_hour_recent_mean_d7 は各 block で MAE がわずかに改善するが、hour corr は -0.8870〜-0.8085 と負であり、解釈可能な分解としては弱い。
```

## Revised Week 6.75: 追加検証 3 - 基準値選択と interaction 探索

期間:

```text
2026-06-24 〜 2026-07-01
```

### 実験候補

- `2-Exp-26`: residual target / baseline selection diagnostics
- `2-Exp-27`: real-data interaction condition search

### 目的

2-Exp-26 では、どの系列で `series_mean` がよく、どの系列で `same_hour_recent_mean` がよいかを診断する。

2-Exp-27 では、休日、値引き、欠品、天気などの条件と時間帯が組み合わさる場面を集め、実データで interaction 成分が見えるかを確認する。

### 実施判断

2-Exp-25 は良好だったため、2-Exp-26/27 は必須ではない。

それでも次のどちらかを本文で強めたい場合に実施する。

- 本文執筆時に、`series_mean_all` の選択が恣意的に見える。
- 実データで interaction 成分が弱い理由を、より具体的な条件分析として示したい。

該当しなければ、2-Exp-26/27 は appendix または今後課題に回し、執筆を優先する。

## Revised Week 6.9: 応用例整理

期間:

```text
2026-06-28 〜 2026-07-03
```

### 作業

- 高残差改善を、異常検知・外れケース補正として整理する。
- hour component を、時間帯別発注・陳列・人員配置の補助情報として整理する。
- series component を、基準値自体の見直し候補として整理する。
- same-hour baseline で改善が小さいことを、強い基準値の診断例として整理する。

### 完了条件

- Discussion に「応用先」を 1 節追加できる。
- ゼミ資料・論文ともに、残差表現を学ぶ意義が予測精度以外にも説明できる。

## Revised Week 7-8: 執筆

期間:

```text
2026-06-23 〜 2026-07-06
```

### 作業

- Introduction
- Method
- Theory
- Experiments
- Discussion
- Limitations

を書く。

### 完了条件

- 8 ページ相当の初稿がある。
- 主要図表が本文から参照されている。
- FreshRetailNet の主張が過大ではなく、synthetic と real data の役割分担が明確である。

## Revised Week 8: 投稿判断

期間:

```text
2026-07-07 〜 2026-07-13
```

### 判断基準

#### Case A

2-Exp-20 で baseline 改善の CI が明確に 0 未満。

主張:

```text
残差分解は解釈可能性と予測補正の両方に寄与する。
```

現時点の位置:

```text
series_mean_all では Case A に寄せられる。ただし FreshRetailNet 全体で常に改善する主張にはしない。
```

#### Case B

MAE の CI は弱いが、hour component / high residual / synthetic が強い。

主張:

```text
残差分解は、強い baseline 下の残差構造を解釈し、
一部条件で予測補正にも寄与する。
```

#### Case C

FreshRetailNet の統計的改善が弱い。

主張:

```text
残差分解の成立条件を synthetic と実データで整理し、
強い baseline 後の residual learning の限界を示す。
```

現時点では、synthetic は Case A 相当、FreshRetailNet は条件付き Case A / Case B として書くのが妥当である。
