# Schedule: 論文化までの残り実験計画

## 現在地

更新日: 2026-06-08

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
| `2-Exp-24` | FreshRetailNet の系列数感度確認 | 推奨 |
| `2-Exp-25` | 基準値選択と target 設計の自動化に向けた診断 | 任意 |
| `2-Exp-26` | 実データの interaction 成分が出る条件の探索 | 任意 |
| `2-Exp-27` | 異常検知・運用支援への応用例整理 | 任意 |

つまり、論文 1 本の骨格に必要な実験は揃っている。次の実験は主張の中心を作るためではなく、限界として残っている「全件評価ではない」「基準値選択に依存する」「実データ interaction が弱い」という点を補強するために行う。

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

## Revised Week 6.5: 追加検証 2 - 基準値選択と interaction 探索

期間:

```text
2026-06-20 〜 2026-06-27
```

### 実験候補

- `2-Exp-25`: residual target / baseline selection diagnostics
- `2-Exp-26`: real-data interaction condition search

### 目的

2-Exp-25 では、どの系列で `series_mean` がよく、どの系列で `same_hour_recent_mean` がよいかを診断する。

2-Exp-26 では、休日、値引き、欠品、天気などの条件と時間帯が組み合わさる場面を集め、実データで interaction 成分が見えるかを確認する。

### 実施判断

2-Exp-24 の結果が次のどちらかに該当する場合に実施する。

- 系列数を増やすと `series_mean` の改善が弱くなる。
- 本文執筆時に、`series_mean_all` の選択が恣意的に見える。

該当しなければ、2-Exp-25/26 は appendix または今後課題に回し、執筆を優先する。

## Revised Week 6.75: 応用例整理

期間:

```text
2026-06-24 〜 2026-06-30
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
