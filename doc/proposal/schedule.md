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

余裕があれば追加で 2 本。

| ID | 目的 | 必須度 |
| --- | --- | --- |
| `2-Exp-24` | FreshRetailNet subset threshold の軽い感度確認 | 任意 |
| `2-Exp-25` | ablation / leakage probe の appendix 用補強 | 任意 |

つまり、論文 1 本の骨格に必要な実験は揃った。査読耐性を上げるなら、追加で 2 本程度を appendix 用に検討する。

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

## Revised Week 6: 任意の補強

期間:

```text
2026-06-16 〜 2026-06-22
```

### 実験候補

- `2-Exp-24`: FreshRetailNet subset threshold sensitivity
- `2-Exp-25`: leakage / ablation appendix

### 実施判断

次のどちらかに該当する場合だけ実施する。

- 本文を書いていて、`series_mean_all` の選び方が恣意的に見える。
- 表現分離について appendix の補足表がないと説明が弱い。

該当しなければ、追加実験より執筆を優先する。

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
