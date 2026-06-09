# Experiments: 論文レベルに必要な検証

## 1. 実験全体の目的

本研究の実験目的は、単に予測精度を上げることではない。

目的は以下の 3 つである。

1. 残差を global/day/hour/interaction に分ける設計が、合成データで正しく働くことを示す。
2. FreshRetailNet で、従来の暗黙的な latent 分離よりも、出力分解モデルの方が解釈しやすいことを示す。
3. 基準値 $b$ が強い状況でも、外れケースや構造が強い subset では $b+\hat r$ が役立つ可能性を検証する。

## 1.1 現在の到達状況

2-Exp-23 までで、論文の主要実験はほぼ揃った。

| 項目 | 状態 | 根拠 |
|---|---|---|
| synthetic component recovery | 達成 | 2-Exp-22 で `output_decomp_centered` が true component を高相関で回復 |
| centering の必要性 | 達成 | `output_decomp_no_center` は global / interaction の解釈が崩れる |
| interaction component の必要性 | synthetic では達成 | `high_interaction` で no-interaction が悪化 |
| FreshRetailNet の予測補正 | 条件付きで達成 | `series_mean_all` で baseline より改善 |
| high residual 改善 | 達成 | `series_mean_all` の calibrated high residual top10 が安定改善 |
| FreshRetailNet の hour component 解釈 | 達成 | `series_mean_all` の hour profile corr が約 0.99 |
| same-hour baseline 下での改善 | 限定的 | `same_hour_recent_mean_d7_all` は改善が小さく、一部 calibration で悪化 |
| real data interaction の強い主張 | 不十分 | FreshRetailNet では interaction 成分の主張は synthetic より弱い |
| 論文用表の再生成 | 達成 | 2-Exp-23 で CSV/Markdown を固定 |

このため、今後は追加実験よりも、本文表の絞り込み、図の選定、主張の過不足調整を優先する。

## 2. 比較するモデル

最低限、以下を比較する。

| ID | モデル | 目的 |
|---|---|---|
| B0 | same-hour baseline | 最強の単純基準 |
| B1 | ANOVA direct decomposition | 非学習の直交分解基準 |
| B2 | single local residual AE | 分離なしの学習基準 |
| B3 | latent concat model | これまでのモデル |
| P1 | output decomposition | 出力を $g,a,c,u$ に分ける |
| P2 | output decomposition + centering constraints | 成分制約あり |
| P3 | output decomposition + leakage suppression | 情報漏れ抑制あり |
| P4 | output decomposition + swap | swap も併用 |

論文としては、最終的に P2 または P3 を提案法にするのが自然である。

P4 は補助的に扱う。
理由は、これまでの結果では swap だけで分離を保証するには弱かったためである。

現時点の採用方針:

| 位置づけ | 採用するモデル |
|---|---|
| 主提案 | `output_decomp_centered` |
| FreshRetailNet の主成功例 | `bias_constrained_001` と `mae_grid_reference` |
| 解釈性の負例 | `output_decomp_no_center` |
| interaction の負例 | `output_decomp_centered_no_interaction` |
| 補助・appendix | swap / leakage probe 系 |

## 3. 実験 A: Controlled Synthetic

### 目的

真の成分 $g,a,c,u$ が分かるデータで、提案モデルが成分を回復できるかを見る。

### データ生成

以下の形で残差を生成する。

$$
r_{i,d,h}
=
g_i+a_{i,d}+c_{i,h}+u_{i,d,h}+\varepsilon_{i,d,h}
$$

売上は、

$$
y_{i,d,h}=b_{i,d,h}+r_{i,d,h}
$$

で作る。

### 変化させる条件

| 条件 | 内容 |
|---|---|
| noise level | ノイズを小・中・大で変える |
| missing rate | 欠品・未観測を変える |
| baseline strength | $b$ が強い場合と弱い場合を作る |
| interaction strength | $u$ の強さを変える |
| sample size | 系列数を変える |

### 指標

真の成分が分かるため、成分回復誤差を直接測る。

$$
\mathrm{MAE}_g
=
\mathrm{MAE}(g,\hat g)
$$

$$
\mathrm{MAE}_a
=
\mathrm{MAE}(a,\hat a)
$$

$$
\mathrm{MAE}_c
=
\mathrm{MAE}(c,\hat c)
$$

$$
\mathrm{MAE}_u
=
\mathrm{MAE}(u,\hat u)
$$

また、各成分の相関も見る。

$$
\rho_g=\mathrm{corr}(g,\hat g)
$$

などである。

### 成功条件

- P2/P3 が B2/B3 より成分回復誤差を下げる。
- interaction が強い条件で、$\hat u$ が $u$ を回復する。
- ノイズや欠損が増えても、成分制約ありモデルが比較的安定する。

### 結果

2-Exp-22 で主成功条件は満たした。

- `base` で global/day/hour はほぼ完全に回復し、interaction も高く回復した。
- `high_interaction` で interaction component の必要性が出た。
- `low_interaction` で interaction ablation delta が小さくなった。
- `low_hour_signal` で hour ablation delta が小さくなった。
- `small_sample` と `high_noise` で成分回復が落ち、失敗条件も示せた。

## 4. 実験 B: Ablation on Synthetic

### 目的

各制約が本当に必要かを確認する。

### 比較

| 条件 | 内容 |
|---|---|
| no constraints | 出力分解だけ |
| + day/hour centering | day/hour の平均ゼロ |
| + interaction centering | interaction の主効果除去 |
| + leakage suppression | 不要情報の漏れ抑制 |
| + swap | 反実仮想的な入れ替え制約 |

### 成功条件

- interaction centering を外すと $\hat u$ が day/hour 主効果を吸収して悪化する。
- leakage suppression を入れると、probe leakage が下がる。
- reconstruction が少し悪化しても、component recovery が改善する。

### 結果

2-Exp-22 で、少なくとも centering の必要性は確認済みである。

`output_decomp_no_center` は residual MAE だけでは極端に悪くないが、global corr が負になり、interaction corr が崩れる。

leakage suppression そのものは本文の主軸にはせず、appendix の補助検証に回す。

## 5. 実験 C: FreshRetailNet Full Evaluation

### 目的

実データで、予測補正と成分の意味を確認する。

### 評価対象

- train split
- eval split
- active subset
- high residual subset
- day-structured subset
- hour-structured subset
- interaction-structured subset

### 指標

#### 予測補正

$$
\mathrm{MAE}(y,b)
$$

と

$$
\mathrm{MAE}(y,b+\hat r)
$$

を比較する。

#### 外れケース補正

baseline の誤差が大きい上位 10% で比較する。

$$
\mathrm{MAE}_{top10}(y,b)
$$

$$
\mathrm{MAE}_{top10}(y,b+\hat r)
$$

#### 成分 ablation

例えば day 成分を消す。

$$
\hat r^{(-a)}
=
\hat g+\hat c+\hat u
$$

このとき、

$$
\Delta_a
=
\mathrm{MAE}(r,\hat r^{(-a)})
-
\mathrm{MAE}(r,\hat r)
$$

を測る。

#### leakage

本来持つべきでない情報が latent に漏れていないかを見る。

例:

- $z_{\mathrm{day}}\to subgroup$
- $z_{\mathrm{hour}}\to subgroup$
- $z_{\mathrm{global}}\to discount$

### 成功条件

- 全体平均で改善しなくても、high residual subset で改善する。
- day subset では $\Delta_a$ が大きい。
- hour subset では $\Delta_c$ が大きい。
- interaction subset では $\Delta_u$ が大きい。
- P2/P3 は B3 より leakage が低い。

### 結果

FreshRetailNet では、成功条件は target に依存する。

`series_mean_all`:

- 全体 MAE が改善した。
- high residual top10 が改善した。
- hour component が residual hour profile と強く対応した。

`same_hour_recent_mean_d7_all`:

- 改善は小さい。
- calibration 後に全体 MAE が悪化する条件がある。
- hour component を主張するには弱い。

したがって、FreshRetailNet では `series_mean_all` を主成功例、`same_hour_recent_mean_d7_all` を限界例として扱う。

## 6. 実験 D: Robustness and Statistics

### 目的

結果が seed や subset に依存しすぎていないかを見る。

### 設定

- 3 seeds を最低条件にする。
- 最終結果は 5 seeds が望ましい。
- 各条件で mean/std を出す。
- 主要比較には paired bootstrap を使う。

### 比較

| 比較 | 目的 |
|---|---|
| B3 vs P2 | 出力分解制約の効果 |
| P2 vs P3 | leakage 抑制の効果 |
| B0 vs P2/P3 | baseline 補正として意味があるか |
| synthetic true vs predicted | 成分回復の妥当性 |

## 7. 論文の主要図表

最低限、以下の図表が必要である。

### Table 1: Synthetic component recovery

各モデルの $\mathrm{MAE}_g,\mathrm{MAE}_a,\mathrm{MAE}_c,\mathrm{MAE}_u$ を比較する。

状態: 2-Exp-23 の `synthetic_component_recovery.md` で生成済み。

### Table 2: FreshRetailNet correction

baseline MAE と corrected MAE を比較する。

状態: 2-Exp-23 の `freshretail_correction.md` で生成済み。本文用には `series_mean_all` と `same_hour_recent_mean_d7_all` の 2 scenario を使う。

### Table 3: Factor subset ablation

subset ごとに $\Delta_g,\Delta_a,\Delta_c,\Delta_u$ を出す。

状態: 本文では hour component を中心に扱い、詳細 ablation は appendix 候補。

### Table 4: Leakage probe

不要情報がどれだけ漏れているかを比較する。

状態: 本文の主張からは外し、appendix または limitation に回す。

### Figure 1: Model architecture

従来の latent concat model と提案する output decomposition model を並べる。

### Figure 2: Synthetic true vs predicted components

真の $g,a,c,u$ と予測成分を heatmap で比較する。

状態: 2-Exp-22 の表が主で、図は appendix 候補。

### Figure 3: FreshRetailNet residual heatmap

baseline 残差と補正後残差を subset 別に示す。

状態: 2-Exp-21 と `scripts/plot_2_exp_21_heatmaps.py` で生成可能。

## 8. 論文としての最低ライン

論文として成立させる最低ラインは以下である。

1. Synthetic で P2/P3 が明確に成分回復で勝つ。
2. FreshRetailNet で全体 MAE が勝たなくても、high residual subset で改善する。
3. P2/P3 が B3 より leakage を下げる。
4. ablation が subset の意味と対応する。
5. 3 seeds 以上で傾向が安定する。

このうち 1 と 3 は必須である。
2 は強い主張に必要であり、出なければ論文の主張を「予測補正」ではなく「解釈可能な残差分解」に寄せる。

## 9. 現時点の論文ライン

2-Exp-23 時点では、論文は次のラインで成立させるのが妥当である。

1. Synthetic で、成分が存在する場合の同定可能性と centering の必要性を示す。
2. FreshRetailNet で、残差に hour 構造が残る target では予測補正と high residual 改善が出ることを示す。
3. 強い same-hour baseline 後の残差では改善が小さいことを、提案法の失敗ではなく適用条件として整理する。
4. latent の完全分離ではなく、出力成分が検証可能な制約空間にあることを主張する。
