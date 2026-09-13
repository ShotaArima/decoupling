# 統一モデル（1 Encoder + 1 Decoder）への方向性整理

作成日: 2026-08-22
関連: [research_direction_2026-08.md](research_direction_2026-08.md)（特に2-2節）、[formulation.md](formulation.md)、[risks.md](risks.md)、[related_work_and_improvement.md](related_work_and_improvement.md)
続編: [direct_y_and_unified_encoder_decoder.md](direct_y_and_unified_encoder_decoder.md)（2026-08-22。元論文ベースで直接 y 予測と 1Enc/1Dec 化を検討し、考慮点 C1〜C9 と仮説・検証ラダー 2-Exp-41〜46 を整理）

このメモは、現行の「3 Encoder + interaction MLP + 4 Decoder + 中心化制約」構成から、
「1 Encoder + 1 Decoder + 損失関数／構造による分解」へ進化させる方向性を、
既存課題・仮説・検証内容として整理する。

## 1. 現在地の確認

現行の主提案（`output_decomp_centered`）は次の構成である。

```text
入力 x (day×hour grid)
  ↓
Enc_g / Enc_d / Enc_h（3 Encoder）
  ↓
z_interaction = MLP(z_day, z_hour)（1 MLP）
  ↓
Dec_g / Dec_d / Dec_h / Dec_u（4 Decoder heads）
  ↓
hard centering（後処理としての平均ゼロ化）
  ↓
r_hat = g_hat + a_hat + c_hat + u_hat
  ↓
y_hat = b + r_hat
```

実証済みの強み:

- 2-Exp-28: 同じ `series_mean` residual 上で、centering ありの4変数分解が
  latent split 系（`global/local` residual 含む）より corrected MAE・
  high-residual top10・hour profile corr のすべてで良い。
- 2-Exp-35: 推定 hour 成分が実測 residual hour pattern を保持
  （profile corr 0.985、ARI 0.65、中心化違反 < 1e-6）。

一方で、この構成は「4本のネットワークが並列に並び、最後に後処理で辻褄を合わせる」
形に見えるため、**1つの原理から4成分が導かれる大きなモデル**としての主張がしづらい。

## 2. 理想形の定式化: 「2つの時間軸での Global/Local 分離」

4成分分解は ad-hoc な細分化ではなく、元論文の Global/Local 分離を
**2つの時間軸（日軸 d と時間帯軸 h）へ直積的に拡張したもの**として定式化できる。

各軸について「軸方向に不変な成分（Global）」と「軸方向に変動する成分（Local）」を
考えると、2軸の組み合わせで 2×2 = 4 成分が一意に出る。

| day軸 | hour軸 | 成分 | 記号 |
|---|---|---|---|
| Global | Global | 系列成分（両軸に不変） | $\hat g_i$ |
| Local | Global | 日成分（日にのみ依存） | $\hat a_{i,d}$ |
| Global | Local | 時間帯成分（時間帯にのみ依存） | $\hat c_{i,h}$ |
| Local | Local | 日×時間帯成分（両軸に依存） | $\hat u_{i,d,h}$ |

このとき中心化制約は、後付けの補正ではなく
**「各軸の Local 成分はその軸の Global（平均）を含まない」という Global/Local 分離の定義そのもの**になる。
これは day×hour grid 上の two-way ANOVA 分解と一致し、4つの部分空間は直交する。

この定式化により、論文の物語は次の1本になる。

```text
元論文: 1つの時間軸で global / local を分離する。
本研究: 小売需要は日と時間帯という2つの周期的時間軸を持つ。
        Global/Local 分離を2軸へ拡張すると、4成分分解と中心化制約が
        「定義」として同時に導かれる。
```

## 3. アーキテクチャの進化パス

鍵になる事実は、**中心化（ANOVA分解）は線形な冪等射影である**という点である。

grid 出力 $\tilde R_i \in \mathbb{R}^{D \times H}$ に対して、
4つの射影 $P_g, P_a, P_c, P_u$（$P_g + P_a + P_c + P_u = I$、互いに直交）を定義できる。

$$
\hat g_i = P_g \tilde R_i,\quad
\hat a_i = P_a \tilde R_i,\quad
\hat c_i = P_c \tilde R_i,\quad
\hat u_i = P_u \tilde R_i
$$

つまり、**1つの Decoder が grid $\tilde R_i$ を1枚出せば、
射影層を通すだけで中心化済みの4成分が自動的に得られ、和は保存される**。
4本の Decoder と後処理としての centering は、原理的には不要にできる。

段階的な移行案:

| Stage | 構成 | 分解の担保 | 位置づけ |
|---|---|---|---|
| A | 共有バックボーン Encoder + 軸別読み出しヘッド + 現行4 Decoder | hard centering（現行） | research_direction 2-2 の最小版 |
| B | 1 Encoder + 1 Decoder + ANOVA射影層 | 構造（射影層） | 統一モデルの主提案候補 |
| C | 1 Encoder + 1 Decoder + soft constraint（$\mathcal{L}_{comp}$ 等） | 損失関数 | 拡張性の検証・mask対応 |

Stage B は「構造で分解を保証する」、Stage C は「損失で分解を誘導する」。
B は保証が厳密（違反ゼロ）で説明が強い。C は制約を観測セル上で定義でき、
祝日成分など非直交な成分への将来拡張が利く。両方を比較するのが筋が良い。

### 損失設計の論点（Stage C）

- $\mathcal{L}_{rec}$: 観測セル上の残差再構成（現行と同じ）
- $\mathcal{L}_{comp}$: 平均ゼロ・周辺平均ゼロの soft 版（formulation.md 10節）
- 軸別 swap / counterfactual 正則化: 元論文の counterfactual global regularization を
  「day軸 Global の差し替え」「hour軸 Global の差し替え」として軸ごとに定義できる。
  2軸 Global/Local 定式化の自然な帰結であり、元論文との接続を強める。
- bias 制約（$\mathcal{L}_{bias}$, $\mathcal{L}_{series\_bias}$）: 2-Exp-28 で残った
  corrected bias への対応として統一枠組み内に組み込む。

### 見落とされがちな論点: mask 下での中心化

hard centering は全セル平均で中心化するが、評価・学習は観測セル上で行う。
欠測が偏る系列では「理論上の平均ゼロ」と「観測セル上の平均ゼロ」がずれる。
soft constraint なら観測セル上で制約を定義でき、このずれを扱える。
これは Stage C 固有の利点として検証する価値がある。

## 4. 既存課題の棚卸し

| ID | 課題 | 根拠 | 統一モデルとの関係 |
|---|---|---|---|
| K1 | 4系統独立 Enc/Dec でパラメータが多く「1つの原理」として説明しづらい | research_direction 2-2 | **本メモの主対象**。Stage A/B で解消を狙う |
| K2 | 中心化が「後処理」に見え、モデルの原理として弱い | formulation.md 8節 | 射影層（Stage B）か損失（Stage C）としてモデル内部化 |
| K3 | interaction 成分が実データで弱い | risks.md R2、2-Exp-26/28 | 2軸定式化では u は「両軸Local」として定義上必ず存在。寄与の小ささは limitation のまま |
| K4 | residual の有効性が baseline に依存（same-hour系では効かない） | risks.md R1/R4 | 統一モデルでも変わらず。baseline 診断を前処理として維持 |
| K5 | latent の解釈性が弱い（subgroup probe 0.0） | 2-Exp-26 | 保証対象を出力成分に置く方針を維持。統一モデルでも latent 識別は主張しない |
| K6 | corrected bias が残る | 2-Exp-28 | bias 制約を損失に統合（Stage C） |
| K7 | 潜在次元数（10/10/10/8）の根拠が未検証 | risks.md 7節、research_direction 2-3 | 統一モデルなら「バックボーン幅」1つに集約され、感度実験が単純化 |
| K8 | mask 下での中心化の定義が理論と実装でずれ得る | 本メモ3節 | Stage C の soft constraint で対応 |
| K9 | 他ドメイン（電力等）での汎化が未検証 | research_direction 2-4 | 2軸 Global/Local 定式化は「任意の2周期軸」への一般化を主張しやすくする |
| K10 | 直接 $y$ 予測では成分分解が不安定 | 2-Exp-26/27 | residual target を維持。統一モデルの主張は「アーキテクチャの統一」に限定 |

### GitHubプロジェクト（users/ari-show/projects/11）との照合

ボードは13件: 理論1件（T-EXT-01）、拡張仮説6件（H-EXT-01〜06）、JIMA原稿タスク6件（JIMA-01〜06）。
本メモの課題・仮説・検証との対応は次のとおり。

| ボード項目 | 内容 | 本メモとの対応 |
|---|---|---|
| T-EXT-01 | 中心化分解の存在・一意性の定式化 | K2 の理論的裏付け。Stage B の射影層は一意性定理の構成的実装になる |
| H-EXT-01 | 任意のK軸への functional ANOVA 拡張 | 本メモ2節「2軸 Global/Local」の一般化。2×2 の直積定式化は K=2 の特殊ケースとして接続できる |
| H-EXT-02 | 欠損・不均衡下の重み付き中心化 | K8 / V3 と同一問題。ボード側は重み付き内積・制約付き最小二乗まで具体化しており、こちらを正とする |
| H-EXT-03 | 成分のデータ適応的選択（group sparsity） | **本メモに欠けていた観点**。K3（interaction が弱い）への建設的対応で、H-EXT-03 が成立すれば「interaction は選択的に落ちる」と主張できる |
| H-EXT-04 | baseline 非依存の汎用残差補正器 | K4 の対応。V1 の評価に baseline 横断条件を含める |
| H-EXT-05 | 共有 backbone による効率化（非劣性） | K1 / H1 / H2 / V1 と同一。Stage A〜B に相当し、「各出力への中心化射影」の記述は Stage B と一致 |
| H-EXT-06 | 二重周期を持つ他ドメイン（電力等） | K9 / H5 / V7 と同一 |
| JIMA-01 | 中心化あり/なしを主表へ統合 | 2-Exp-32/34 系の結果整理。K2 の実証面 |
| JIMA-02 | 主結果系列の決定と数値統一（Exp-32系 vs Exp-23系の不整合） | **本メモに欠けていた課題**。K11 として下に追加 |
| JIMA-03 | 潜在次元 {2,4,8,16} 感度分析 | K7 / V5 と同一 |
| JIMA-04 | direct vs residual の根拠整理 | K10 と同一。「成分別出力モデルの direct 版比較がない」という limitation は V1 に direct 条件を足すことで将来解消できる |
| JIMA-05 | interaction あり/なしの CI と限界 | K3 の実証面。CI [-0.00414, 0.00163] で優位性未確認 → H-EXT-03 の動機になる |
| JIMA-06 | 一意性定理の本文用圧縮 | T-EXT-01 の JIMA 版 |

追加課題:

| ID | 課題 | 根拠 |
|---|---|---|
| K11 | 主結果の数値が実験系列間で不整合（Exp-32系: baseline 0.0721→0.0583 / Exp-23系: 0.0697→0.0501, calibration後 0.0534）であり、主結果をどちらに置くか未決定 | JIMA-02 |

逆に、本メモにあってボードに未登録の項目は次の3つ。ボードへの追加候補になる。

1. **2軸 Global/Local 定式化の明文化**（本メモ2節）— H-EXT-01 の K軸一般化の手前にある
   「元論文との接続の物語」であり、JIMA の motivation 強化にも使える。
2. **hard centering（射影）vs soft constraint（損失）比較**（V2）— H-EXT-02 の重み付き中心化とも
   接続する設計判断だが、独立した検証項目としてはどこにも登録されていない。
3. **軸別 swap / counterfactual 正則化**（H4 / V6）— 2軸定式化の帰結として元論文の正則化を
   拡張するもの。

## 5. 仮説

| ID | 仮説 | 期待される含意 |
|---|---|---|
| H1 | 中心化は線形射影なので、1 Decoder + 射影層は現行の 4 Decoder + centering と同等の corrected MAE / 分解指標を達成できる | アーキテクチャ統一が性能を犠牲にしないこと |
| H2 | 共有バックボーンにより、パラメータ数・学習時間が減り、seed 間分散（成分の安定性）も下がる | 「軽量で安定した1つの枠組み」という訴求 |
| H3 | hard centering（射影）を soft constraint に置き換えても、十分な λ で分解指標（中心化違反、profile corr、ARI/NMI）は維持され、観測セル上制約により mask 偏在系列で成分推定が改善する | 損失設計による分解の担保が可能であること |
| H4 | 2軸 Global/Local 定式化の下で、軸別 swap / counterfactual 正則化は成分の混入をさらに抑える | 元論文の正則化の自然な拡張としての位置づけ |
| H5 | 統一枠組みは day×hour に限らず任意の2周期軸データ（電力需要の曜日×時刻等）で機能する | 「小売固有のトリック」から「2重周期構造一般の枠組み」への格上げ |

## 6. 検証内容

| ID | 実験 | 比較条件 | 主指標 | 対応仮説 |
|---|---|---|---|---|
| V1 | アーキテクチャ ladder | 現行4系統 / Stage A / Stage B | corrected MAE・WAPE、high-residual top10、bias、パラメータ数、学習時間、seed分散 | H1, H2 |
| V2 | hard vs soft centering | Stage B / Stage C（λ sweep） | 中心化違反量、profile corr、ARI/NMI、corrected MAE | H3 |
| V3 | mask-aware centering | 全セル平均中心化 / 観測セル平均中心化 | 欠測偏在 subset での成分推定誤差・corrected MAE | H3 |
| V4 | synthetic 成分回復 | 統一モデル vs 現行（2-Exp-22/29 と同設定） | true component recovery corr | H1 |
| V5 | 次元感度 | バックボーン幅 {2,4,8,16} | corrected MAE、seed分散、計算量 | K7 解消 |
| V6 | 軸別正則化 ablation | swap なし / day軸のみ / hour軸のみ / 両軸 | 成分間相関、leakage、corrected MAE | H4 |
| V7 | 他ドメイン | 電力需要（曜日×時刻）等 | V1 と同じ指標 | H5 |

実験は synthetic（真の成分あり）→ FreshRetailNet `series_mean`（主成功条件）の順で行い、
same-hour 系 baseline は適用条件の診断（K4）として併記する。

## 7. スケジュール上の位置づけ

research_direction_2026-08.md の優先順位に従う。

- **JIMA（〜08-31）**: 統一モデルは本文に入れない。今後課題として
  「2軸 Global/Local 定式化とアーキテクチャ統一」を1段落で予告する。
  2節の定式化（4成分 = 2軸 Global/Local の直積）自体は追加実験なしで
  motivation の記述強化に使える可能性があるため、原稿次第で導入に反映する。
- **APIEMS（〜09-30）/ 修論**: V1〜V4 を主実験として統一モデルを主提案化。
  V5 は既存インフラで先行実施可（コストパフォーマンス最高）。
  V6〜V7 は修論スコープ。

## 8. リスク

- Stage B の射影層は grid 全体を1つの Decoder で出すため、現行の
  「成分ごとに入力表現が異なる」構造が失われ、成分の質が落ちる可能性がある。
  その場合は Stage A（共有バックボーン + 軸別ヘッド）で止め、
  「読み出しヘッドの分岐は残すが原理は1つ」という主張にする。
- soft constraint（Stage C）は λ 調整が増え、「シンプルにする」という動機と
  逆行し得る。hard 射影（Stage B）を default にし、soft は mask 対応と
  拡張性の議論に限定する。
- 統一しても K4（baseline 依存）は解消しない。統一モデルの主張は
  あくまで「同じ性能をより少ない原理と部品で」に置き、補正性能の向上は
  主張しない（上がれば bonus）。
