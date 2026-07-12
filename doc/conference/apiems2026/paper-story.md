# APIEMS 2026 Paper Story: ロジックとストーリーラインの整理

作成日: 2026-07-12
入力: 26-06-09 / 26-06-30 ゼミ資料、doc/proposal/*、2-Exp-1〜28 実験ドキュメント

このドキュメントは、APIEMS 2026 full paper (英語, 8 pages, 2-column) を書くための
「主張・ロジック・構成・ゼミコメント対応」を一箇所に固定するものである。
執筆中に迷ったら、ここに戻る。

---

## 0. 論文の中心主張(1文)

```text
小売需要予測では、強い単純基準値の後に残る残差にまだ再現可能な構造が残ることがある。
本研究は、その残差を series / day / hour / day-hour interaction の
4 つの「出力成分」に分けて補正するフレームワークを提案し、
(1) 成分が存在する条件では成分を回復できること(synthetic)、
(2) 実データでは残差に hour 構造が残る基準値の下で補正と解釈が同時に成立すること、
(3) 潜在表現を分けるだけではこれが達成できないこと、を示す。
```

ポイントは 3 つ。

1. **条件付きの主張**にする。「常に精度が上がる」とは言わない。
2. 貢献の中心は「精度改善」ではなく「**出力空間での識別可能な分解**」に置く。
3. latent split の失敗(Exp-27)と no-centering の失敗(Exp-28)という
   **2 つの負例が、提案の必然性を支える根拠**である。負例を隠さず、論理の柱として使う。

---

## 1. ストーリーライン(論理の鎖)

論文全体は、次の 7 リンクの鎖として書く。各リンクに根拠実験を対応させる。
1 つでもリンクを飛ばすと「行間が大きい」という前回指摘が再発する。

| # | リンク(主張) | 根拠 | 論文での場所 |
|---|---|---|---|
| L1 | 小売需要では単純な基準値が非常に強い。学習モデルは直接予測でこれを超えにくい | same-hour mean WAPE 0.3405 vs MLP 0.5664 vs 直接予測の提案系 0.6167(Exp-1, 比較実験) | Intro / Exp |
| L2 | 元論文の global/local 分離を直接予測に移植すると、day/hour 分割は小幅改善、interaction は不安定 | Exp-26: WAPE 0.6233 → 0.6133(day/hour)、interaction 入りは 0.6267 に悪化 | Exp(bridge) |
| L3 | したがって問いを「売上はいくらか」から「基準値からどの方向に・どの単位で外れるか」に変える(r = y − b) | 古典的残差分析の系譜(Frisch–Waugh, 残差診断, boosting)+ L1 の実証 | Intro / Formulation |
| L4 | 残差 target は有効。しかし潜在表現を 4 つに分けるだけでは、global/local reference を超えられない | Exp-27: corrected MAE — global/local 0.0593 が最良、4-factor latent は 0.0614〜0.0671。ablation では day/hour latent が使われているのに性能に結びつかない(decoder 内で混合) | Exp(negative) |
| L5 | 残差の**出力**を 4 成分に分け、centering 制約で担当範囲を固定すると、補正と解釈が同時に成立する | Exp-28: corrected MAE 0.0572(vs 0.0609)、high-residual top10 0.2656(vs baseline 0.2923、latent split は悪化)、hour corr 0.992〜0.996。no-center 条件は负例: 成分平均が 0 からずれ、interaction に主効果が混入 | Method / Exp(main) |
| L6 | この分解は、成分が真に存在する条件では正しい成分を回復する。失敗条件も特定できる | Exp-22 synthetic: base で corr ≈ 0.97〜0.999、small_sample で interaction corr 0.43 に崩壊、no-center で解釈崩壊 | Exp(synthetic) |
| L7 | 実データでの効果は基準値に依存する。これは弱点ではなく「基準値診断」という機能である | Exp-16〜19, 23: series_mean では MAE 0.0697→0.0501・top10 0.2788→0.1914・hour corr 0.99。same-hour/weekday-hour では改善小・hour corr 負またはゼロ。Exp-20: 5 seed bootstrap CI で有意。Exp-24/25: 12k 系列・3 block で再現 | Exp(sensitivity) / Discussion |

### 鎖の読み方(執筆時の意識)

- L2・L4 は**わざと失敗を見せる**リンク。「latent を分ければよいという素朴な拡張は
  実際に試して不十分だった。だから出力分解が必要」という**消去法の論証**になる。
  これが「小さい改善の積み重ね」批判への最大の防御である(→ §2.1)。
- L5 が提案の本体。式は少なく、**「同じ補正量を成分間で移し替えられる余白」を
  centering が潰す**という 06-30 資料の数値例(g=12, a=±2, c=∓6 の表)を
  図または小さな表として本文に入れる。これが一番伝わる説明だった。
- L7 は結論を「限界」ではなく「使い方」に反転させる。
  「hour corr が負になる」= 「その基準値はすでに時間帯構造を吸収している」という
  診断情報であり、実務者への deliverable の 1 つとして提示する。

---

## 2. ゼミコメント(26-06-30)への対応

### 2.1 「小さい改善を積み重ねただけの手法なのでは?」

最重要コメント。対応は「数字での反論」と「貢献の立て付け直し」の 2 本立て。

**(a) 貢献を精度改善として書かない。** Contributions を次の 3 つで書く。

```text
C1. Reframing: 強い基準値を置き換えるのではなく、基準値後の残差を
    series/day/hour/interaction に分けて補正・説明する問題として定式化した。
C2. Method: 潜在空間ではなく出力空間で分解し、centering 制約で
    成分の担当範囲を一意に固定する(ANOVA 型の識別条件を NN の出力に課す)。
    latent split(Exp-27)と no-centering(Exp-28)の対照実験で、
    この設計が必要条件であることを示した。
C3. Diagnosis: どの基準値の後にどの残差構造が残るかを測る評価プロトコル
    (hour profile corr, high-residual top10, component ablation)を提示し、
    基準値選択そのものを診断対象にした。
```

「MAE を下げた」は C1〜C3 の**検証結果**として書く。主張にはしない。

**(b) 改善幅は選んで見せる。** 平均 MAE の改善(0.0697→0.0501)だけだと
28% 改善でも「基準値が弱いだけ」と読まれ得る。強い数字は次の 2 つ。

- **high-residual top10: 0.2788 → 0.1914(約 31% 改善、CI [−0.0905, −0.0842])**。
  「基準値が大きく外したケースを補正できる」= 実務価値に直結し、
  latent split 系は同条件で**悪化**しているため、差別化にもなる。
- **hour corr ≈ 0.99**。予測改善ではなく「読める」ことの定量化。
  global/local には対応物が存在しない(n/a)ことも表に残す。

**(c) 負例を貢献に数える。** 「residual にすれば latent split が効くはず」という
自然な仮説を Exp-27 で棄却したこと自体が knowledge contribution である。
Discussion で明示的に「naive extension が効かない理由(decoder 内の情報混合)」を
1 段落使って書く。小さい工夫の羅列ではなく、
**仮説→棄却→設計変更→検証**という一本の研究として読ませる。

### 2.2 「純粋な和の形ではなく、重み付けを変えてみると?」

論文では次のロジックで**和 + centering を defend**し、重み付けは拡張として整理する。

1. **固定スカラー重みは無意味**: r̂ = w_g ĝ + w_a â + ... の定数 w は
   各 head の出力スケールに吸収される(ĝ' = w_g ĝ と置けば同値)。
   学習可能にしても識別性を悪くするだけで表現力は増えない。
2. **文脈依存の重み(gating)は意味が変わる**: 例えば w_h(d)·c_h のように
   日に依存する重みを時間帯成分に掛けると、その積はもはや「時間帯成分」ではなく
   day×hour interaction である。つまり文脈依存重み付けは、
   centering が固定した担当範囲を再び曖昧にし、u 成分と重複する。
   本提案では interaction は u が明示的に担当しており、機能的に包含されている。
3. **和 + centering は「成分を後から検査できる」最小の形**である
   (2 因子 ANOVA 分解と同型で、平均ゼロ制約の下で分解は一意)。
   解釈可能性を目的とする本研究では、これが正しい既定値。

その上で、修論への「プラスアルファ」として次を position する(§5 参照):

- 系列ごと・成分ごとの **shrinkage 重み** λ_c ∈ [0,1](ノイズが強い系列では
  成分を縮める)。これは解釈を壊さずに導入でき、calibration(Exp-18/19)の
  一般化として自然。
- **文脈依存 gating** は「u 成分の構造化」として再解釈して実験する
  (u_{i,d,h} = w(day features)·promo_shape(h) のような低ランク分解)。

発表 Q&A 用の一言回答:

```text
定数重みはスケールに吸収されるので和と等価です。
文脈依存の重みは interaction 成分と役割が重複し、
成分の担当範囲を固定するという本研究の目的と衝突します。
そのため本論文では和 + centering を採用し、
重み付けは interaction の構造化として修論で扱います。
```

### 2.3 「4 つの Encoder/Decoder を 1 つにしたい」(修論方向)

- 本提案の主張は「**出力空間**の分解と centering」にあり、
  encoder が 4 つか 1 つかは主張と**直交**する。したがって国際学会論文は
  現行アーキテクチャのまま出してよい(architecture 図に「shared backbone に
  置き換え可能」と一言添える)。
- 修論での拡張実験として: shared encoder + 4 つの軽量 head に統合し、
  (i) パラメータ数・学習時間、(ii) corrected MAE、(iii) hour corr が
  どこまで保たれるかを見る。「どこまで共有しても成分の意味が保たれるか」は
  representation の科学として意味のある問いになる。

### 2.4 「提案が小さいので、ストーリーラインと考察の押し出し方を工夫」

- ストーリーは §1 の鎖で固定(消去法の論証 + 条件の特定)。
- 考察(Discussion)は「できるようになること」を前面に:
  - 発注・補充: 「この SKU は夕方に基準値より +X 外れやすい」→ 時間帯別の発注補正。
  - 販促分析: 値引き日の 13 時台だけの上振れを u 成分として検出。
  - **基準値診断**: 運用中の予測基準値に対して hour corr を測れば、
    「基準値がどの構造を取りこぼしているか」を定量化できる。
    same-hour baseline で hour corr が負 = 時間帯構造は吸収済み、という読み。
  - 異常検知: 4 成分で説明できない ε が大きいセルの検出。
- APIEMS の文脈(IE/MS)では "interpretable decision support for retail
  operations" として売る(scope-and-fit.md の方針を踏襲)。
  予測モデルの SOTA 競争はしない、と Introduction で明言する。

### 2.5 「国際学会用の論文を先に見せる → 中間発表スライド」

作業順序は §6 参照。paper → スライドの順で、表・図は Exp-23 の
再生成可能な表から引く。

---

## 3. 論文構成案(APIEMS full paper, 8 pages, English)

テンプレート制約: A4 / 2 columns / Times / abstract < 200 words / keywords ≤ 5 /
著者年引用(Gusfield, 1997)方式(doc/conference/apiems2026/submission.md)。

### Title 案

```text
Interpretable Residual Decomposition for Retail Demand Forecasting:
Correcting and Explaining Deviations from Strong Baselines
```

(scope-and-fit.md の案 "Multi-Granularity" は latent 手法を連想させるので、
 "Residual Decomposition + Interpretable + Baseline" を前面に)

### Keywords 案

`retail demand forecasting; residual decomposition; interpretability; baseline correction; decision support`

### 章立てとページ配分

| 章 | ページ | 内容 | 使う素材 |
|---|---:|---|---|
| 1. Introduction | 1.0 | L1→L3 の導入。強い基準値、問いの転換、C1〜C3、条件付き主張の宣言 | Exp-1 の baseline 比較(数字 2〜3 個) |
| 2. Related Work | 0.5 | 4 系統を各 2〜3 文: (i) global/local・static/dynamic 潜在分離(元論文)、(ii) global-local forecasting / trend-season 分解、(iii) 残差分析の古典(Frisch–Waugh, 診断, boosting)、(iv) 小売需要予測と FreshRetailNet | proposal/related_work_and_improvement.md |
| 3. Problem Formulation | 0.75 | y, b, r の定義。4 成分分解と各成分の運用的意味(表 1 つ)。centering 制約と「移し替え余白」の説明(06-30 の数値例を凝縮) | 26-06-30 §2 |
| 4. Proposed Method | 1.0 | Encoder→head→centering→和、損失(L_rec + L_center + L_bias)、calibration は 1 段落。図 1: アーキテクチャ + 出力分解の模式図。「経験的 ANOVA 分解との違い = 特徴量から未来日の成分を予測できる・欠測(欠品)を mask できる・系列間で情報を pool できる」を必ず書く | 26-06-09 §3, Exp-11 |
| 5. Experiments | 2.5 | 下の E1〜E4 | Exp-22/23/26/27/28, 20, 24/25 |
| 6. Discussion | 0.75 | 誰が何に使えるか(§2.4)、なぜ latent split が失敗するか、限界(interaction は実データで弱い、bias が残る、単一データセット)、拡張(weighting=shrinkage/gating、shared encoder、他データセット) | Exp-27/28 の読み、26-06-09 §4.5 |
| 7. Conclusion | 0.25 | 条件付き主張の再掲 | — |
| References | 0.75 | 著者年、アルファベット順 | ref.bib |
| (buffer) | 0.5 | 図表の回り込み調整 | — |

### 実験節の構成(E1〜E4)

| 節 | 問い | 表/図 | 出典 |
|---|---|---|---|
| E1: Synthetic component recovery | 成分が存在すれば回復できるか。いつ失敗するか | 表: scenario × (global/day/hour/interaction corr)。base / no-center / high_noise / small_sample を抜粋 | Exp-22, 23 |
| E2: Latent split vs output decomposition | なぜ出力分解か(**主実験**) | 表: 6 モデル × (corrected MAE, top10 MAE, hour corr, residual R2)。Exp-26 の direct 結果は本文 2〜3 文 + 数字で圧縮 | Exp-26/27/28 |
| E3: Baseline sensitivity | どの基準値の後に構造が残るか | 表: series_mean / same_hour_d7 / weekday_same_hour × (baseline MAE, corrected MAE, top10, hour corr)。図: hour profile overlay(series_mean は重なる、same-hour は崩れる) | Exp-16〜19, 21, 23 |
| E4: Robustness | seed / 規模 / 系列選択に依存しないか | 本文数行 + 小表(bootstrap CI, 12k, block0-2)。詳細は appendix またはスライドへ | Exp-20, 24, 25 |

E2 の中核表(数字は確定済み、Exp-28 FreshRetailNet):

| model | corrected MAE ↓ | top10 corrected ↓ | hour corr ↑ | residual R2 ↑ |
|---|---:|---:|---:|---:|
| series_mean baseline | 0.0721 | 0.2923 | — | — |
| global/local latent (Exp-27 最良の latent 系) | 0.0609 | 0.2928 | n/a | 0.0351 |
| 4-factor latent (day/hour) | 0.0620 | 0.3004 | n/a | −0.0091 |
| 4-factor latent (+interaction) | 0.0614 | 0.3032 | n/a | −0.0241 |
| output decomp, no centering | 0.0617 | 0.2954 | 0.8469 | 0.0173 |
| output decomp + centering (no u) | **0.0572** | 0.2681 | **0.9944** | **0.1984** |
| output decomp + centering (full) | 0.0572 | **0.2656** | 0.9919 | 0.1855 |

この 1 枚で「latent split では top10 が悪化する / no-center では hour corr が
落ちる / centering 付き出力分解だけが全指標で成立する」が読める。
論文の説得力はほぼこの表と E3 の表で決まる。

### Abstract 草案(<200 words)

```text
Retail demand at store-product-hour granularity is often predicted
surprisingly well by simple baselines such as per-series means or
recent same-hour averages. Rather than competing with such baselines,
we ask what structure remains in their residuals and how it can be
corrected and explained. We propose a residual decomposition framework
that predicts the deviation from a given baseline as the sum of four
output components — series, day, hour, and day-hour interaction —
with mean-zero (centering) constraints that fix the role of each
component and make the decomposition identifiable. On synthetic data
with known components, the model recovers them almost perfectly and
we characterize failure conditions (high noise, small samples).
On FreshRetailNet-50K, when residuals retain hourly structure, the
method reduces cell-level MAE by 28% and the error of the 10% largest
baseline failures by 31%, while the learned hour component matches the
true residual hour profile (correlation 0.99). Controlled comparisons
show that decomposing latent representations alone fails to achieve
either gain, and that centering is necessary for interpretability.
The framework also serves as a diagnostic that reveals which structures
a deployed baseline already absorbs, supporting ordering and
replenishment decisions in retail operations.
```

(現状 194 words。数値は最終表に合わせて要再確認。)

---

## 4. 主張と根拠の対応(言えること・言えないこと)

### 言えること(本文で主張する)

| 主張 | 強さ | 根拠 |
|---|---|---|
| synthetic では成分が存在すれば回復できる | 強 | Exp-22 corr ≈ 0.97+ |
| centering がないと解釈が崩れる | 強 | Exp-22 no-center, Exp-28 no-center(interaction への主効果混入) |
| series_mean residual では補正+解釈が成立 | 強 | Exp-19/20/23/24/25(CI, 12k, 3 block) |
| latent split だけでは不十分 | 強 | Exp-27(+Exp-28 の直接比較) |
| same-hour 系 baseline 後は構造が薄い(=診断になる) | 強 | Exp-16〜19, hour corr 負 |
| 実データで hour 成分が読める | 中〜強 | hour corr, ablation delta |

### 言わないこと(査読・質疑で突かれる前に自分で線を引く)

| 言わない主張 | 理由 | 論文での扱い |
|---|---|---|
| 強い基準値を上回る予測器を作った | direct 予測では same-hour mean に届かない(Exp-1) | Intro で明示的に否定 |
| どの基準値でも改善する | same-hour では改善小 | E3 で限界例として提示 |
| interaction 成分が実データで効く | FreshRetailNet では寄与ほぼ 0 | synthetic 検証に限定、実データは hour 中心 |
| latent 表現が識別された | 保証は出力成分のみ | Method で明記 |
| データセット横断の一般性 | FreshRetailNet のみ | Limitation + future work |
| bias まで解決した | centered でも corrected bias 残存(−0.34〜−0.43) | calibration(Exp-18/19)併用と明記 |

---

## 5. 修論への「プラスアルファ」ロードマップ(論文の Future Work と対応)

ゼミコメント「プラスアルファの方向性で修論に持っていきたい」への回答。
国際学会論文の Future Work に 1 文ずつ書き、修論で実験する。

| 方向 | 内容 | ゼミコメントとの対応 | 優先度 |
|---|---|---|---|
| 統合アーキテクチャ | shared encoder + 4 head。共有度合いと成分保持のトレードオフ | 「Encoder/Decoder を 1 つに」 | 高 |
| 成分の重み付け | 系列別 shrinkage 重み(解釈を保つ)/ gating は u の構造化として | 「重み付けを変えてみると」 | 高 |
| 別データセット | M5(日次: hour→曜日に置換して 4 成分の一般化を確認)や他の時間帯別小売データ | 「別のデータセット?」 | 中〜高 |
| interaction が効く条件探索 | 販促頻度が高い subset、イベント期間 | 26-06-09 §5.2 | 中 |
| 全系列交差検証 | FreshRetailNet 全体での検証 | 26-06-09 §4.5 | 中 |

---

## 6. 執筆の作業順序

1. **E2/E3 の表を LaTeX 化**(Exp-23 の md/csv から。数字が主張を決めるので最初)
2. 3 章(Formulation)→ 4 章(Method): 既に 26-06-30 資料の記法で確定している
3. 5 章(Experiments): E1→E4。図 2 点(hour profile overlay, heatmap)は Exp-21 から
4. 1 章(Introduction): 鎖 L1〜L3 + C1〜C3。**最後から 2 番目に書く**
5. 2 章・6 章・7 章・Abstract
6. 全体を通して「直交分解」→「component decomposition with centering constraints」、
   「同定」→「identifiable in the output space」等の語彙統一(paper_direction.md §5)
7. 中間発表スライド: 論文の図表を流用し、§1 の鎖をそのまま 1 スライド 1 リンクで

### 未解決の事務事項(open-questions.md より)

- 締切表記(6/30, 7/31)の意味と final submission の確認 → 事務局照会
- LaTeX テンプレートの有無(現状 Word のみ確認)。現リポジトリの
  thesis.tex は和文テンプレのままなので、**APIEMS 用に英文 2-column
  (Word テンプレ準拠の書式)へ組み替えが必要**

---

## 7. 発表・査読の想定 Q&A

| 想定質問 | 回答の骨子 |
|---|---|
| 観測残差を平均で分解する(ANOVA)だけで良いのでは? NN は要るのか? | 経験的 ANOVA は観測済みセルの事後分解しかできない。提案は特徴量(天気・販促・休日・欠品)から**未来日の成分を予測**し、欠品セルを mask し、系列間で情報を pool する。centering は ANOVA の識別条件を NN 出力に移植したもの |
| same-hour baseline で効かないなら実用性がないのでは? | same-hour が使える(=直近同時刻が観測できる)運用ばかりではない。また hour corr が負になること自体が「その基準値は時間帯構造を吸収済み」という診断情報で、基準値選択の指針になる |
| 重み付き和にしないのか? | §2.2 の回答(スケール吸収 / gating は interaction と重複) |
| 改善が小さいのでは? | 主張は C1〜C3。数字は top10 −31% と hour corr 0.99 を出す。latent split は同条件で top10 悪化 |
| なぜ 2 段階(baseline→residual)で end-to-end にしないのか? | baseline は実務で既に運用されている与件であり、置き換えないことが要件。b を固定するから「b が何を取りこぼすか」という診断が定義できる |
| interaction 成分は要るのか? | 実データでは寄与小と正直に言う。synthetic の high_interaction で必要性を示し、識別枠組みとしての完備性(2 因子の全分解)のために保持 |

---

## 8. 関連ドキュメントへのポインタ

- 主張の変遷と根拠: doc/proposal/paper_direction.md, proposal_rationale.md
- 実験の鎖: doc/2-Exp-26 → 27 → 28(bridge → negative → main)
- 表の再生成: `uv run decoupled-ts paper-tables --config configs/2-Exp-23_paper_tables.json`
- 学会要件: doc/conference/apiems2026/submission.md, open-questions.md
