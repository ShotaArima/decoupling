# 直接 y 予測と 1 Encoder / 1 Decoder 化の方向性（元論文ベースの検討）

作成日: 2026-08-22
参照論文: Tonekaboni et al., "Decoupling Local and Global Representations of Time Series"
(AISTATS 2022, arXiv:2202.02262)。以下「元論文」。
関連: [unified_model_direction.md](unified_model_direction.md)、
[formulation.md](formulation.md)、
[../2-Exp-40_jima_main_results.md](../2-Exp-40_jima_main_results.md)、
ボード項目 H-EXT-05（共有 backbone）、H-EXT-07（soft centering）、H-EXT-08（軸別 swap）、T-EXT-01/02

## 0. 位置づけ

JIMA（残差 4 成分 + 中心化、現行アーキテクチャ）は 2-Exp-40 で実験的に閉じた。
本文書は次のフェーズ（APIEMS / 修論）に向けて、次の 2 つの方向を元論文の構造に
沿って検討し、考慮すべき点とそれを解決するための仮説を整理する。

1. **方向 A**: 残差 $r=y-b$ ではなく、売上 $y$ を直接扱うモデル
2. **方向 B**: 4 系統の Encoder/Decoder を 1 Encoder / 1 Decoder に統合する実装

## 1. 元論文の構造（本検討で使う部分の要約)

| 要素 | 元論文 | 本研究への示唆 |
|---|---|---|
| Encoder | **2 つ**: Enc_l（非重複窓ごとの local 表現 $z_t$）と Enc_g（系列全体の global 表現 $z_g$） | Encoder は「軸ごとのスコープの違い」であり、系統を分けること自体は本質ではない |
| Decoder | **1 つ**: $p(X \mid Z_l, z_g)$。窓ごとに $[z_g, z_t]$ を受けて元系列を再構成 | **元論文はすでに 1 Decoder**。4 Decoder は本研究側の逸脱であり、統合は「元論文への回帰」として書ける |
| 予測対象 | **元系列 $X$ を直接**、尤度 $p(X\mid\cdot)$ で再構成（残差ではない） | 直接 y も「元論文への回帰」。残差はむしろ本研究側の変形だった |
| 事前分布 | $z_g \sim N(0,1)$、$Z_l \sim GP(0,k)$（時間相関を持つ） | day 軸の latent 列に GP prior（週周期カーネル）、hour 軸に periodic kernel が自然な拡張 |
| 欠損 | 観測セルのみで負の対数尤度を評価 | 欠品 mask の扱いは既に同型（formulation 10 節） |
| 正則化 | counterfactual: $z_g^*$ で生成した $X^*$ を再 encode し、global が入れ替わることを要求 | 軸別版（day-Global / hour-Global の差し替え）= H-EXT-08 |

T-EXT-02 の 2 軸 Global/Local 定式化と合わせると、本研究の位置づけは
「元論文の 1 時間軸 Global/Local を、日軸・時間帯軸の 2 軸に直積拡張したもの」であり、
方向 A・B はどちらも**元論文の設計に立ち返る動き**として一貫して説明できる。

## 2. 方向 A: 直接 y を当てるモデル

### 2.1 鍵になる恒等式: 基準値は g 成分の経験推定である

`series_mean` baseline を使う場合、次の恒等式が成り立つ。

```text
y = b + r,  b = 系列の観測平均
r の 4 成分分解の g 成分 ≈ 「系列平均からの平均的ズレ」（学習で推定）
⇒ 直接 y の 4 成分分解では、g 成分が「系列水準そのもの」を担うだけ
```

つまり two-way ANOVA の見方では、**残差版と直接版の違いは
「grand mean を経験平均 b で固定するか、g 成分として学習するか」だけ**である。
中心化制約は a / c / u の担当範囲を固定するので、水準の受け皿は g に一意に決まる。
残差版は「g の初期値を b に固定した直接版」と読める。

これは 2-Exp-26/27 の結果と矛盾しない。あのとき不安定だったのは
**latent split の直接予測**（水準が day/hour latent に漏れて混ざる）であり、
中心化つき**出力分解**の直接版は一度も試していない
（JIMA-04 の limitation「成分別出力モデルの direct 版比較がない」がまさにこれ）。

### 2.2 元論文から借りる要素

1. **尤度ベースの decoder**: 元論文は $p(X\mid Z_l,z_g)$ の NLL を最小化する。
   直接 y では非負・不等分散（2-Exp-33: scale corr 0.97 = 乗法的）が問題になるが、
   Gaussian-MAE の代わりに対数正規または負の二項の尤度を使えば、
   残差の加法前提（formulation の「第一近似」limitation）を正面から解消できる。
2. **観測セルのみの NLL**: 欠品 mask の扱いは現行と同じでよい。
3. **counterfactual 正則化**: 直接 y では系列水準の情報が大きいため、
   g への情報集中を促す元論文型の正則化が残差版より効きやすい可能性がある。

### 2.3 論文上の意味

直接 y 版が成立すれば、主張は「基準値の後処理補正」から
「基準値を内包する生成モデル」へ格上げできる。b は不要になるのではなく、
**g 成分の閉形式推定（= ラダーの第一段）として枠組み内に残る**。
Exp-38〜40 で確立した「empirical は退化ケース」の物語がそのまま使える。

## 3. 方向 B: 1 Encoder / 1 Decoder 化

### 3.1 Decoder 側: 元論文への回帰 + ANOVA 射影

元論文の Decoder は 1 つで、$[z_g, z_t]$ の concat を受ける。本研究の対応物は:

```text
Dec([z_g, z_day(d), z_hour(h), z_u(d,h)]) → セル (d,h) の値 ỹ(d,h)（grid を 1 枚出力）
  ↓
ANOVA 射影 P_g, P_a, P_c, P_u（T-EXT-02。線形・冪等・和が恒等）
  ↓
中心化済み 4 成分（構造として保証、後処理不要）
```

4 Decoder + 後処理 centering は、1 Decoder + 射影層に置き換えられる
（unified_model_direction.md の Stage B）。射影は T-EXT-01 の一意性定理の
構成的実装なので、「モデルの一部としての中心化」という強い説明になる。

### 3.2 Encoder 側: 共有 backbone + 軸別 pooling ヘッド

元論文の Enc_l / Enc_g は同じ入力 X を異なるスコープで読む。2 軸版では
「スコープ = どの軸で pool するか」なので、1 つの backbone に対する
4 つの読み出しで実現できる。

```text
backbone(x) → H ∈ R^{D×24×hidden}（cell-level 特徴）
  z_g   = pool_{d,h}(H)     （両軸 pool）
  z_day = pool_h(H)_d       （hour 軸 pool、日ごと）
  z_hour= pool_d(H)_h       （day 軸 pool、時間帯ごと）
  z_u   = H_{d,h}           （pool なし）
```

これは H-EXT-05（共有 backbone、非劣性）の具体形であり、
「Global/Local の違いは pooling の違い」という説明は元論文の
Enc_l / Enc_g の関係の自然な一般化になる。

### 3.3 確率的要素（VAE / GP prior）の扱い

元論文は VAE（posterior sampling、GP prior、band 構造の精度行列）だが、
本研究の現行実装は deterministic である。段階として:

- 第 1 段: deterministic のまま A・B を実装（主張は予測補正と分解の一意性）
- 第 2 段: day latent 列に GP prior（`kernels.py` が既存。週周期 + RBF）、
  hour latent に periodic kernel を導入し、未来日外挿を GP conditional で行う
  （元論文の forecasting 経路の 2 軸版）

第 2 段は不確実性定量化が主張に必要になった時点で入れる。最初から入れない。

## 4. 考慮が必要な点と解決のための仮説

| ID | 考慮点 | 根拠 | 解決仮説 | 検証 |
|---|---|---|---|---|
| C1 | **水準の再学習**: 直接 y ではモデルが基準値で説明済みの水準構造を再学習し、成分が混ざる | 2-Exp-26/27（latent split の direct で実証）、formulation 2 節 | **H-A1**: 中心化制約下では水準の受け皿が g に一意に固定されるため、latent split で起きた混合は出力分解では起きない。direct 版 `output_decomp_centered` は residual 版と同等の corrected MAE・成分品質を出す | V41: direct vs residual を同一 y スケールの MAE で比較（JIMA-04 の未実施比較） |
| C2 | **スケール・乗法性**: y は非負・不等分散（scale corr 0.97）。加法 MAE は高売上系列に引きずられる | 2-Exp-33/34 | **H-A2**: 尤度ベース decoder（対数正規 / 負の二項）で直接 y を扱えば、残差の加法前提を回避しつつ log1p の部分的解消（0.95→0.80）を超えられる | V42: MAE / NLL / スケール別 subset MAE で加法・log・尤度型を比較 |
| C3 | **欠品（検閲）**: y 直接では欠品セルの「真の需要」が観測できない問題が残差版より重くなる | README（censoring mask）、元論文の欠損 NLL | **H-A3**: 観測セル限定 NLL（元論文と同型）で学習すれば残差版と同等に扱える。検閲補正（潜在需要推定）は主張に含めず limitation に置く | V41 内で stockout 率別 subset を確認 |
| C4 | **未来日の生成**: residual 版は b の持ち越し + r̂。direct 版は decoder が未来日 y を生成する必要がある | 2-Exp-39、元論文の GP conditional | **H-A4**: day latent を特徴量から構成する現行方式で direct 版でも未来日を生成できる（GP prior は第 2 段）。2-Exp-39 の NN 優位は direct 版でも保たれる | V43: 未来日評価（future_mask）の direct 版 |
| C5 | **共有化による成分品質の低下**: 1 Enc/1 Dec では表現が混ざり、成分の質が落ちる恐れ | 2-Exp-40 の no_center の不安定性 | **H-B1**: 担当範囲は射影が構造として固定するため、encoder/decoder の共有は成分品質を落とさない（H-EXT-05 の非劣性）。むしろパラメータ減で seed 分散が下がる | V44: 現行 4 系統 vs 共有 backbone vs 1Enc+1Dec+射影の ladder（容量を揃える） |
| C6 | **射影と soft constraint の関係**: 1 Dec 化で centering をどこに置くか | H-EXT-07、unified_model_direction Stage B/C | **H-B2**: 射影（hard）を既定にすれば違反ゼロが構造保証され、mask 偏在時のみ観測セル上の soft constraint が優る | V45: hard 射影 vs soft（λ sweep）×欠損偏在（H-EXT-07 と共通） |
| C7 | **正則化の必要性**: 統合後、counterfactual/swap は必要か | 元論文 Eq.4-5、2-Exp-7/26 | **H-B3**: 射影が出力側の担当を固定するため、latent 側の正則化は「成分品質の底上げ」であって必須ではない。軸別 swap は ablation として位置づける（H-EXT-08） | V46: 軸別 swap の on/off ablation |
| C8 | **評価の公平性**: direct と residual は目的変数のスケールが違い、絶対値比較できない | JIMA-04 の注意 | **H-A5**: 最終予測 ŷ を同一の y スケールで評価（cell MAE / WAPE / top10 / bias）すれば公平に比較できる。residual 版の corrected 指標はそのまま direct 版の予測指標と並ぶ | V41 の評価設計に組み込み |
| C9 | **b の実務的役割の喪失**: b は運用の土台・比較対象・診断軸だった | formulation 3 節 | **H-A6**: direct 版でも g_hat（+尤度の位置パラメータ）を b と並べて報告すれば、「b を内包する一般化」として実務の物語を保てる。ladder の第一段（empirical）は不変 | 論文構成で担保（V41 の報告様式） |

## 5. 検証ラダー（実験番号案）

| 番号案 | 内容 | 依存 | 対応仮説 |
|---|---|---|---|
| 2-Exp-41 | **direct 版 output_decomp_centered** vs residual 版（同一 y スケール評価、2-Exp-40 と同一データ条件） | なし（現行コードに direct target を追加） | H-A1, H-A5, H-A3 |
| 2-Exp-42 | 尤度 decoder（log-normal / NegBin）による direct 版 | 41 | H-A2 |
| 2-Exp-43 | direct 版の未来日評価（future_mask） | 41 | H-A4 |
| 2-Exp-44 | アーキテクチャ ladder: 4 系統 → 共有 backbone → 1Enc+1Dec+射影 | なし（41 と並行可） | H-B1 |
| 2-Exp-45 | hard 射影 vs soft constraint × 欠損偏在 | 44 | H-B2（= H-EXT-07） |
| 2-Exp-46 | 軸別 counterfactual/swap ablation | 44 | H-B3（= H-EXT-08） |
| （第 2 段） | GP prior（週周期・periodic kernel）と GP conditional 外挿 | 44 | 元論文回帰の完成形 |

41（direct の成否）と 44（統合の非劣性）が分水嶺であり、この 2 本を先行させる。
両方成立すれば、「元論文の 2 軸拡張としての 1 Encoder / 1 Decoder 生成モデル」を
APIEMS / 修論の主提案として立てられる。片方でも不成立なら、
成立した側のみを拡張として報告し、残差版（JIMA の完成形）を主軸に維持する。

## 6. まとめ

- 直接 y も 1 Enc/1 Dec も、元論文（1 Decoder・直接再構成・尤度・GP prior）への
  **回帰**として位置づけられる。本研究の独自部分は「2 軸直積の Global/Local と
  ANOVA 射影による出力保証」に集約される。
- 残差版は「g を経験平均に固定した特殊ケース」、empirical ANOVA は
  「全成分を経験平均で推定する退化ケース」。ラダーが 3 段の同一枠組みになる。
- 分水嶺は 2-Exp-41（direct の成否）と 2-Exp-44（統合の非劣性）。
