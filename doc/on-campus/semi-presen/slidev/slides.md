---
theme: default
title: 需要予測のための共通要因と系列固有要因の分離表現学習に関する研究
info: 修士2年 中間発表ドラフト（Slidev版・Keynote清書前提の簡易デザイン）
class: text-center
highlighter: shiki
lineNumbers: false
drawings:
  enabled: false
transition: none
mdc: true
---

# 需要予測のための<br>共通要因と系列固有要因の<br>分離表現学習に関する研究

<div class="mt-10 text-lg">

修士2年 中間発表

東京都市大学大学院 ○○研究室（要修正）

有馬 翔太

2026-XX-XX（要修正）

</div>

<!--
タイトルは4月頭の主題提出で固定。
口頭で添える一文:「主題提出時は潜在表現の分離を軸としていましたが、実験の結果、
表現を分けるだけでは補正の解釈が保証されないと分かり、分離を出力レベルまで拡張しました。
今日はその過程を報告します」— 路線変更ではなく、タイトルの問いを遂行した結果として語る。0:15
-->

---

# 研究背景（1）— 小売需要の強い規則性と基準値

- スーパー・コンビニでは、**店舗×商品ごとの需要予測** が発注・補充・廃棄削減の基礎
- 食品・日用品の売上には **強い規則性** がある: 毎週同じパターンを繰り返す<br>
  <span class="text-sm text-gray-500">例: 平日夕方に売上が集中する</span>
- このため、**単純な計算でもかなり当たる**

<div class="mt-4 p-4 border rounded bg-gray-50 dark:bg-gray-800">

**基準値（baseline）** = 実務で既に使える単純な予測値<br>
例: 店舗×商品ごとの直近平均、直近7日の同時間帯平均

</div>

<div class="mt-6 text-center text-lg font-bold">

では、基準値どおりに売れない時はどうするか？ →

</div>

<!--
1枚目は「情景＋規則性＋基準値」に絞る。数式記号はまだ出さない。
「単純な方法で十分当たる」と自分で言い切り、「なら何を研究するのか」という疑問を
聴衆に持たせてから次ページで答える構成。
（口頭補足: 高度なモデルに売上全体を学習させても、基準値で説明済みの規則構造を学び直すだけになりやすい）
「定常」という専門用語は使わず「規則性・繰り返し構造」と言う。
非定常性への質疑には「規則から外れる変動こそ残差成分として読む対象。
レベルシフト等は移動平均型の基準値が追従する」と回答（appendix参照）。0:30
-->

---

# 研究背景（2）— 基準値から外れる「ズレ」こそが知りたい情報

**基準値どおりに売れない「ズレ」の実例と、現場の打ち手**:

- 夕方だけ基準値を上回って売れる → **補充タイミングの見直し**
- 休日の特定時間帯だけ大きく外れる → **販促・人員配置の見直し**

ズレは **店舗×商品 ／ 日 ／ 時間帯** という単位で現れ、現場の対応もこの単位で決まる

<div class="mt-4 p-4 border rounded bg-blue-50 dark:bg-blue-900">

売上をゼロから予測し直すのではなく、<br>
基準値から **どの方向に・どの単位で** 外れるかを学習する

</div>

<div class="mt-4">

**研究目的**: 基準値との残差を、系列（＝店舗×商品）・日・時間帯・日×時間帯という解釈可能な成分に **分離して学習** し、基準値の **補正** とズレの **説明** を同時に行う<br>
<span class="text-sm text-gray-500">最終予測 ＝ 基準値 ＋ ズレの補正（$\hat y = b + \hat r$）</span>

</div>

<!--
前ページの「基準値どおりに売れない時は？」に、ズレの実例→単位→問いの転換→目的の順で答える。
「分離して学習」がタイトルの「分離表現学習」と対応することを口頭で一言添える。
成分名の並びは直前の「ズレの単位」と同じなので、初出の専門用語なしで目的まで到達できる。
このページの目的文が、最後の結論（診断・補正・解釈）とそのまま対応する。0:30
-->

---

# 先行研究（1）— 共通要因と系列固有要因の分離表現学習

多数の系列を同時に学習する際、表現を 2 種類に分けて学習する枠組み（本研究のタイトルの立場）:

<div class="mt-2 p-4 border rounded bg-gray-50 dark:bg-gray-800 text-center">

系列群 → Encoder → [ $z_{\text{global}}$: 系列間で**共通**の要因 ｜ $z_{\text{local}}$: **系列固有**の要因 ] → Decoder → 予測

</div>

**代表的な研究**

- Tonekaboni ら [1]: 時系列表現を global / local に分離（decoupling）して学習
- FHVAE [2] / Disentangled Sequential Autoencoder [3]: 静的要因と動的要因の分離（音声・映像）

**何が嬉しいか**

- 共通構造と個別構造を分けて持てると、解釈・転移・少データ系列への対応がしやすい

<!--
この系譜が研究全体の土台なので丁寧に説明する（2枚構成の1枚目）。
「分離表現学習」というタイトルの語がここで定義される。0:40
-->

---

# 先行研究（2）— 小売需要に持ち込むと2つの課題

<div class="mt-4 space-y-4">

<div class="p-4 border rounded">

**課題1: 分離が保証されるのは潜在空間まで**<br>
$z$ を分けても Decoder 内で情報が混ざるため、<br>
**出力された予測補正がどの要因に由来するかは保証されない**

</div>

<div class="p-4 border rounded">

**課題2: 「系列固有（local）」の中身が粗い**<br>
小売では local の中に **日・時間帯・日×時間帯** という異なる運用単位が混在<br>
→ global / local の二分では、運用で読みたい粒度に届かない

</div>

</div>

<div class="mt-4 text-sm text-gray-500">

補足: 基準値の後に残る変動を分析対象にする考え方自体は統計学で確立している（FWL 定理など、詳細は appendix）

</div>

<!--
2枚構成の2枚目。「潜在の分離」と「出力の対応」のギャップ、
「local の粒度」の2点が、次の3つの差分にそのまま接続する。0:40
-->

---

# 本研究の位置づけ — 先行研究との3つの差分

<div class="mt-6 space-y-4">

<div class="p-3 border rounded">

**1.** 売上 $y$ を直接扱うのではなく、**基準値後の残差** $r=y-b$ を対象にする<br>
<span class="text-sm text-gray-500">（規則構造は基準値に任せる）</span>

</div>

<div class="p-3 border rounded">

**2.** local を1つの潜在変数として扱わず、系列・日・時間帯・日×時間帯という **運用単位** へ分ける

</div>

<div class="p-3 border rounded">

**3.** 潜在表現の分離で止めず、**出力される補正成分そのもの** に平均ゼロ・主効果除去の制約を課す<br>
<span class="text-sm text-gray-500">（潜在空間の分離 → 出力成分の分離へ拡張）</span>

</div>

</div>

<!--
「分離表現学習の枠組みを、出力レベルの分離まで拡張した」という位置づけ。
質疑「既存研究との違いは？」への先回り。0:35
-->

---

# 提案手法（1）— モデル全体像

```mermaid {scale: 0.75}
flowchart LR
    A["入力<br>売上・残差の履歴<br>日/時間帯特徴・マスク"] --> B["Encoder<br>4つの表現を抽出"]
    B --> C["成分別 Decoder<br>ĝ, â, ĉ, û を出力"]
    C --> D["平均ゼロ・主効果除去<br>担当範囲を固定"]
    D --> E["補正量<br>r̂ = ĝ+â+ĉ+û"]
    E --> F["最終予測<br>ŷ = b + r̂"]
```

- 独立に学習した補正モデルの足し合わせではなく、**単一損失で end-to-end に同時学習**

<!--
流れは「入力 → Encoder → 成分別 Decoder → 平均ゼロ・主効果除去 → 補正量の算出 → 最終予測」。
基準値 b と足し合わせて最終予測になるところまで1枚で見せる。0:45
-->

---

# 提案手法（2）— 4成分の意味と分解の非一意性

| 成分 | 意味 | 具体例 |
| --- | --- | --- |
| $g_i$ | 系列固有のズレ | 常に基準値より高く売れる |
| $a_{i,d}$ | 日のズレ | 特定の日だけ売れる |
| $c_{i,h}$ | 時間帯のズレ | 夕方だけ売れる |
| $u_{i,d,h}$ | 日×時間帯のズレ | 休日の夕方だけ売れる |

<div class="mt-6 p-4 border rounded bg-yellow-50 dark:bg-yellow-900">

ただし制約がなければ分け方は **一意ではない**<br>
（$g$ に定数を足し $a$ から引いても和は不変 → どちらの成分の担当か決まらない）

</div>

<!--
「なぜ制約が必要か」を自由度の問題として腹落ちさせる。次ページへの伏線。0:40
-->

---

# 提案手法（3）— Decoder 出力に成分の担当範囲を課す

Decoder が出す制約前の成分を $\tilde g,\tilde a,\tilde c,\tilde u$ とする

<div class="grid grid-cols-2 gap-5 mt-3">
<div class="p-3 border rounded">

**系列成分 $g$**

$$
\hat g_i=\tilde g_i
$$

系列全体に共通な1つのスカラなので、$d,h$ 方向の処理は行わない。他成分の和が0のため、

$$
\hat g_i=\frac1{DH}\sum_{d,h}\hat r_{i,d,h}
$$

</div>
<div class="p-3 border rounded bg-blue-50 dark:bg-blue-900">

**日×時間帯成分 $u$**

$$
\begin{aligned}
\hat u_{i,d,h}
&=\tilde u_{i,d,h}
-\frac1D\sum_{d'}\tilde u_{i,d',h}
-\frac1H\sum_{h'}\tilde u_{i,d,h'}\\
&\quad+\frac1{DH}\sum_{d',h'}\tilde u_{i,d',h'}
\end{aligned}
$$

日方向と時間帯方向の平均を引き、<br>二重に引いた全体平均を戻す

</div>
</div>

<div class="mt-3 text-sm text-gray-500">

$\hat a_{i,d}=\tilde a_{i,d}-D^{-1}\sum_{d'}\tilde a_{i,d'}$、
$\hat c_{i,h}=\tilde c_{i,h}-H^{-1}\sum_{h'}\tilde c_{i,h'}$。<br>
これにより $\sum_d\hat a=0$、$\sum_h\hat c=0$、$\sum_d\hat u=\sum_h\hat u=0$ が成り立つ。

</div>

<!--
観測残差 r の閉形式 ANOVA 分解ではなく、NN の Decoder 出力に実際に適用する式。0:55
-->

---

# 提案手法（4）— 平均ゼロ制約が分け方を1つに固定する

同じ残差を表す2つの分解があると仮定し、各成分の差を $\Delta g,\Delta a,\Delta c,\Delta u$ とする:

$$
0=\Delta g_i+\Delta a_{i,d}+\Delta c_{i,h}+\Delta u_{i,d,h}
$$

<div class="mt-4 grid grid-cols-4 gap-3 text-center">
<div class="p-3 border rounded">

**1. 全体平均**

$\Delta g_i=0$

<span class="text-sm text-gray-500">$a,c,u$ の平均は0</span>

</div>
<div class="p-3 border rounded">

**2. 時間帯平均**

$\Delta a_{i,d}=0$

<span class="text-sm text-gray-500">$c,u$ が消える</span>

</div>
<div class="p-3 border rounded">

**3. 日平均**

$\Delta c_{i,h}=0$

<span class="text-sm text-gray-500">$a,u$ が消える</span>

</div>
<div class="p-3 border rounded bg-blue-50 dark:bg-blue-900">

**4. 残り**

$\Delta u_{i,d,h}=0$

<span class="text-sm text-gray-500">他の差がすべて0</span>

</div>
</div>

<div class="mt-5 p-3 border rounded bg-blue-50 dark:bg-blue-900 text-center">

すべての成分差が0 → **同じ残差に対して別の分け方は存在しない**<br>
これが、二元配置 ANOVA の主効果・交互作用分離と同様に一意に定まる理由

</div>

<!--
「一意に定まる」の行間。2つの分解の差を取り、制約を使った平均操作で各差が0になることを順に示す。0:45
-->

---

# 実験の全体像 — 2つの問いに分けて検証

<div class="mt-6 space-y-6">

<div class="p-4 border rounded">

**問い1: 分解は正しくできるのか？**<br>
→ 実験1: **合成データ**（真の成分 $g,a,c,u$ を知っている唯一の設定）で、<br>
　推定成分が真の成分を回復できるかを確認

</div>

<div class="p-4 border rounded">

**問い2: 実データで効くのか？いつ効くのか？**<br>
→ 実験2: **FreshRetailNet-50K** [4]（生鮮小売の公開データ）で、<br>
　補正の効果と、効く条件・効かない条件を確認

</div>

</div>

<div class="mt-4 text-sm text-gray-500">

データ・指標の詳細は各実験の冒頭で説明

</div>

<!--
実験パートの地図だけを見せるページに整理（詳細設定は各実験スライドへ移動）。0:25
-->

---

# 実験1: residual MAE と $R^2$ は何を測るか

真の残差を $r$、モデルの予測を $\hat r$、評価セル数を $N$ とする

<div class="grid grid-cols-2 gap-5 mt-4">
<div class="p-4 border rounded">

**residual MAE：平均でどれだけ外したか**

$$
\mathrm{MAE}=\frac1N\sum_{i,d,h}|r_{i,d,h}-\hat r_{i,d,h}|
$$

- **0 が理想、小さいほど良い**
- 残差と同じ単位で読める
- 例: MAE 0.10 → 平均絶対誤差が 0.10

</div>
<div class="p-4 border rounded bg-blue-50 dark:bg-blue-900">

**residual $R^2$：残差の変動をどれだけ再現したか**

$$
R^2=1-
\frac{\sum_{i,d,h}(r_{i,d,h}-\hat r_{i,d,h})^2}
{\sum_{i,d,h}(r_{i,d,h}-\bar r)^2}
$$

- **1 が理想**
- 0 = 全セルに残差平均 $\bar r$ を出す場合と同等
- 負 = 残差平均だけを出すより悪い

</div>
</div>

<!--
MAE は「外した量」、R2 は「平均だけの予測を基準に、変動の形をどれだけ取れたか」。役割の違う2指標として説明する。0:45
-->

---

# 実験1: 合成データ — 設定と成分回復の主結果

**設定**: 1500系列 × 35日 × 24時間帯、5 seed<br>
真の成分 $g,a,c,u$ を埋め込み、推定成分と直接比較

**指標**: 成分回復 corr = 真の成分と推定成分の相関（**1 に近いほど正しく取り出せている**）

| 成分 | 回復 corr（base 条件） |
| --- | --- |
| global $g$ | 0.9995 |
| day $a$ | 0.9979 |
| hour $c$ | 0.9991 |
| interaction $u$ | 0.9654 |

<div class="mt-2 grid grid-cols-3 gap-3 text-center">
<div class="p-2 border rounded">residual MAE<br><b>0.0965</b><br><span class="text-sm">noise floor ≈ 0.0957</span></div>
<div class="p-2 border rounded">residual $R^2$<br><b>0.9764</b><br><span class="text-sm">変動の97.64%を再現</span></div>
<div class="p-2 border rounded">high interaction時の $u$ corr<br><b>0.9890</b></div>
</div>

<div class="mt-2 text-sm text-gray-500">

生成時のガウスノイズは $\sigma=0.12$。その理論 MAE は $0.12\sqrt{2/\pi}\approx0.0957$であり、実測 MAE 0.0965 はほぼノイズ限界。

</div>

<!--
corr が「回復精度」であることをこのページで定義する。
表は主要行のみ。全条件は appendix。0:35
-->

---

# 実験1: 時間帯成分はピークの位置と形状を再現した

<img src="/figures/fig2_component_recovery.png" class="mt-4 w-full" />

<div class="grid grid-cols-2 gap-5 mt-3">
<div>

**左4枚：真の $c_h$ と推定 $\hat c_h$**

- peak-hour が異なる4 subgroup（7 / 12 / 18 / 21時）
- 推定値はピーク位置だけでなく、谷・符号・振幅まで追従
- subgroup平均 profile の corr はすべて **1.000**

</div>
<div>

**右1枚：推定 profile の t-SNE**

- 各1系列を $[\hat c_0,\ldots,\hat c_{23}]$ の24次元ベクトルで表現
- t-SNE の入力は **推定時間帯成分のみ**
- subgroup label は埋め込み後の色分けにだけ使用

</div>
</div>

<!--
左は「当たった」だけでなく、時間帯成分として読みたい位相・符号・振幅が合うことを説明。右は subgroup label を t-SNE 入力に使っていないことを明言する。0:50
-->

---

# 実験1: 回復した時間帯成分は「読める」と同時に「見分けられる」

<div class="grid grid-cols-2 gap-6 mt-5">
<div class="p-4 border rounded bg-blue-50 dark:bg-blue-900">

**1. 出力成分としての解釈性**

- ピーク時刻の違いが $\hat c_h$ の横方向のずれとして現れる
- 正負の切り替わりから、「どの時間帯で基準値を上回る／下回るか」を読める
- 5 seed 全体でも hour corr は **0.9991**

</div>
<div class="p-4 border rounded">

**2. 系列タイプの識別可能性**

- t-SNE で同一 subgroup が小さくまとまる<br>
  → 同じタイプ内の $\hat c$ が安定
- 4クラスタが重ならない<br>
  → peak-hour の違いが $\hat c$ に保持される
- $g,a,u$ を使わずに分離<br>
  → 時間帯の違いが $c$ に割り当てられた

</div>
</div>

<div class="mt-5 p-3 border rounded bg-yellow-50 dark:bg-yellow-900">

**ただし**: t-SNE は定性的な可視化であり、クラスタ間の距離や向き自体に意味はない。<br>
また、subgroup を明確に作った合成データでの結果であり、実データで同様の分離を保証するものではない。

</div>

<!--
左図の意味は「profile を読める」、右図の意味は「profile のみで系列タイプを見分けられる」。t-SNE を分類精度の証明のように言わない。0:55
-->

---

# 実験1: 合成データ — 平均ゼロ・主効果除去を外すと何が起きるか

**Ablation**: 同じモデルから平均ゼロ・主効果除去の制約だけを外して比較

| | residual MAE | global corr | interaction corr |
| --- | ---: | ---: | ---: |
| **制約あり** | **0.0965** | **0.9995** | **0.9654** |
| 制約なし | 0.1069 | **-0.8976** | **0.0262** |

<div class="mt-4 p-4 border rounded bg-yellow-50 dark:bg-yellow-900">

**予測が当たっていても、成分の中身はデタラメになり得る**<br>
→ 予測精度と成分の解釈性は別物。平均ゼロ・主効果除去は、**解釈のための構造化**

</div>

<!--
このページの意味: 「平均ゼロ・主効果除去がなぜ必要か」の実験的証拠。
精度だけ見ていると気づけない崩壊を、真の成分を知っている合成データだから示せる。0:30
-->

---

# 実験2: FreshRetailNet — 設定と統制比較（潜在分離 vs 出力分解）

**設定**: 2000系列学習／500系列 validation／500系列 test、5 seed<br>
28日履歴 → 2日先×24時間帯、同一 split・calibration なし

同一の残差を対象にした統制比較:

| モデル | corrected MAE | residual $R^2$ | top10 MAE |
| --- | ---: | ---: | ---: |
| baseline | 0.0721 | — | 0.2923 |
| 潜在 global/local | 0.0614 ± 0.0030 | 0.0627 | 0.2873 ± 0.0084 |
| 潜在 4分割（interactionあり） | 0.0619 ± 0.0008 | -0.0058 | 0.2998 ± 0.0023 |
| 出力分解・制約なし | 0.0635 ± 0.0007 | 0.0325 | 0.2903 ± 0.0073 |
| **出力分解 + 平均ゼロ制約** | **0.0583 ± 0.0011** | **0.2483** | **0.2508 ± 0.0154** |

<div class="mt-2 p-3 border rounded bg-blue-50 dark:bg-blue-900">

潜在 global/local 比: MAE **-0.00311** [95% CI -0.00479, -0.00117]、top10 **-0.0364** [-0.0529, -0.0200]

</div>

<!--
発表の山場。先行研究の枠組み（潜在分離）との直接対決。
「5 seed の paired bootstrap でも改善は統計的に一貫（appendix）」と口頭で1行添える。0:50
-->

---

# 実験2: FreshRetailNet — 補正器は指標によって使い分ける

6000系列学習／1500 validation／1500 test、3 seed、選択と評価を分離:

| series mean 残差 | baseline | 経験的 ANOVA | NN出力分解 |
| --- | ---: | ---: | ---: |
| 平均 MAE | 0.0697 | 0.0544 | **0.0516 ± 0.0013** |
| top10 MAE | 0.2788 | **0.1793** | 0.2082 ± 0.0112 |
| 相対 bias | ≈0 | **≈0** | -0.217 ± 0.054 |

<div class="mt-4 p-4 border rounded bg-blue-50 dark:bg-blue-900">

平均誤差は系列間 pooling する NN、<br>
大きく外したケースと無偏性はマスク付き平均の経験的 ANOVA が優位

</div>

- どの軸に構造が残るかを診断し、補正器と運用指標を選ぶ

<!--
「常に勝つ」とは主張しない。基準値が強い条件で改善が小さいのは失敗ではなく設計。
このページで主張のスタンスを固定し、次の診断ページへつなぐ。0:30
-->

---

# 実験2: FreshRetailNet — hour profile による残差診断

<img src="/figures/fig1_hour_profile_overlay.png" class="mt-2 w-full" />

- (a) 系列平均基準値: 残差に時間帯構造が残り、成分が corr **0.996** で追従<br>
  → **早朝は過大予測・朝夕ピークは過小予測** という運用上の読みがそのまま得られる
- (b) 同時間帯直近平均基準値: 残差 profile の振幅が約1桁小さい → 基準値が時間帯構造を **吸収済み**

<div class="mt-2 p-3 border rounded bg-blue-50 dark:bg-blue-900">

改善が小さい条件は失敗ではなく、**基準値が何を吸収済みかを示す診断情報**

</div>

<!--
「効く・効かない」を診断できることが強み、への転換を図1枚で。
実験パートの締め。0:40
-->

---

# 現時点でのまとめ

**本研究で取り組んだこと**

特定の基準値を差し引いた残差に着目し、<br>
残差を **「系列全体」「日」「時間帯」「日×時間帯」** の4成分に分けて学習する方法を検討した。

<div class="mt-5"></div>

**現時点で確認できたこと**

1. **合成データ**：真の4成分を高い相関で回復できた
2. **成分制約**：平均ゼロ・主効果除去がないと、予測できても成分の意味が崩れる
3. **FreshRetailNet-50K**：残差に時間帯構造が残る条件で、補正と時間帯 profile の解釈を確認した

<div class="mt-5 p-4 border rounded bg-blue-50 dark:bg-blue-900">

**ここまでの結論**<br>
残差に構造が残っている場合、4成分の出力分解により、<br>
**基準値の補正と「どの単位で外れたか」の説明を同時に行える**。<br>
ただし、効果は基準値がどの構造を先に吸収しているかに依存する。

</div>

<!--
「何をしたか → 3つの確認事項 → ここまで言える結論」の順で総括する。0:50
-->

---

# 今後の課題 — 実データで「いつ・なぜ効くか」を深掘りする

**1. 実データでの相互作用成分 $u$ の検証**

- FreshRetailNet-50K では時間帯成分 $c$ は明確だが、$u$ の追加価値はまだ一貫していない
- 「特定の日の特定時間帯だけ外れる」事例を抽出し、$u$ の profile・ablation・再現性を確認する

**2. 販促 flag などの外生特徴量の活用**

- 値引き、activity flag、休日、天候は現在も入力に含むが、各出力成分との対応は未検証
- 販促が「どの日・時間帯のズレ」に対応するかを紐付ける
- 特徴量の有無による成分 ablation と、販促有無での $u$ の変化を比較する

<div class="mt-5 p-3 border rounded bg-blue-50 dark:bg-blue-900">

**目標**: 「販促の日は夕方だけ基準値を上回る」といった、<br>
**外生要因 → 日×時間帯のズレ → 運用上の打ち手** の対応を示す。

</div>

<!--
相互作用成分が弱いという現状と、外生特徴量をどのように検証するかを対応させる。0:50
-->

---

# 今後の課題 — 成分の信頼度と一般化性を扱う

**3. 加法モデルの拡張**

- 現在は $\hat r=\hat g+\hat a+\hat c+\hat u$ と同じ重みで加算
- 各系列・状況の信頼度に応じて、成分を弱める重み $\lambda_k\in[0,1]$ を検討する

<div class="mt-4 p-3 border rounded bg-yellow-50 dark:bg-yellow-900">

**注意**: 全データ共通の固定重みは Decoder 出力のスケールに吸収される。<br>
そのため、単なる係数追加ではなく、**系列や状況に応じた shrinkage** として設計する。

</div>

**4. 他の期間・系列・データセットへの一般化**

- 現在の結果は FreshRetailNet-50K の特定 split を中心とした検証
- 未知の店舗・商品、別期間、別の小売データでも、成分の意味と補正効果が保たれるかを確認する

<!--
重み付けは固定スカラではなく、状況依存の shrinkage として説明する。最後に単一データセットという外的妥当性の限界を明示する。0:50
-->

---

# 今後のスケジュール — 学会発表のフィードバックを修士論文へつなげる

| 時期・機会 | 位置づけ | そこまでにまとめる内容 |
| --- | --- | --- |
| **JIMA 秋季学会（国内）** | 国内での中間報告 | 4成分分解の定式化と、合成・実データの主結果 |
| **APIEMS 2026（国際）** | 英語論文・国際発表 | 基準値感度、潜在分離との比較、適用条件を国際会議論文として固定 |
| **学内発表** | 修論に向けた進捗確認 | 学会で得た指摘と、相互作用・外生特徴量の追加結果を反映 |
| **2026年度末　修士論文発表会** | 最終成果 | 理論、予測性能、解釈性、適用条件、限界を統合 |

<div class="mt-6 p-4 border rounded bg-blue-50 dark:bg-blue-900">

**進め方**: 国内学会で定式化と説明を磨き、国際学会で適用条件を議論し、<br>
そのフィードバックを外生特徴量・重み付けの追加実験と修士論文へ反映する。

</div>

<div class="mt-8 text-center text-lg">

ご清聴ありがとうございました

</div>

<!--
単なる予定表ではなく、各発表機会で何を固め、次に何を持ち越すかを示す。0:50
-->

---
layout: center
class: text-center
---

# Appendix

---

# Appendix: 非定常性の扱い

**想定質疑**: 「規則性がある前提だが、非定常な需要は扱わないのか？」

- 本研究は需要の定常性を仮定していない。仮定するのは「**単純な基準値で説明できる構造が大きい**」という実務上の観察のみ
- レベルシフトやトレンドは、移動平均型の基準値（直近平均・直近7日同時間帯平均）が **追従して吸収** する
- 基準値が追従しきれない **系統的なズレ** こそが残差成分の対象:
  - 日成分 $a_{i,d}$ = 日単位の変動（特定日のイベント等）
  - 相互作用成分 $u_{i,d,h}$ = 日×時間帯の変動
- つまり非定常な変動は「扱わない」のではなく、**基準値の更新と残差成分の分担で扱う**

---

# Appendix: 先行研究の系譜詳細

**潜在分離**

- FHVAE [2]、Disentangled Sequential Autoencoder [3]、Tonekaboni ら [1] の decoupling
- 主対象は潜在空間の static/dynamic・global/local 分離

**残差化・残差診断**

- Frisch–Waugh–Lovell、Robinson の部分線形モデル
- Breusch–Pagan、Ljung–Box、ARCH — 基準モデル後の残差構造を調べる確立した方法論

**global-local 予測・分解型予測**

- Deep Factors、DeepGLO ／ Autoformer、FEDformer
- 分解粒度は trend/season や scale であり、系列・日・時間帯・相互作用という小売運用単位ではない

---

# Appendix: 参考文献

<div class="mt-4 text-[15px] leading-relaxed space-y-4">

**[1]** S. Tonekaboni, C.-L. Li, S. O. Arik, A. Goldenberg, and T. Pfister, “Decoupling Local and Global Representations of Time Series,” *Proc. 25th Int. Conf. Artificial Intelligence and Statistics (AISTATS)*, PMLR, vol. 151, pp. 8700–8714, 2022.

**[2]** W.-N. Hsu, Y. Zhang, and J. Glass, “Unsupervised Learning of Disentangled and Interpretable Representations from Sequential Data,” *Advances in Neural Information Processing Systems*, vol. 30, pp. 1878–1889, 2017.

**[3]** Y. Li and S. Mandt, “Disentangled Sequential Autoencoder,” *Proc. 35th Int. Conf. Machine Learning (ICML)*, PMLR, vol. 80, pp. 5670–5679, 2018.

**[4]** Y. Wang *et al*., “FreshRetailNet-50K: A Stockout-Annotated Censored Demand Dataset for Latent Demand Recovery and Forecasting in Fresh Retail,” *arXiv preprint arXiv:2505.16319*, 2025. doi: [10.48550/arXiv.2505.16319](https://doi.org/10.48550/arXiv.2505.16319).

</div>

---

# Appendix: Synthetic 全条件の成分回復

| 条件 | MAE | $R^2$ | global | day | hour | interaction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 0.0965 | 0.9764 | 0.9995 | 0.9979 | 0.9991 | 0.9654 |
| low interaction | 0.0959 | 0.9765 | 0.9995 | 0.9977 | 0.9991 | 0.8211 |
| high interaction | 0.0969 | 0.9767 | 0.9994 | 0.9982 | 0.9990 | 0.9890 |
| high noise | 0.2700 | 0.8428 | 0.9990 | 0.9795 | 0.9948 | 0.8334 |
| short history | 0.0959 | 0.9764 | 0.9993 | 0.9978 | 0.9985 | 0.9547 |
| small sample | 0.1461 | 0.9446 | 0.9912 | 0.9801 | 0.9902 | 0.4252 |
| high stockout | 0.0983 | 0.9755 | 0.9992 | 0.9975 | 0.9985 | 0.9637 |
| low hour signal | 0.0951 | 0.9519 | 0.9994 | 0.9983 | 0.9895 | 0.9800 |

- 相互作用成分は系列数に敏感 → 実データでは interaction を主張しすぎない根拠

---

# Appendix: 統計的検証（5 seed paired bootstrap）

`series_mean_all` での改善（baseline 比）:

| 指標 | 改善 | 95% CI |
| --- | --- | --- |
| raw corrected cell MAE | −0.0196 | [−0.0201, −0.0189] |
| calibrated corrected cell MAE | −0.0163 | [−0.0167, −0.0157] |
| calibrated high residual top10 MAE | −0.0874 | [−0.0905, −0.0842] |

- 5 seed で一貫 → seed 依存の偶然ではない

---

# Appendix: 5 seed 統制比較の全モデル

| モデル | corrected MAE | WAPE | residual $R^2$ | top10 MAE |
| --- | ---: | ---: | ---: | ---: |
| latent: global/local | 0.0614 ± 0.0030 | 0.9925 | 0.0627 | 0.2873 |
| latent: global/day/hour | 0.0625 ± 0.0009 | 1.0094 | -0.0140 | 0.3005 |
| latent: + interaction | 0.0619 ± 0.0008 | 1.0001 | -0.0058 | 0.2998 |
| output: 制約なし | 0.0635 ± 0.0007 | 1.0268 | 0.0325 | 0.2903 |
| output: 制約あり・interactionなし | 0.0595 ± 0.0032 | 0.9615 | 0.2440 | **0.2500** |
| **output: 制約あり・全成分** | **0.0583 ± 0.0011** | **0.9422** | **0.2483** | 0.2508 |

- 全成分 vs interactionなしの MAE 差は -0.00119、95% CI [-0.00414, 0.00163]
- 実データで interaction を足す一貫した優位は未確認

---

# Appendix: 規模・系列 block を変えても改善は再現

| 系列数 | baseline MAE | corrected MAE | top10 corrected | hour corr |
| --- | ---: | ---: | ---: | ---: |
| 2k | 0.0721 | 0.0598 | 0.2643–0.2688 | 0.943–0.983 |
| 6k | 0.0697 | 0.0510–0.0514 | 0.2126–0.2132 | 0.990–0.994 |
| 12k | 0.0694 | 0.0487–0.0494 | 0.1964–0.1994 | 0.982–0.992 |

<div class="mt-4"></div>

| 6k block | baseline MAE | corrected MAEの改善幅 | calibrated top10（bias制約） |
| --- | ---: | ---: | ---: |
| block 0 | 0.0697 | 0.0186–0.0190 | 0.1988 |
| block 1 | 0.0671 | 0.0173–0.0181 | 0.1796 |
| block 2 | 0.0693 | 0.0175–0.0179 | 0.1788 |

---

# Appendix: log 残差でもモデルの順序は不変

| 残差スケール | モデル | corrected MAE | top10 MAE | hour corr |
| --- | --- | ---: | ---: | ---: |
| 加法 | 経験的 ANOVA | 0.0544 | **0.1793** | 1.0* |
| 加法 | NN出力分解 | **0.0516 ± 0.0013** | 0.2082 | 0.995 |
| log1p | 経験的 ANOVA | 0.0510 | **0.1804** | 1.0* |
| log1p | NN出力分解 | **0.0484 ± 0.0010** | 0.2024 | 0.995 |

- log1p で両モデルとも平均 MAE が約6%改善、優劣関係は不変
- scale dependence corr は 0.953 → 0.802：log1p で部分的に低下
- *経験的 ANOVA の hour corr = 1 は定義上のトートロジー

---

# Appendix: 実験データと設定の出典

- 合成データ全条件: `2-Exp-22_synthetic_difficulty_final`
- 5 seed 潜在分離 vs 出力分解: `2-Exp-32_latent_vs_output_multiseed_freshretailnet`
- bias calibration と bootstrap: `2-Exp-19` / `2-Exp-20`
- 規模・block 頑健性: `2-Exp-24` / `2-Exp-25`
- 経験的 ANOVA・additive/log 比較: `2-Exp-34_fullscale_empirical_and_log`
- hour profile / component profile 図: `2-Exp-21` / `2-Exp-29`

<div class="mt-4 text-sm text-gray-500">

MAE は観測セルのみ。欠品セルは学習・評価から除外。± は seed 間標準偏差。

</div>

---

# Appendix: その他の方法詳細

- 平均ゼロ制約と主効果除去の数式詳細
- 損失関数の全体（再構成 + bias 制約 + 正則化）
- calibration 2方式（`mae_grid_reference` / `bias_constrained_001`）の比較表
- direct target での global/local vs 4分割の WAPE 比較（2-Exp-26）
- FreshRetailNet のデータ仕様・欠品マスクの扱い
