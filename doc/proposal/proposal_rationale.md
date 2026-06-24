# Proposal Rationale: Related Work から提案へ

## 1. この文書の目的

この文書は、proposal における「提案の根拠」を、RELATED WORK の流れとして書くためのメモである。

単に「実験で良かったからこのモデルにした」と書くのではなく、次の順に整理する。

1. 既存研究が何を明らかにしているか。
2. その既存研究を本研究ではどのように利用するか。
3. 既存研究だけでは小売需要の問題に対して何が不足するか。
4. その不足を埋めるために、なぜ残差出力分解を提案するのか。

本研究の位置づけは、次の一文に集約できる。

```text
既存研究が示してきた local/global 表現分離、global-local forecasting、
残差構造分析、分解型予測の知見を、小売需要における
baseline residual の output-level decomposition として再構成する。
```

## 2. 残差を説明対象にする根拠

### 2.1 残差は単なる誤差ではなく、未説明構造の担い手になり得る

Frisch and Waugh / Lovell の partial regression や Robinson の partialling-out は、ある共変量で説明済みの成分を取り除いた後の残差を使うことで、残った変動に意味を与える。ここで残差は、単なるノイズではなく、基準モデルで取り除いた後に残る対象である。

Breusch-Pagan、Box-Pierce / Ljung-Box、ARCH などの残差診断・誤差構造モデルも、残差に残る異分散、自己相関、動学を明示的に扱う。これらは、「モデルが説明しきれなかった部分に構造があるなら、その構造を調べる」という研究の流れである。

この系譜から本研究が利用する点は、次である。

```text
売上 y を直接説明するだけでなく、
基準値 b が説明しきれなかった残差 r=y-b を
別の説明対象として扱ってよい。
```

ただし、この文献群から同時に得られる注意点も重要である。残差の意味は、第一段階の基準モデルに依存する。したがって、本研究では「残差を見れば常に意味がある」とは主張しない。どの基準値に対する残差なのか、そしてその残差に再現可能な構造が残っているかを、研究対象として明示する。

このため、本研究では次の問題設定を採用する。

```math
r_{i,d,h}=y_{i,d,h}-b_{i,d,h}
```

ここで $b_{i,d,h}$ は、単なる前処理の平均ではなく、既に使える基準値、比較対象、残差構造を診断する軸である。

### 2.2 本研究への接続

小売需要では、直近同時間帯平均や系列平均のような単純な基準値が強い。この状況で売上 $y$ を直接予測すると、モデルは既に説明できる水準や周期性を再学習しやすい。

そこで本研究は、問いを次のように変える。

```text
売上はいくらか。
```

ではなく、

```text
既に使える基準値から、どの方向に、どの要因で外れるか。
```

を問う。

この方向性は、残差化や残差構造分析の文献から導いたものである。ただし、因果推論の control function や DML のように残差を因果識別に使うわけではない。本研究での残差は、基準値で説明できなかった需要変動を、予測補正と解釈の対象にするためのものである。

## 3. local/global 表現分離から得た方向性

### 3.1 Decoupling Local and Global Representations of Time Series

Tonekaboni らの *Decoupling Local and Global Representations of Time Series* は、時系列表現を global representation と local representation に分ける。global はサンプル全体に比較的安定した情報、local は時間窓ごとに変化する情報である。

この論文から本研究が利用する点は、次である。

| 元論文の考え | 本研究での利用 |
|---|---|
| 時系列には時間不変な global 情報と、時間変動する local 情報が混在する | 小売需要でも、系列固有の水準と日・時間帯の変動を分けて考える |
| global/local の情報混入を抑える必要がある | 小売需要でも、成分の担当範囲が混ざる問題を重視する |
| シミュレーションで因子回復を評価する | 本研究でも synthetic で真の成分回復を評価する |

一方、この論文をそのまま小売需要に適用するには不足がある。元論文の local は時間窓ごとの local representation であり、小売運用で知りたい「日」「時間帯」「日×時間帯」を明示的に分けるものではない。また、分離の主対象は latent representation であり、最終的な補正量がどの成分に由来するかまでは直接読めない。

したがって、本研究では元論文の local/global 分離を出発点にしつつ、次の方向へ移す。

```text
latent local/global separation
  から
residual output decomposition into series/day/hour/interaction
  へ
```

### 3.2 FHVAE, DSVAE, C-DSVAE

FHVAE、DSVAE、C-DSVAE は、系列データに含まれる static / dynamic、content / dynamics のような要因を潜在変数として分ける研究である。これらは、時系列には時間を通じて保たれる要因と、時間とともに変化する要因が混在することを示している。

本研究では、これらの文献を次のように活用する。

| 文献群 | 本研究での役割 |
|---|---|
| FHVAE | 時間を通じて安定な要因と時変要因を分ける発想の根拠 |
| DSVAE | static / dynamic posterior factorization の系譜として参照 |
| C-DSVAE | latent 間の情報混入を抑える問題意識の根拠 |

ただし、これらの研究は主に latent space の分離を扱う。本研究が必要とするのは、潜在表現に名前を付けることではなく、予測された残差補正量がどの運用単位に対応するかを読めるようにすることである。

そのため、本研究では latent disentanglement の考えを受け取りつつ、主張の対象を latent から output component へ移す。

## 4. global-local forecasting と分解型予測から得た方向性

### 4.1 Deep Factors と DeepGLO

Deep Factors や DeepGLO は、多数の関連時系列に対して、global な共通構造と local な系列固有補正を組み合わせる forecasting の流れである。

これらの文献から本研究が利用する点は、次である。

```text
全体に共通する構造と、系列固有・局所的な補正を分けることは、
forecasting において自然であり有効である。
```

小売需要では、基準値 $b$ が既に大きな共通構造や水準を吸収している場合がある。その上で残る $r=y-b$ は、基準値で説明できなかった local correction として扱える。

ただし、Deep Factors や DeepGLO の local は、主に系列固有の補正や local model として扱われる。本研究が知りたいのは、残差補正が「系列」「日」「時間帯」「日×時間帯」のどこに由来するかである。したがって、global-local forecasting の発想を、より小売運用に近い粒度へ分ける必要がある。

### 4.2 CoST, Autoformer, FEDformer, TimeMixer

CoST、Autoformer、FEDformer、TimeMixer は、時系列予測において trend / season や multi-scale decomposition を使う研究である。これらは、時系列を一枚岩として扱うのではなく、構造を分けて扱うことが予測に有効であることを示している。

本研究では、これらの文献を次の根拠として使う。

| 文献群 | 本研究で受け取る点 |
|---|---|
| CoST | trend / season のように意味の異なる表現を分けることが forecasting に効く |
| Autoformer / FEDformer | decomposition をモデル内部に組み込むことが予測モデルとして自然である |
| TimeMixer | microscopic / macroscopic な multi-scale 情報を分けて統合する考え方 |

一方、これらの分解は trend / season や scale といった比較的 coarse な単位である。本研究の対象である小売需要では、より具体的に、系列、日、時間帯、日×時間帯という運用上の単位で残差を読みたい。

したがって、本研究では decomposition-based forecasting の考えを受け取りつつ、分解対象を raw time series ではなく、強い基準値の後に残る residual correction に置く。

## 5. 自己教師あり表現学習と局所パターン研究の使い方

TNC、TS2Vec、TS-TCC、TF-C などの自己教師あり時系列表現学習は、時系列から有用な representation を得ることが分類、予測、異常検知などに効くことを示している。また、Shapelets、ROCKET、InceptionTime、INSHAPE などは、局所パターンや多スケール局所特徴が時系列タスクで重要であることを示している。

本研究では、これらを主提案の直接の土台にはしない。理由は、本研究の中心が pretraining や分類性能ではなく、基準値からの残差補正を運用単位で説明することだからである。

ただし、次の 2 点で活用する。

1. 表現の良さは、単一の予測 MAE だけでなく、複数の下流評価で見るべきである。
2. high residual case や局所的なズレは、平均的な予測誤差とは別に評価する価値がある。

このため、本研究では全体 MAE だけでなく、high residual top10、hour profile correlation、component ablation を評価に入れる。

## 6. 以上の文献から導いた本研究の設計

上記の文献を踏まえると、本研究の設計は次のように導かれる。

| 文献・研究群 | 受け取った知見 | 本研究での設計 |
|---|---|---|
| Frisch-Waugh-Lovell, Robinson, residual diagnostics | 残差は基準モデルが説明しきれなかった構造として扱える | $r=y-b$ を研究対象にする |
| Decoupling Local and Global Representations | 時系列には global と local の要因が混在し、分離が必要 | 小売需要でも系列成分と局所成分を分ける |
| FHVAE, DSVAE, C-DSVAE | static/dynamic の潜在分離と情報混入抑制が重要 | 成分混入の問題意識を導入する |
| Deep Factors, DeepGLO | global な共通構造と local 補正の組み合わせが forecasting に有効 | 基準値後の residual correction として考える |
| CoST, Autoformer, FEDformer, TimeMixer | 分解型予測は有効だが、多くは trend/season や scale 単位 | 小売運用単位である series/day/hour/interaction へ細分化する |
| TNC, TS2Vec, TS-TCC, Shapelets, anomaly detection 系 | 局所パターンや high residual case の評価が重要 | high residual top10、hour profile、component ablation を評価する |

ここから、本研究の提案は次のようになる。

```math
\hat r_{i,d,h}
=
\hat g_i
+
\hat a_{i,d}
+
\hat c_{i,h}
+
\hat u_{i,d,h}
```

ここで、出力成分を単に並べるだけでは、成分間で主効果が混ざる可能性がある。そこで、平均ゼロ制約と interaction の主効果除去を入れる。

この制約の目的は、latent が真の因子と一対一対応することを保証することではない。目的は、出力された補正量が series / day / hour / interaction という担当範囲を持つようにし、profile comparison や ablation で検査できるようにすることである。

## 7. 本研究の新規性の書き方

RELATED WORK の流れから見ると、本研究の新規性は次の 3 点である。

### 7.1 残差を主対象にする

既存の decomposition forecasting の多くは、raw time series やその潜在表現を対象にする。本研究は、強い小売 baseline の後に残る residual を対象にする。

これは、既存の基準値を置き換えるのではなく、基準値が外した部分を診断し補正する立場である。

### 7.2 分解粒度を小売運用単位に合わせる

既存研究の local/global、static/dynamic、trend/season、multi-scale decomposition は重要だが、粒度が粗い。本研究では、残差補正量を series / day / hour / interaction に分ける。

これにより、補正量が「どの店舗・商品系列のズレか」「どの日のズレか」「どの時間帯のズレか」「特定の日と時間帯の組み合わせのズレか」として読める。

### 7.3 latent ではなく output を制約する

既存の disentanglement 研究は latent representation の分離を重視する。しかし、latent を分けても decoder 内で情報が混ざると、出力成分の意味は保証されない。

本研究では、出力補正量そのものを分解し、平均ゼロ制約によって担当範囲を固定する。したがって、評価対象は latent の名前ではなく、出力成分の profile、ablation、成分回復である。

## 8. Proposal 本文向けの文章案

以下は、proposal の「提案の根拠」または RELATED WORK 末尾に入れる文章案である。

> 時系列表現学習では、系列全体を通じて安定な global 要因と、時点や区間ごとに変化する local 要因を分ける研究が進んできた。FHVAE、DSVAE、C-DSVAE は static / dynamic な潜在要因の分離を扱い、Tonekaboni らの Decoupling Local and Global Representations of Time Series は、global representation と local representation の情報混入を抑える枠組みを提案した。これらの研究は、時系列には異なる時間スケールの要因が混在し、それらを分けることが有効であることを示している。
>
> 一方で、これらの研究の多くは latent representation の分離を主対象にしている。小売需要では、local と呼ばれる変動の中に、日単位の変動、時間帯単位の変動、日と時間帯の相互作用が混在する。また、latent を分けても decoder 内で情報が混ざれば、最終的な補正量がどの要因に由来するかは読みにくい。したがって、本研究では latent の分離そのものではなく、出力された residual correction を運用上読める成分として分けることに焦点を移す。
>
> Forecasting の分野でも、Deep Factors や DeepGLO のように global な共通構造と local な補正を組み合わせる研究、CoST、Autoformer、FEDformer、TimeMixer のように trend / season や multi-scale structure を分解する研究がある。これらは、時系列を単一のブラックボックスとして扱うのではなく、構造化して予測することの有効性を示している。しかし、多くの分解は raw series に対する coarse な trend / season や scale の分解であり、小売運用で必要な series / day / hour / interaction の粒度までは下りていない。
>
> さらに、統計学や時系列分析では、残差は単なるノイズではなく、基準モデルが説明しきれなかった構造を調べる対象として扱われてきた。Frisch-Waugh-Lovell や Robinson の residualization、Breusch-Pagan や Ljung-Box の残差診断、ARCH 型モデルは、基準モデルの後に残る変動や誤差構造を分析対象にできることを示している。ただし、残差の意味は基準モデルに依存するため、どの基準値の後に何が残るかを明示する必要がある。
>
> 以上を踏まえ、本研究では小売需要における強い基準値 $b$ の後に残る residual $r=y-b$ を対象にし、その補正量を series / day / hour / day-hour interaction の 4 成分として出力する。平均ゼロ制約により成分の担当範囲を固定し、予測精度だけでなく、成分回復、profile comparison、component ablation、high residual case の補正によって評価する。これにより、本研究は local/global 表現分離と分解型予測の知見を、baseline residual の output-level decomposition として小売需要に適用する。

## 9. 文献ごとの使い方まとめ

| 文献 | proposal での使い方 | 本研究との差分 |
|---|---|---|
| Frisch-Waugh-Lovell / Robinson | 残差を、説明済み成分を取り除いた後の対象として扱う根拠 | 本研究は因果・係数推定ではなく、小売 baseline 後の需要補正を扱う |
| Breusch-Pagan / Ljung-Box / ARCH | 残差構造を診断・モデル化できる根拠 | 本研究は分散や自己相関ではなく、series/day/hour/interaction の残差成分を扱う |
| FHVAE | 時間不変要因と時変要因を分ける発想の根拠 | 小売需要の output correction までは扱わない |
| DSVAE | static / dynamic latent factorization の根拠 | latent 分離中心で、出力補正量の担当範囲は保証しない |
| C-DSVAE | latent 間の情報混入を抑える問題意識の根拠 | 本研究は情報混入を output constraint で扱う |
| Decoupling Local and Global Representations | local/global 分離を本研究の出発点にする | local を day/hour/interaction に再分解し、output residual に移す |
| Deep Factors / DeepGLO | global 構造と local 補正を組み合わせる forecasting の根拠 | 本研究は residual correction を運用単位に分解する |
| CoST | 意味の異なる成分を分ける forecasting 表現の根拠 | trend/season ではなく series/day/hour/interaction を扱う |
| Autoformer / FEDformer | decomposition をモデル内部に組み込む予測の根拠 | raw series decomposition ではなく baseline residual decomposition を扱う |
| TimeMixer | microscopic / macroscopic な multi-scale 分解の根拠 | scale ではなく小売カレンダー・運用単位の成分を扱う |
| TNC / TS2Vec / TS-TCC / TF-C | 表現を複数タスク・複数粒度で評価する流れの根拠 | 本研究では pretraining ではなく output component 評価に使う |
| Shapelets / INSHAPE | 局所パターンが解釈対象になる根拠 | 本研究では subsequence ではなく hour / day-hour residual を扱う |
| Anomaly Transformer / TranAD / GenAD | high residual や異常的なケースを評価対象にする根拠 | 本研究では異常検知ではなく high residual correction を評価する |
