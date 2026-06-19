# Related Work and Improvement Plan

## 目的

このメモは、周辺研究レビューを踏まえて、本研究をどこに位置づけ、今後どの方向に改良するべきかを整理する。

結論から言うと、本研究は「local/global の潜在表現を分ける研究」そのものではなく、次の位置づけにするのが最も筋が通る。

```text
強い baseline 後の residual に残る構造を、
series / day / hour / interaction の出力成分として分解し、
補正と解釈の両方に使う研究。
```

したがって、関連研究は以下の流れで整理する。

1. static/dynamic, global/local の潜在表現分離
2. self-supervised / contrastive な時系列表現学習
3. global-local forecasting と trend/season decomposition
4. local pattern interpretability と anomaly/high-residual 評価
5. 既存研究との差分としての residual output decomposition

## 現状の課題

現在の実験から見えている課題は 6 つある。

| 課題 | 現状 | proposal での扱い |
|---|---|---|
| latent split だけでは弱い | 2-Exp-27 で `global/local` residual が latent four-factor より良い | 出力分解を主張の中心にする |
| direct forecasting では interaction が不安定 | 2-Exp-26 で interaction 付きは WAPE が悪化 | raw demand ではなく residual target へ進む理由にする |
| residual の有効性は baseline に依存する | `series_mean` では改善、same-hour 系では改善が小さい | 適用条件として明記する |
| 実データ interaction が弱い | synthetic では強いが FreshRetailNet では主張しにくい | interaction は synthetic と limitation 中心 |
| global latent の解釈性が弱い | subgroup probe が不安定または 0.0 | latent 解釈ではなく output 解釈に寄せる |
| bias / high residual の trade-off がある | calibration や bias 制約で結果が変わる | 補正性能だけでなく high residual と bias も評価する |

このため、今後の改良は「より複雑な latent を作る」より先に、**出力成分の制約、baseline 診断、残差構造の可視化、評価軸の整理**を優先する。

## 周辺研究の地図

### 1. Static/Dynamic・Global/Local 表現分離

FHVAE, DSVAE, C-DSVAE, S3VAE, GP-VAE, Decoupling Local and Global Representations of Time Series は、系列に含まれる時間不変要因と時間変動要因を分ける流れである。

この系譜から得られる示唆は次である。

- 時系列には、系列全体に共通する global/static 要因と、時点ごとに変化する local/dynamic 要因がある。
- local 表現には時間相関があるため、GP prior や temporal smoothness が自然である。
- counterfactual regularization や contrastive objective により、latent 間の情報混入を抑えようとする研究がある。

本研究との違いは、既存研究の多くが **latent space の分離**を主対象にするのに対し、本研究は **output space の成分分解**を主対象にする点である。

2-Exp-27 の結果は、この違いを支持している。latent を `global/day/hour/interaction` に分けるだけでは `global/local` reference を上回らなかった。したがって、本研究では latent の完全な識別ではなく、出力成分が ANOVA 的制約を満たすことを重視する。

### 2. Self-Supervised / Contrastive 時系列表現学習

CPC, TNC, TS2Vec, TS-TCC, TF-C などは、ラベルなし時系列から有用な表現を学ぶ研究である。

この系譜から得られる示唆は次である。

- 時系列表現は、分類・予測・異常検知など複数タスクで評価できる。
- timestamp-level, subsequence-level, instance-level など、表現の粒度を変えることが重要である。
- 時間領域と周波数領域、局所文脈と大域文脈など、複数ビューの整合性が有効である。

本研究への改良案としては、残差系列に対して self-supervised pretraining を行い、その後に output decomposition を学習する流れが考えられる。

ただし、現段階では主張を増やしすぎないため、self-supervised pretraining は本文の主実験ではなく今後課題に回すのがよい。

### 3. Global-Local Forecasting と分解型予測

Deep Factors, DeepGLO, TLAE は、global な共通構造と local な補正を組み合わせる forecasting の系譜である。CoST, Autoformer, FEDformer, TimeMixer は、trend/season や multi-scale decomposition をモデル内部に組み込む研究である。

この系譜から得られる示唆は次である。

- forecasting では、global な共通パターンと local な系列固有補正を分けることが有効である。
- trend/season decomposition や multi-scale decomposition は、性能だけでなく構造化された予測にも役立つ。
- ただし既存研究の分解は、多くの場合 trend/season や scale といった粗い分割に留まる。

本研究の差分は、分解の粒度を小売運用上の単位に落としている点である。

```text
既存研究: trend / season, global / local, coarse multi-scale
本研究: series / day / hour / day-hour interaction
```

また、既存の forecasting decomposition は raw series の予測を主眼に置く。一方、本研究は強い baseline 後の residual を対象にするため、baseline が何を吸収し、何が residual に残るかを診断する必要がある。

### 4. Local Pattern Interpretability と High-Residual Evaluation

Shapelets, INSHAPE, ROCKET, InceptionTime は、局所的な subsequence pattern や multi-scale local features の重要性を示す系譜である。Anomaly Transformer, TranAD, GenAD は、異常検知や high-residual case の評価に近い。

この系譜から得られる示唆は次である。

- 予測平均だけでなく、局所パターンや外れケースの説明が重要である。
- high residual case は、単なる誤差ではなく、運用上の検知・補正対象として意味を持つ。
- 表現の価値は forecasting MAE だけでなく、異常検知、subset discovery、原因分析でも評価できる。

本研究で high residual top10 を見るのは、この文脈に接続できる。つまり、提案法は平均的な予測器ではなく、baseline が外したケースをどの軸のズレとして説明するかを見る residual diagnostic tool として位置づけられる。

## 本研究の新規性

周辺研究を踏まえると、本研究の新規性は次の 4 点に整理できる。

1. 強い baseline 後の residual を主対象にする。
2. residual を `series/day/hour/interaction` という小売運用に対応した成分へ分ける。
3. latent ではなく output component に平均ゼロ・主効果除去制約を入れる。
4. 予測補正、成分回復、ablation、hour profile、high residual top10 を組み合わせて評価する。

特に重要なのは 3 である。

```text
既存の local/global disentanglement は latent の情報混入を抑えようとする。
本研究は、latent が完全に識別されることを要求せず、
出力された residual correction が ANOVA 的に読めることを保証対象にする。
```

2-Exp-26 と 2-Exp-27 は、この立場を補強する。

- direct target では day/hour split は小幅改善に留まる。
- residual target でも latent split だけでは `global/local` を安定して上回らない。
- したがって、主張の中心は latent split ではなく output decomposition に置くべきである。

## 改良方針

### 改良 1: 出力分解を主モデルとして固定する

今後は `latent_concat` 系を主提案にしない。

主提案は次に固定する。

```text
output_decomp_centered
```

理由:

- synthetic で成分回復が最も明確。
- centering なしでは成分相関が崩れる。
- 2-Exp-27 により、latent split だけでは不十分であることが確認された。

### 改良 2: Baseline/residual 診断を前処理として明示する

残差学習は baseline に依存するため、どの residual target を使うかを診断する必要がある。

最低限、次を診断する。

- residual hour profile の train/test 再現性
- residual variance の大きさ
- high residual top10 の分布
- baseline MAE と residual structure の関係

これにより、`series_mean` では効き、same-hour 系では効きにくい理由を「提案法の失敗」ではなく「baseline が residual structure を吸収した結果」と説明できる。

### 改良 3: Bias と high residual の二目的評価にする

FreshRetailNet では、全体 MAE、bias、高残差上位の補正が trade-off になる。

そのため、評価を次の 3 軸に固定する。

| 軸 | 指標 |
|---|---|
| 平均補正 | corrected cell MAE / WAPE |
| 外れケース補正 | high residual top10 corrected MAE |
| 系統誤差 | corrected bias |

単一の MAE だけで勝ち負けを決めない。

### 改良 4: Temporal prior / smoothness は今後課題にする

Decoupling 論文や GP-VAE 系の示唆から、day 成分や residual 成分に temporal prior を入れる余地がある。

候補:

- day component に smoothness penalty
- weekday periodic prior
- hour component に cyclic smoothness
- interaction component に low-rank または sparsity 制約

ただし、現段階で追加すると主張が散るため、本文の主実験には入れず、今後課題として扱う。

### 改良 5: Self-supervised pretraining は拡張案に留める

TS2Vec や TNC のような pretraining を residual 系列に入れる案はある。

候補:

- residual history から timestamp-level representation を pretrain
- day/hour component head を fine-tune
- high residual detection に transfer

ただし、現行 proposal の核は output decomposition であり、pretraining を入れると別論文の範囲になる。したがって、今後課題として短く述べるのがよい。

## Related Work に書くべき段落構成

本文の related work は次の 6 段落で書く。

1. 時系列には static/global と dynamic/local の因子が混在する。
2. FHVAE, DSVAE, C-DSVAE, Decoupling 2022 は latent disentanglement を扱う。
3. TNC, TS2Vec, TS-TCC などは自己教師あり表現の有効性を示すが、因子の運用解釈は暗黙的である。
4. Deep Factors, DeepGLO, CoST, Autoformer, FEDformer, TimeMixer は forecasting における global-local / decomposition の重要性を示す。
5. Shapelets や anomaly detection 系は local pattern や high residual case の実用的意義を示す。
6. 既存研究は coarse な latent/decomposition が中心であり、baseline 後の residual を `series/day/hour/interaction` の output component に分ける点が本研究の差分である。

締めは次の形にする。

```text
Prior work has demonstrated the value of local/global separation, contrastive time-series representations, and decomposition-based forecasting. However, most methods either disentangle latent representations or decompose raw time series at coarse levels such as trend and seasonality. In contrast, this work focuses on the residual left by strong retail baselines and constrains the output correction itself into operational components: series, day, hour, and day-hour interaction.
```
