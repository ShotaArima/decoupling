# 2-Exp-35: FreshRetailNet-50K における成分分離の t-SNE 確認

## 背景

2-Exp-29 では、合成データの系列ごとの推定 hour 成分 profile（24次元）を t-SNE
で可視化し、既知の subgroup が分離することを確認した。2-Exp-21 では
FreshRetailNet-50K の平均 residual hour profile と推定 hour 成分の対応を確認したが、
系列ごとの分離構造を t-SNE と定量指標で確認していない。

本実験では、2-Exp-29 の出力成分版 t-SNE を FreshRetailNet-50K に移す。ただし、
FreshRetailNet-50K には synthetic の真の成分や真の subgroup がない。そのため、
「真の成分回復」とは呼ばず、実測残差に存在する系列別 hour pattern が推定 hour
成分に保存されるかを確認する。

ここで可視化するのは VAE/AE の latent vector 自体ではなく、decoder が出力した
`hour_component` である。本研究の保証対象が latent の一意な識別ではなく、
中心化された出力4成分であるため、2-Exp-29 と同じ方針を採る。

## 目的と仮説

`series_mean` baseline 後の残差を

```text
global + day + hour + interaction
```

へ分解したとき、次を確認する。

1. 実測 residual hour profile から独立に作った系列群が、推定 `hour_component`
   の24次元 profileにも保存される。
2. 中心化ありでは day/hour の平均ゼロ制約と interaction の両周辺平均ゼロ制約が
   数値的に成立する。
3. 中心化なしではクラスタ構造が見える場合でも、4成分の担当範囲は固定されない。
4. city IDによる色分けは補助的に確認するが、city分離を成功条件にはしない。

## 合成データ実験との対応

| 2-Exp-29 synthetic | 2-Exp-35 FreshRetailNet-50K |
|---|---|
| 真の subgroup（peak 7/12/18/21時） | 実測 residual hour profile にK-meansを適用した4群 |
| 真の hour 成分 | 観測セルだけで計算した residual hour profile |
| 推定 hour 成分 | 推定 `hour_component` |
| subgroup色付きt-SNE | 実測profile群およびcity ID色付きt-SNE |
| 真値との相関 | 実測profileとの系列別相関 |

K-meansの群は実測 residual から作り、推定成分からは作らない。t-SNE上の座標も
評価には使わず、定量指標は元の24次元profile空間で計算する。

## 実験条件

本番設定は `configs/2-Exp-35_freshretailnet_component_tsne.json`。

- dataset: FreshRetailNet-50K
- sample: `store_id × product_id`
- history: 28日 × 24時間
- baseline: `series_mean`
- train / validation / test: 6000 / 1500 / 1500系列
- validation: train split の非重複holdout
- test: eval split
- seeds: 17, 23, 31
- epochs: 12
- model: 4出力成分 `global/day/hour/interaction`
- variants:
  - `output_decomp_no_center`: 中心化なし対照
  - `output_decomp_centered`: 中心化あり提案

smoke設定は `configs/2-Exp-35_freshretailnet_component_tsne_smoke.json`。
ローカルcacheの300 / 300 / 100系列を用い、1 epochで入出力だけを確認する。

## 評価指標

### hour pattern の保存

各系列について、観測セルのみを使って residual と推定hour成分を日方向に平均し、
平均0・L2 norm 1へ正規化する。

| 指標 | 内容 |
|---|---|
| `profile_corr_median` | 系列ごとの実測profileと推定profileのPearson相関の中央値 |
| `aggregate_profile_corr` | 全系列平均profile同士の相関 |
| `cluster_recovery_ari` | 実測profile群と推定profile側K-means群のARI |
| `cluster_recovery_nmi` | 同じ2群のNMI |
| `empirical_cluster_silhouette_in_component_space` | 実測profile群が推定成分空間で分かれる度合い |
| `city_id_silhouette_in_component_space` | city IDの分離度。探索的指標 |

### 中心化制約

| 指標 | centeredで期待する値 |
|---|---:|
| `day_center_abs_mean` | `< 1e-6` |
| `hour_center_abs_mean` | `< 1e-6` |
| `interaction_day_marginal_abs_mean` | `< 1e-6` |
| `interaction_hour_marginal_abs_mean` | `< 1e-6` |

成分間相関も補助指標として保存する。中心化モデルでは、global/day/hour/interaction
が出力グリッド上で直交するため、相関の絶対値はほぼ0になるはずである。

## 事前に定める成功条件

本番3 seedのうち2 seed以上で、中心化ありモデルが次を満たすことを主成功条件とする。

- `profile_corr_median >= 0.50`
- `cluster_recovery_ARI >= 0.25`
- `cluster_recovery_NMI >= 0.30`
- `empirical_cluster_silhouette_in_component_space >= 0.10`
- 4つの中心化違反指標がすべて `< 1e-6`

city IDのsilhouetteは成功判定に使わない。FreshRetailNet-50Kのcityとhour patternが
一致するという事前仮説がなく、評価系列のクラス不均衡にも影響されるためである。

中心化なしよりt-SNEが視覚的に明瞭であることも成功条件にしない。中心化の役割は
クラスタを人工的に強めることではなく、成分の平均・周辺平均を固定して担当範囲を
一意にすることである。

## 実行方法

smoke学習:

```bash
uv run decoupled-ts residual-sweep \
  --config configs/2-Exp-35_freshretailnet_component_tsne_smoke.json
```

smoke解析:

```bash
uv run --with matplotlib python scripts/analyze_2_exp_35_freshretailnet_tsne.py \
  --run-dir runs/2-Exp-35_freshretailnet_component_tsne_smoke/series_mean_all \
  --out-dir figures/2-Exp-35-smoke
```

本番学習:

```bash
uv run decoupled-ts residual-sweep \
  --config configs/2-Exp-35_freshretailnet_component_tsne.json
```

本番解析:

```bash
uv run --with matplotlib python scripts/analyze_2_exp_35_freshretailnet_tsne.py \
  --run-dir runs/2-Exp-35_freshretailnet_component_tsne/series_mean_all \
  --out-dir figures/2-Exp-35
```

カテゴリで色分けする場合は、`static_id_columns` のindexを指定する。例えば
first categoryはindex 3である。

```bash
uv run --with matplotlib python scripts/analyze_2_exp_35_freshretailnet_tsne.py \
  --run-dir runs/2-Exp-35_freshretailnet_component_tsne/series_mean_all \
  --out-dir figures/2-Exp-35-first-category \
  --metadata-index 3 --metadata-name first_category_id
```

## 出力

各variantは解析用に次を保存する。

```text
global_component.npy
day_component.npy
hour_component.npy
interaction_component.npy
residual.npy
observed.npy
static_ids.npy
```

解析スクリプトの出力は次のとおり。

```text
figures/2-Exp-35/
  freshretailnet_hour_component_tsne.pdf
  freshretailnet_hour_component_tsne.png
  freshretailnet_hour_cluster_profiles.pdf
  freshretailnet_hour_cluster_profiles.png
  metrics.json
  metrics_by_seed.csv
  representative_seed_assignments.csv
```

`freshretailnet_hour_component_tsne` は、中心化なし/ありを行に置き、実測残差由来の群と
city IDで色分けする。`freshretailnet_hour_cluster_profiles` は各群について、
実測 residual profile と推定hour成分profileを重ねる。

## 図の見方

実際に生成した代表seedの図は次のとおりである。

- [hour component t-SNE](../figures/2-Exp-35/freshretailnet_hour_component_tsne.png)
- [hour cluster profiles](../figures/2-Exp-35/freshretailnet_hour_cluster_profiles.png)

### `freshretailnet_hour_component_tsne.png`

この図では、1点が1つの `store_id × product_id` 系列を表す。点同士が近いほど、
推定されたhour component profileの形が似ている。t-SNEの横軸・縦軸の値自体には
意味がなく、点の近さと色のまとまりだけを見る。

| 位置 | 条件 | 色 | 見る内容 |
|---|---|---|---|
| 左上 | 中心化なし | 実測 residual profileから作った4群 | 推定hour成分が実測patternを保持できるか。色が混ざりやすい |
| 右上 | 中心化なし | `city_id` | 分離が都市IDの単純な分類ではないことを確認する補助図 |
| 左下 | 中心化あり | 実測 residual profileから作った同じ4群 | 中心化によって実測hour patternの対応が強くなるか |
| 右下 | 中心化あり | `city_id` | 中心化後も都市IDは混在するか。hour成分と都市分類を混同しないための確認 |

左側が主たる比較である。左上と左下を比べ、中心化ありの左下で同じ色の点が
よりまとまれば、中心化によって推定hour componentが実測residualのhour patternを
保持しやすくなったと解釈する。右側は成功条件ではない。`city_id`とhour patternは
同じ要因とは限らないため、city IDが分離しないことは失敗を意味しない。

色の番号 `0`〜`3` に意味の順序はない。例えばcluster 0を朝型、cluster 1を夜型と
直接呼んではならない。各clusterのprofile形状は、次のprofile図で確認する。

### `freshretailnet_hour_cluster_profiles.png`

この図はt-SNEより直接的である。横方向の4列がcluster 0〜3、上段が中心化なし、
下段が中心化ありである。

- 黒線: 観測セルから計算した実測 residual hour profile
- 赤い破線: モデルが出力した推定 `hour_component` profile
- 上段と下段で黒線がほぼ同じ: 同じデータの実測profileを表示しているため
- 下段で赤破線が黒線に近づく: 中心化ありでhour patternをhour componentが担当できている

まず上段では、赤破線が0付近に留まり、黒線の時間帯変化を十分に追えていないかを
確認する。次に下段で、赤破線が黒線の山・谷・符号の変化に近づいているかを見る。
上下段で変化するのはモデルの赤破線であり、黒線ではない。

### 初見の読者に示す順番

図だけを最初に見せると、t-SNEは印象に依存しやすく、中心化の効果が伝わりにくい。
次の順番で説明すると、主張と根拠の対応が明確になる。

1. まずprofile図の上段/下段を示し、同じ実測profileに対して、中心化なしでは赤破線が
   合わず、中心化ありでは黒線に近づくことを確認する。
2. 次にt-SNEの左上/左下を示し、系列単位でも実測residual profile由来の色のまとまりが
   中心化ありで強くなることを確認する。
3. 図の印象を数値で固定する。中心化ありでは profile correlation `0.9854 ± 0.0123`、
   ARI `0.6538 ± 0.0806`、NMI `0.6077 ± 0.0601`であり、中心化なしの
   `0.7870 / 0.0421 / 0.0743`を上回る。
4. 最後に、中心化違反量が中心化ありで全て `1e-6`未満、中心化なしではday平均や
   interaction周辺平均が残ることを示し、改善が単なる見た目ではなく、成分の担当範囲を
   固定する制約の効果であることを説明する。

この順番であれば、読者は「実測patternがある」→「推定hour成分がそれを捉える」→
「中心化ありで再現性が高い」→「制約も数値的に成立する」という4段階で納得できる。
ただしsilhouetteは `0.0836 ± 0.0054`で事前閾値 `0.10`を下回るため、4群が完全に
分離したとは説明しない。主張は、完全なlatent識別ではなく、中心化された出力hour
成分が実測残差のhour patternを高い相関と群対応で保持したことに限定する。

## 解釈上の制約

- t-SNEの分離だけから一般化性能や潜在変数の識別性を主張しない。
- 実測profile群は真の生成要因ではなく、データ駆動の参照ラベルである。
- 実測 residual と推定成分は同じ系列を使うため、主張は「保持された構造」の確認に
  限定し、未知系列への分類性能とは呼ばない。
- 4成分の意味づけはt-SNEではなく、中心化違反量、成分profile、ablation、
  既存の2-Exp-17/21/28と合わせて判断する。

## smoke確認

ローカルCPUで2026-07-20にFreshRetailNet-50K smokeを実行し、次を確認した。

- 中心化あり/なしの2 variantが1 epoch完走した。
- 解析に必要なcomponent、residual、observed、static IDの各`.npy`が保存された。
- t-SNE図、クラスタprofile図、JSON、CSVが生成された。
- 中心化ありの制約違反量は、day `5.3e-9`、hour `2.9e-8`、
  interactionのday/hour周辺平均 `2.3e-9 / 1.6e-11` だった。
- 中心化なしではday平均 `0.334`、interactionの両周辺平均 `0.0873` が残った。

smokeのeval先頭100系列はcity IDが1種類だけだったため、city色分けパネルは自動的に
省略された。本番1500系列またはcategory指定時には、2クラス以上があれば表示される。

1 epochのsmokeでは中心化ありの`profile_corr_median`は0.236、ARIは0.003であり、
成功条件を満たさない。これは性能評価ではなく入出力確認の結果であり、本番12 epoch・
3 seedの判定とは分ける。

## 本番結果

`ssh my` 上で本番設定を実行し、seed 17 / 23 / 31の結果を取得してローカルで解析した。

| variant | profile corr median | ARI | NMI | silhouette | 中心化違反 |
|---|---:|---:|---:|---:|---|
| no center | 0.7870 +/- 0.0188 | 0.0421 +/- 0.0270 | 0.0743 +/- 0.0335 | -0.0082 +/- 0.0161 | 残る |
| centered | 0.9854 +/- 0.0123 | 0.6538 +/- 0.0806 | 0.6077 +/- 0.0601 | 0.0836 +/- 0.0054 | すべて `< 1e-6` |

中心化ありでは、3 seedすべてで profile corr、ARI、NMI、4つの中心化制約を満たした。
一方、silhouetteは3 seedとも事前閾値 `0.10` をわずかに下回った。そのため、
「実測残差のhour patternを推定hour成分が強く保持する」と結論づけられるが、
「4つのクラスタが完全に分離する」とまでは主張しない。

city IDのsilhouetteは平均 `-0.0403` であり、hour成分の分離がcity IDの単純な
分離ではないことも確認した。t-SNEでは中心化ありの4群が視覚的にまとまる一方、
city IDは混在している。

したがって、FreshRetailNet-50Kについての2-Exp-35の回答は、次のとおりである。

> 合成データと同じ「既知の真値subgroup回復」はできないが、実測残差から独立に
> 抽出した系列別hour patternは、中心化された推定hour成分に高い相関とクラスタ
> 対応をもって保存される。中心化制約は実データでも機能する。ただし、t-SNEを
> 根拠に完全なlatent識別や完全クラスタ分離を主張することはできない。
