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

## 結果記入欄

本番実行後、seed別指標、3 seed平均、成功条件の判定、図から読める範囲をここへ追記する。
