# 2-Exp-29: 論文図用 synthetic 成分回復の可視化

## 背景

論文(APIEMS 2026)の論証構造のうち、P3「出た成分は本物か」を支える図が表しかなかった。

- R1(回復): synthetic の成分回復は Exp-22 の相関表のみ
- R2(実データ対応): hour corr は Exp-21 の CSV のみで論文用の図がない
- R4(有用性): 推定成分だけで系列の型(subgroup)が識別できることは未可視化

元論文 (Tonekaboni et al.) は Fig 4(t-SNE)と Fig 5(反実仮想生成)で latent の分離を可視化している。
本研究は latent の識別を主張しない方針のため、これらの **出力成分版** を図にする。

## 目的

論文用の図を 2 枚(+スライド用素材)を再現可能な形で生成する。

| 図 | 内容 | 証明対象 |
|---|---|---|
| Fig.1 | 実データ(FreshRetailNet)の hour profile overlay。series_mean と same_hour_d7 を同一縦軸で並べる | R2 + 基準値診断(P5) |
| Fig.2 | synthetic の subgroup 別「真の hour 成分 vs 推定 hour 成分」4 パネル + t-SNE パネル(任意) | R1 + R4 |

Fig.2 の t-SNE は、per-series の推定 hour 成分 profile(24 次元)を subgroup で色分けしたもの。
元論文 Fig 4 の「global 表現の t-SNE」の出力成分版であり、latent には触れない。

## 実験構成

- config: `configs/2-Exp-29_synthetic_component_recovery_figure.json`
  - Exp-22 final の `base` シナリオと同一設定(1500 系列、35 日、24 時間帯、noise 0.12)
  - variant は `output_decomp_centered` のみ、seed 17 のみ
  - `output.save_latent_arrays: true` により `hour_component.npy` / `true_hour.npy` /
    `subgroup.npy` などの per-series 配列を保存する(既存パイプライン機能。コード変更なし)
- smoke: `configs/2-Exp-29_synthetic_component_recovery_figure_smoke.json`(120 系列、1 epoch)

subgroup と真の peak hour の対応は生成コード(`retail_data.py` の
`component_residual_retail`)で `peaks = [7, 12, 18, 21]`。

## 実行コマンド

リモート(`ssh my`)のリポジトリ直下で実行する。

smoke(動作確認、CPU でも数十秒):

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-29_synthetic_component_recovery_figure_smoke.json
```

本番:

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-29_synthetic_component_recovery_figure.json
```

図の生成(matplotlib は依存に含まれないため `--with` で注入):

```bash
# Fig.1: 実データ hour profile overlay(2-Exp-21 の出力が必要)
uv run --with matplotlib python scripts/plot_fig1_hour_profile_overlay.py \
    --run-dir runs/2-Exp-21_freshretailnet_visualization \
    --out figures_paper/fig1_hour_profile_overlay.pdf --png

# Fig.2: synthetic 成分回復(subgroup 4 パネル + t-SNE)
uv run --with matplotlib python scripts/plot_fig2_component_recovery.py \
    --variant-dir runs/2-Exp-29_synthetic_component_recovery_figure/base/seed_17/output_decomp_centered \
    --out figures_paper/fig2_component_recovery.pdf --tsne --png
```

PDF を論文リポジトリの `images/` にコピーする。

## 出力

```text
runs/2-Exp-29_synthetic_component_recovery_figure/base/seed_17/output_decomp_centered/
  hour_component.npy   # 推定 hour 成分 (N, D, H)
  true_hour.npy        # 真の hour 成分 (N, D, H)
  subgroup.npy         # subgroup index (N,)
  (day/global/interaction も同様に保存)
figures_paper/
  fig1_hour_profile_overlay.pdf/.png
  fig2_component_recovery.pdf/.png
```

## 成功条件

- Fig.1: series_mean パネルで corr が 0.99 前後、same_hour パネルで残差 profile が
  ほぼ平坦(振幅が 1 桁小さい)であること。ローカル確認済み: 0.9961 / −0.8946。
- Fig.2: 4 subgroup すべてで真の成分と推定成分の profile corr が高く(Exp-22 実績では
  hour corr 0.999)、peak 位置(7/12/18/21 時)が一致すること。
  t-SNE で subgroup ごとのクラスタが分かれること。

## 悪い結果の場合

- Fig.2 の corr が低い場合: Exp-22 base の hour corr 0.9991 と矛盾するため、
  設定差(seed、epochs)をまず疑う。seed を 23/31 に変えて再実行する。
- t-SNE が分離しない場合: t-SNE パネルは落とし、4 パネル版のみを論文に使う
  (R1 の主張は 4 パネルだけで成立する)。

## smoke 確認

ローカル(CPU)で実行済み(2026-07-12):

- sweep が完走し、`hour_component.npy` / `true_hour.npy` / `subgroup.npy` が保存される
- `plot_fig2_component_recovery.py --tsne` が 5 パネルの PDF/PNG を出力する
- smoke は 1 epoch・テスト 18 系列のため corr は低い(0.37〜0.98)。性能判断には使わない

## 論文での使い方

- Fig.1 → 5 章「基準値の感度と calibration」付近。基準値により残差に残る構造が
  変わること(診断)と、hour 成分が残差構造と対応すること(R2)を示す
- Fig.2 → 5 章「合成データ実験」。成分回復表(Exp-22)の視覚化(R1)。
  t-SNE パネルは「推定成分だけで系列の型が識別できる」(R4)の補助証拠
- 関連: doc/conference/apiems2026/paper/check_list.md 項目 4

## 予約

2-Exp-30 は入力介入実験(discount/holiday を反実仮想に変えて成分応答を見る、R3)用に予約。
