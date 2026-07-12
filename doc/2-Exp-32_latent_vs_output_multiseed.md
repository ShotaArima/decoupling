# 2-Exp-32: latent 分割型と成分別出力型の複数 seed 比較

## 目的

2-Exp-28 の単一 seed 結果を再検証し、中心化を伴う成分別出力型が
global/local latent 型を安定して上回るかを確認する。

## 評価設計

- FreshRetailNet 公式 train の系列0--1999を学習、2000--2499を validation に使う。
- 公式 eval の500系列を最終 test とし、モデル選択には使用しない。
- 5 seed: `17, 23, 31, 47, 59`。
- 2-Exp-28 と同じ6モデル、12 epoch、`series_mean` residual、calibrationなし。
- 全モデルで同じ系列分割を使い、variant ごとに乱数をresetして同一seed内の比較を揃える。
- 主指標は corrected cell MAE。副指標は corrected WAPE、high residual top10
  corrected MAE、residual R2。
- 主比較は `output_decomp_centered` と `paper_global_local_residual`。

## GPU実行方針

- 本番 config は `device: cuda` とし、CPUへの暗黙フォールバックを許さない。
- CUDA AMPを有効化する。
- validationはinference modeで実行する。
- latent probeとpermutation診断を無効化し、probe無効時はtrain全体の再推論も行わない。
- DataLoaderのみ2 workerでprefetchし、CPU上のモデル並列化は行わない。

## コマンド

smoke:

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-32_latent_vs_output_multiseed_smoke.json
uv run decoupled-ts residual-result-analysis --config configs/2-Exp-32_paired_analysis_smoke.json
uv run decoupled-ts residual-experiment --config configs/2-Exp-32_freshretailnet_gpu_smoke.json
```

FreshRetailNet本番:

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-32_latent_vs_output_multiseed_freshretailnet.json
uv run decoupled-ts residual-result-analysis --config configs/2-Exp-32_paired_analysis_freshretailnet.json
```

## 判定

- seedごとの値、平均と標準偏差、同一seed内の差と95% bootstrap区間を報告する。
- MAE差の区間が0未満なら、中心化あり成分別出力型の優位を支持する。
- 区間が0をまたぐ場合は優越性を主張せず、単一seedの傾向は再現しなかったと報告する。

## 結果

5 seed の実行と paired bootstrap 解析を完了した。

| model | corrected MAE | WAPE | residual R2 | top10 MAE |
|---|---:|---:|---:|---:|
| latent: global/local | 0.0614 ± 0.0030 | 0.9925 ± 0.0487 | 0.0627 ± 0.0590 | 0.2873 ± 0.0084 |
| latent: global/day/hour | 0.0625 ± 0.0009 | 1.0094 ± 0.0143 | -0.0140 ± 0.0357 | 0.3005 ± 0.0055 |
| latent: global/day/hour/interaction | 0.0619 ± 0.0008 | 1.0001 ± 0.0126 | -0.0058 ± 0.0125 | 0.2998 ± 0.0023 |
| 成分別出力、中心化なし | 0.0635 ± 0.0007 | 1.0268 ± 0.0115 | 0.0325 ± 0.0370 | 0.2903 ± 0.0073 |
| 成分別出力、中心化、interactionなし | 0.0595 ± 0.0032 | 0.9615 ± 0.0514 | 0.2440 ± 0.0518 | **0.2500 ± 0.0134** |
| 成分別出力、中心化、全成分 | **0.0583 ± 0.0011** | **0.9422 ± 0.0179** | **0.2483 ± 0.0624** | 0.2508 ± 0.0154 |

主比較の `output_decomp_centered - paper_global_local_residual` は次の通り。

| metric | paired差 | 95% CI | seed勝敗 |
|---|---:|---:|---:|
| corrected MAE | -0.00311 | [-0.00479, -0.00117] | 4勝1敗 |
| corrected WAPE | -0.0503 | [-0.0775, -0.0189] | 4勝1敗 |
| top10 corrected MAE | -0.0364 | [-0.0529, -0.0200] | 5勝0敗 |

中心化あり全成分は、global/local latent に対して平均 MAE を約5.1%、top10 MAEを
約12.7%改善し、paired差の95%区間はいずれも0をまたがなかった。
また、4成分 latent 型に対する MAE 差は -0.00358
（95% CI [-0.00470, -0.00235]、5勝0敗）、中心化なし成分別出力に対する差は
-0.00523（95% CI [-0.00615, -0.00408]、5勝0敗）であった。

一方、全成分とinteractionなしの MAE 差は -0.00119
（95% CI [-0.00414, 0.00163]、3勝2敗）であり、interactionを含めることの
一貫した優位は確認できなかった。したがって、Exp-32は「latent分割だけでは不十分で、
中心化を伴う成分別出力が有効」という主張を支持するが、実データにおけるinteraction
成分の追加価値までは支持しない。
