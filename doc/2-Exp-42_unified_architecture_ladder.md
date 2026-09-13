# 2-Exp-42: 統一アーキテクチャ ladder（分水嶺 B: 1 Encoder / 1 Decoder の非劣性）

作成日: 2026-08-22
対応: [proposal/direct_y_and_unified_encoder_decoder.md](proposal/direct_y_and_unified_encoder_decoder.md)
の検証ラダー「2-Exp-44 案」（仮説 H-B1）、ボード項目 H-EXT-05、
[proposal/unified_model_direction.md](proposal/unified_model_direction.md) の Stage A/B
（番号は実施順に合わせて 42 とする。direct 系の 42/43 案は後続番号へ繰り下げ）
関連: [2-Exp-40_jima_main_results.md](2-Exp-40_jima_main_results.md)、
[2-Exp-41_direct_vs_residual_output_decomp.md](2-Exp-41_direct_vs_residual_output_decomp.md)

## 背景

現行の主提案は 3 Encoder + interaction MLP + 4 Decoder head + 後処理 centering という
4 系統構成であり、「1 つの原理から 4 成分が導かれるモデル」としての主張がしづらい
（unified_model_direction K1/K2）。元論文（Tonekaboni et al. 2022）は
Enc_l / Enc_g + **1 Decoder** であり、統合は元論文への回帰でもある。

鍵は 2 点。

1. **Global/Local の違いは「どの軸で pool するか」の違い**。1 つの backbone への
   4 種の pooling 読み出しで、4 つの Encoder を置き換えられる。
2. **中心化（ANOVA 分解）は線形冪等射影**（T-EXT-01/02）。1 つの Decoder が
   grid を 1 枚出せば、射影で中心化済み 4 成分が構造として得られ、和は保存される。

## 実装（本 PR に含むコード変更）

`residual_models.py` に `UnifiedDecompositionResidualModel`（variant type
`unified_decomposition`）を追加した。

```text
backbone: cell-level MLP（[grid, mask] → hidden）……共通
z_global = proj_g( pool_{d,h}(feats) )   （両軸 masked pool）
z_day    = proj_d( pool_h(feats) )       （hour 軸 pool、日ごと）
z_hour   = proj_h( pool_d(feats) )       （day 軸 pool、時間帯ごと）
z_u      = proj_u( feats )               （pool なし、cell ごと）
```

| architecture | 出力側 | 中心化 | 対応 Stage |
|---|---|---|---|
| `shared_encoder` | 現行と同型の 4 head | hard centering（現行方式） | Stage A |
| `single_decoder` | **1 Decoder**: cell ごとに `[z_g, z_day(d), z_hour(h), z_u(d,h)]` を受けて grid を 1 枚出力 → **ANOVA 射影**で 4 成分へ分解 | 射影（構造保証、`center_components` 不要） | Stage B |

`single_decoder` の Decoder 入力は元論文の `Dec([z_g, z_t])` の 2 軸版そのものである。
射影は grand mean / 行 / 列 / 残りへの直交分解なので、中心化違反は定義上ゼロになる。

## 目的と仮説

**H-B1（主仮説）**: 成分の担当範囲は射影（または centering）が構造として固定する
ため、Encoder/Decoder の共有は成分品質・補正性能を落とさない。統一モデルは
現行 4 系統に対して**非劣性**を保ちつつ、パラメータ数を削減する。

成立すれば、「1 つの backbone + 1 つの Decoder + 射影」という単一原理のモデルとして
APIEMS / 修論の主提案候補になる（元論文回帰 + 2 軸拡張 + 出力保証）。

## 実験条件

2-Exp-40 と同一条件（residual `series_mean`。direct との組み合わせは
2-Exp-41 の結果を見て後続実験で行う）。

- dataset: FreshRetailNet、6000/1500/1500、train_holdout
- model: hidden 160、dims 10/10/10/8
- train: epochs 上限 60、early_stopping_patience 8、lr 1e-3
- seeds: 17, 23, 31, 47, 59

| variant | 構成 | 位置づけ |
|---|---|---|
| `output_decomp_centered` | 現行 4 系統 | reference（2-Exp-40 主表と同条件） |
| `unified_shared_encoder` | 共有 backbone + 4 head | Stage A |
| `unified_single_decoder` | 共有 backbone + 1 Decoder + 射影 | Stage B（主提案候補） |

合計 run 数: 1 scenario × 5 seed × 3 variant = 15。
パラメータ数は実行後に集計し、削減率を報告する。

## 評価指標

| 指標 | 見る理由 |
|---|---|
| `corrected_cell_mae` / WAPE / top10 / bias | 非劣性の主判定 |
| `component_*_mean_abs`（中心化違反） | 射影の構造保証の確認（single_decoder は 0 のはず） |
| `hour_component_residual_profile_corr` | 成分品質の維持 |
| seed 間 SD | 共有化による安定性変化（H2: 下がる期待） |
| パラメータ数（別集計） | 削減の定量化 |
| best / stopped epoch | 収束挙動の変化 |

## 事前に定める成功条件・分岐

1. **非劣性マージン**: 統一モデル（いずれかの architecture）の corrected MAE が
   reference に対して paired 平均差 **+0.0025 以内**（2-Exp-37/41 と同じ幅）なら
   非劣性成立とする。
2. **構造保証**: `unified_single_decoder` の中心化違反 4 指標がすべて 1e-6 未満
   （射影の実装確認）。
3. **成分品質**: hour profile corr が reference 比 -0.05 以内。
4. Stage B が非劣性を満たせば主提案候補に採用。Stage A のみ成立なら
   「読み出しヘッドの分岐は残すが原理は 1 つ」（unified_model_direction のリスク節）
   へ縮退。両方不成立なら現行 4 系統を維持し、共有化は今後課題に戻す。

## 実行方法

smoke:

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-42_unified_architecture_ladder_smoke.json
```

本番:

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-42_unified_architecture_ladder_freshretailnet.json
```

## 出力

```text
runs/2-Exp-42_unified_architecture_ladder_freshretailnet/
  series_mean/seed_{17,23,31,47,59}/{3 variants}/
  all_results.csv / aggregate.csv / summary.json
```

## smoke 確認

2026-08-22 にローカル CPU で smoke（synthetic、3 variant、seed 17）を実行し、
3 variant の完走と、`single_decoder` の射影による中心化違反 ~1e-9 を確認した。

## 実行中に発見したリークと修正（重要）

**初回の本番実行は無効である。** 初回実装では interaction latent を
`z_u = proj_u(cell特徴)` と cell-level 特徴から作っていたため、
入力チャネル 0（観測残差）を interaction 経路がコピーする恒等写像に退化し、
corrected MAE 0.0048〜0.0068 という異常値（reference の 10 倍良い）が出た。
これは 2-Exp-33 の「完全な経験的 ANOVA は窓内で恒等写像に退化する」と同型の
リークである。

修正: interaction latent を現行モデルと同じく `InteractionEncoder(z_day, z_hour)`
（pooled latent のみ）から構成するよう変更した。現行モデルのこの設計は、
まさにこのリークを塞ぐ役割を持っていたことが判明した。

**設計上の教訓: 窓内 transductive 評価では、cell-level の読み出し経路は
観測値コピーへの抜け道になる。** 修正後の smoke では unified 系の MAE が
reference と同水準（1.41 前後）に戻ることを確認した。

## 本番結果（リーク修正後）

2026-08-22 にローカル GPU で実行。epoch 上限 60 → unified 系が全 run 上限到達
（best 58〜60）のため上限 120 で再実行 → **なお全 run 上限到達**（best 118〜120）。
reference は 60 以内で early stopping する（best 27〜51）。

### 窓内評価（上限 120。unified は未収束のまま）

| variant | corrected MAE | top10 | bias | hour corr | params |
|---|---:|---:|---:|---:|---:|
| reference（現行 4 系統） | 0.04828 ± 0.00032 | 0.1942 | -0.263 | **0.9941** | 188,202 |
| unified 共有 backbone（Stage A） | 0.02740 ± 0.00091 | 0.1071 | -0.179 | 0.9774 | 44,842 |
| unified 1Enc+1Dec+射影（Stage B） | 0.02957 ± 0.00140 | 0.1176 | -0.182 | 0.9709 | 43,879 |

paired: Stage A -0.02088、Stage B -0.01871（いずれも 5 勝 0 敗）。

**ただしこの窓内の差は額面どおりに受け取れない。** 理由:

- unified の backbone は cell ごとに非線形変換してから pool するため、
  `z_day(d)` がその日の 24 時間プロファイルを豊かに符号化できる。
  interaction 経路 `f(z_day, z_hour)` がこれを使うと、観測残差の
  **一段間接的な記憶**が可能になる（修正したリークの弱い版）。
- 実際、窓内 MAE の改善に伴い hour profile corr は低下しており
  （0.994 → 0.971〜0.977）、u 成分が構造ではなく記憶を蓄えている兆候がある。
- epoch を増やすほど窓内差が拡大し続ける（60→120 で -0.005 → -0.021）ことも
  記憶仮説と整合する。

このため、公正なアーキテクチャ比較として**未来日評価を追加実行**した。

### 未来日評価（`future_mask_days: 2`。コピー経路が遮断される決定的な比較）

| variant | future corrected MAE | 未来日改善率 | future hour corr |
|---|---:|---:|---:|
| reference | 0.05487 ± 0.00106 | 23.7% | 0.9709 |
| unified Stage A | 0.05296 ± 0.00022 | 26.3% | 0.9404 |
| **unified Stage B** | **0.05090 ± 0.00060** | **29.2%** | **0.9766** |

paired（future corrected MAE）:

| 比較 | 平均差 | 勝敗 |
|---|---:|---|
| Stage A − reference | -0.00191 ± 0.00114 | **Stage A 5勝0敗** |
| Stage B − reference | **-0.00397 ± 0.00133** | **Stage B 5勝0敗** |

### 事前に定めた成功条件との照合

1. **非劣性（+0.0025 以内）**: 未来日評価で Stage A/B とも成立。Stage B は
   非劣性を超えて**優越**（-0.0040、5 勝 0 敗）。窓内評価は記憶混入のため
   主判定に使わない（下記の方針変更）。
2. **構造保証**: 全条件で中心化違反 < 1e-8（Stage B は射影により ~3e-9）。
3. **成分品質**: 未来日 hour corr は Stage B が reference を上回る（0.977 vs 0.971）。
   Stage A は低い（0.940）ので、主提案候補は Stage B とする。
4. **パラメータ**: 188k → 44k（**77% 削減**）。

### 結論

1. **H-B1 は成立し、Stage B（1 Encoder + 1 Decoder + ANOVA 射影）を主提案候補とする。**
   未来日評価で reference に 5 seed 全勝（改善率 23.7% → 29.2%）、
   パラメータ 77% 削減、中心化は射影による構造保証。
2. **評価方法の教訓（本実験のもう 1 つの成果）**: 窓内 transductive 評価は
   容量の大きい encoder では観測値の記憶で汚染される。アーキテクチャ比較の
   主判定は未来日評価（または held-out 期間）で行うべきである。
   これは 2-Exp-30/33 の「窓内では素朴分解と構造学習を区別できない」を
   アーキテクチャ比較のレベルで再確認したものである。
3. 残る注意: unified 系は 120 epoch でも上限到達しており、収束値は
   さらに良い可能性がある。ただし窓内の伸びは記憶成分を含むため、
   epoch 上限の引き上げは未来日指標が伸びる範囲でのみ意味がある。
   後続実験で未来日指標での early stopping（selection metric の変更）を検討する。

### 後続実験への接続

- 主提案候補が Stage B に決まったため、後続は
  (i) 未来日指標ベースの model selection、(ii) unified の hidden/latent 次元感度、
  (iii) 2-Exp-39 の未来日 5 seed 比較への unified 行の追加、
  (iv) 軸別 swap 正則化（H-EXT-08）の unified 上での検証、の順で行う。
- direct × unified の組み合わせは、2-Exp-41 で direct 化自体が棄却されたため行わない。
