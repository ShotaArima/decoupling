# 2-Exp-31: 残差軸診断(相対振幅と permutation test)

## 背景

外部レビュー(2026-07-13、指摘3)より:

```text
診断指標と言うなら判定のプロトコルが必要。
図の corr = -0.895 を「意味がない」と退けるのは妥当だが、
振幅がどの程度なら「構造が残っている」と判定するのか。
振幅比を定量的な診断量として定義すれば、この主張は検証可能になる。
```

論文は「基準値がどの軸を取りこぼしているかの診断」を主張に昇格させる方針
(Intro/貢献への反映は本実験の結果確認後)。そのための**検証可能な判定規則**を実装する。

## 診断プロトコル

軸 k ∈ {day, hour} について、モデルに依存しない(model-free)診断量を定義する。

1. **相対振幅**: 観測残差の軸別平均 profile の標準偏差を、残差の平均絶対値で正規化する。

   ```text
   A_k = std_k( r̄_k ) / mean|r|
   ```

2. **permutation test**: 帰無仮説「軸 k に構造がない」の下では、軸 k 方向の
   セルの並びは交換可能である。そこで (残差, 観測フラグ) のペアを
   各行内(hour 軸なら系列×日ごと、day 軸なら系列×時間帯ごと)でシャッフルし、
   帰無分布の振幅を B 回計算して p 値を得る。観測パターンは保たれる。

3. **判定規則**: permutation test が棄却された軸についてのみ、
   対応する成分(およびその corr)を解釈する。
   棄却されない軸の成分 corr は解釈しない(例: same-hour baseline 後の
   hour corr = -0.89 は「構造がない」ため判定対象外)。

これは古典的残差診断(Breusch-Pagan の分散、Ljung-Box の自己相関)の系譜に
連なる「小売の運用軸(日・時間帯)についての残差診断」であり、
論文2章の引用と接続する。

## 実装

- `src/decoupled_ts/residual_experiments.py`
  - `residual_axis_diagnostics(arrays, config)`: 上記の診断量を計算。
    `diagnostics.n_permutations`(既定 0 = 検定は無効、振幅のみ)で制御。
    `run_variant` から常に呼ばれるが、既存実験の挙動は変えない
    (n_permutations 未指定なら振幅系メトリクスが増えるだけ)。
  - 出力キー: `diag_{day,hour}_relative_amplitude` / `diag_{day,hour}_profile_std` /
    `diag_{day,hour}_permutation_pvalue` / `diag_{day,hour}_null_amplitude_p95` /
    `diag_{day,hour}_null_relative_amplitude_p95` / `diag_residual_abs_mean`
- 診断は test 配列(eval 系列の残差と観測マスク)に対して行うため、
  同一 scenario 内の variant 間では同じ値になる(モデル非依存)。

### GPU 対応と進捗表示(2026-07-13 改修)

初版の numpy 実装は permutation ループがシングルスレッド CPU で、
FreshRetailNet 本番サイズ(1500×28×24、500回×2軸)で variant あたり約3.5分の
無音区間が発生していた。次の通り改修した。

- permutation の並べ替え(argsort / take_along_dim)と profile 計算を torch 化し、
  学習と同じ device(CUDA があれば GPU)でチャンク実行する。
  チャンクサイズは `diagnostics.permutation_chunk`(既定 16)。
- 乱数は CPU 側の seed 付き generator で生成してから device へ送るため、
  **帰無分布は device に依らず決定論的**(CPU/GPU で同一の p 値)。
- tqdm で `diagnostics {axis} permutations` の進捗バーを表示(残り時間 ETA 付き)。
- 計測: 本番サイズで CPU 47秒(旧 約3.5分)、GPU では数秒の見込み。
- ユニットテスト4条件+smoke を torch 版で再実行し、相対振幅は旧実装と完全一致、
  判定(棄却/非棄却)も同一であることを確認済み。

### latent probe の無効化(2026-07-13 追加改修)

permutation の GPU 化後も、probe 段階で数分の無音区間(CPU 400%超)が残った。
原因は `run_latent_probes` の z_interaction probe で、6000系列では
**約400万行 × 最大168クラス**の LogisticRegression(max_iter=1000)を
複数回 fit していたため。対処:

- `probes.enabled` フラグを追加(既定 true = 既存実験の挙動は不変)。
- 有効時は `latent probes (sklearn, CPU)` の tqdm 段階バー(4 stage)を表示。
- Exp-31 の config は両方 `"probes": {"enabled": false}` に設定。
  本実験の目的は model-free の残差診断+hour corr であり、latent probe は不要。
  (論文でも latent の解釈性は主張しない方針のため、この実験で probe を
  落とすことに主張上の損失はない)

## ユニット検証(ローカル、2026-07-13)

| 条件 | hour p | day p | 判定 |
|---|---:|---:|---|
| 純ノイズ | 0.905 | 0.866 | 両軸とも非棄却 ✓ |
| hour 構造のみ | 0.005 | 0.925 | hour のみ棄却 ✓ |
| day 構造のみ | 0.274 | 0.005 | day のみ棄却 ✓ |
| hour 構造+欠測30% | 0.005 | 0.378 | 欠測下でも正しく棄却 ✓ |
| n_permutations 未指定 | — | — | 振幅のみ出力(後方互換) ✓ |

smoke(synthetic、真の day/hour 構造あり)でも両軸 p = 0.0099(最小値)、
相対振幅 day 0.41 / hour 0.30 ≫ 帰無95%点 0.03 / 0.07 を確認。

## 実行コマンド

smoke(ローカル実行済み):

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-31_residual_axis_diagnostics_smoke.json
```

FreshRetailNet(リモート `ssh my`。6000/1500系列、3基準値 × 2 variant、
permutation 500回):

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-31_residual_axis_diagnostics_freshretailnet.json
```

## 見るべき出力

`runs/2-Exp-31_residual_axis_diagnostics_freshretailnet/{scenario}/seed_17/{variant}/metrics.json`

| scenario | 予想 |
|---|---|
| `series_mean_all` | hour: 棄却・相対振幅 ~0.5(構造あり→補正・解釈可)。day: 要観察 |
| `same_hour_recent_mean_d7_all` | hour: 相対振幅 ~0.04 で**非棄却が理想**(構造なし→corr は判定対象外)。ただし n=1500系列×30日で検出力が高いため、微小構造でも棄却される可能性あり。その場合は「統計的には非ゼロだが振幅が帰無95%点の数倍以内で実務上無視できる」という効果量ベースの読みに切り替える |
| `weekday_same_hour_mean_all` | hour: 吸収済みの想定。day: 曜日は吸収されるが曜日以外の日効果が残る可能性 |

**重要な注意(判定の設計)**: 検定は「構造が厳密にゼロか」を見るため、大標本では
実務上無視できる微小構造も棄却し得る。論文の判定規則は p 値のみに依存させず、
「p 値(存在)+相対振幅と帰無95%点の比(効果量)」の2段で書くのが安全。
例: 「p < 0.05 かつ A_k が帰無95%点の3倍以上」など。閾値は結果を見て決める。

## 論文での使い方(結果確認後)

- 05 実験設定に診断量の定義(A_k と permutation test)を追加
- 05 基準値感度(図1の考察)に「(a) A_hour ≈ 0.5・棄却 / (b) A_hour ≈ 0.04」の判定を追記
- 01 研究目的・貢献 C3 に「診断」を昇格(External review 3 対応)
- 02 残差診断の系譜(Breusch-Pagan / Ljung-Box)との接続1文
- 06.2 / 07 / Abstract のトーン調整
