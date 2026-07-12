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

3. **判定規則**: p 値だけでなく、効果量
   `E_k = A_k / null_relative_amplitude_p95` を併用する。
   本稿では `p < 0.05` かつ `E_k >= 3` を軸構造の候補とし、さらに成分
   ablation と補正性能が改善する場合に、その成分と corr を実用上解釈可能とする。
   `p >= 0.05` または `E_k < 3` の軸では corr を解釈しない。
   閾値付近では E を連続量として併記し、強い構造と同列には扱わない。

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
本稿では「p < 0.05 かつ A_k が帰無95%点の3倍以上」を構造候補の閾値とし、
ablation と補正性能による実用上の確認を追加する。

## FreshRetailNet 実行結果(seed 17、2026-07-13)

中心化あり・全成分の variant を主結果とし、interaction なしを感度確認として併記する。
`top10 MAE` は基準値の絶対誤差が上位10%のセルにおける補正後 MAE である。

| scenario | variant | baseline MAE | corrected MAE | top10 MAE | residual R2 | day: A / null95 | hour: A / null95 | hour p | hour corr |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `series_mean_all` | interaction なし | 0.0697 | 0.0520 | 0.2108 | 0.335 | 0.124 / 0.011 | 0.542 / 0.012 | 0.002 | 0.996 |
| `series_mean_all` | 全成分 | 0.0697 | **0.0496** | **0.1933** | **0.413** | 0.124 / 0.011 | 0.542 / 0.012 | 0.002 | 0.993 |
| `same_hour_recent_mean_d7_all` | interaction なし | 0.0580 | 0.0574 | 0.2516 | 0.019 | 0.137 / 0.014 | 0.041 / 0.012 | 0.002 | -0.929 |
| `same_hour_recent_mean_d7_all` | 全成分 | 0.0580 | **0.0558** | **0.2386** | **0.091** | 0.137 / 0.014 | 0.041 / 0.012 | 0.002 | -0.903 |
| `weekday_same_hour_mean_all` | interaction なし | 0.0420 | 0.0429 | 0.1956 | 0.001 | 0.127 / 0.014 | 0.000 / 0.013 | 1.000 | 0.000 |
| `weekday_same_hour_mean_all` | 全成分 | 0.0420 | 0.0423 | 0.1955 | 0.000 | 0.127 / 0.014 | 0.000 / 0.013 | 1.000 | 0.000 |

day 軸は3シナリオすべてで p = 0.002 であり、相対振幅は帰無95%点の
約9--11倍であった。一方、hour 軸の結果は基準値によって大きく異なる。

### 考察

1. `series_mean_all` では、hour の相対振幅は 0.542 で、帰無95%点 0.012 の
   約46倍である。全成分モデルは cell MAE を 0.0697 から 0.0496 へ28.8%、
   top10 MAE を 0.2788 から 0.1933 へ30.7%改善した。hour 成分を除いたときの
   MAE 増加量も 0.0128 と最大であり、残差補正の主因が時間帯構造であることを
   性能・振幅・ablation の3方向から確認できる。hour corr 0.993 は、この十分な
   振幅を確認した上で解釈可能である。

2. `same_hour_recent_mean_d7_all` では、hour の p 値は 0.002 だが、相対振幅は
   0.041で帰無95%点の約3.3倍にとどまり、`series_mean_all` の約13分の1である。
   大標本により微小な時間帯構造まで検出されたと考えるのが妥当で、負の hour corr
   をモデル失敗の中心的証拠とはしない。全成分モデルの MAE 改善は3.8%、top10 の
   改善は5.9%にとどまり、hour ablation の増加量 0.0006 より day の 0.0026 が
   大きいことから、残る改善余地は時間帯主効果より日効果・相互作用側にある。

3. `weekday_same_hour_mean_all` では、hour の相対振幅は実質0、p = 1.000 であり、
   基準値が時間帯構造を吸収したことが直接確認できる。補正後 MAE は 0.0420 から
   0.0423へわずかに悪化し、residual R2 も0付近である。これは「構造がない軸の
   成分は解釈せず、追加補正もしない」という診断プロトコルの負の対照になっている。
   day 構造は残るものの、現設定ではそれを有効な補正へ変換できていない。

以上より、軸の解釈可否は p 値だけで決めず、相対振幅と帰無95%点の比、成分
ablation、補正性能を合わせて判断する。特に `same_hour_recent_mean_d7_all` は
「構造なし」ではなく「統計的には検出されるが効果量が小さい」と記述する。

## 論文への反映

- 05 実験設定に診断量の定義(A_k と permutation test)を追加
- 05 基準値感度に3基準値の診断表と、p 値・効果量を併用した判定を追加
- 01 研究目的・貢献 C3 に「診断」を昇格(External review 3 対応)
- 02 残差診断の系譜(Breusch-Pagan / Ljung-Box)との接続1文
- 06.2 / 07 / Abstract のトーン調整
