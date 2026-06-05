# Theory: 残差直交分解と収束保証の整理

## 1. 基本設定

売上を $y_{i,d,h}$、基準値を $b_{i,d,h}$ とする。

残差は次で定義する。

$$
r_{i,d,h}=y_{i,d,h}-b_{i,d,h}
$$

本研究では、残差が次の構造を持つと仮定する。

$$
r_{i,d,h}
=
g_i
+ a_{i,d}
+ c_{i,h}
+ u_{i,d,h}
+ \varepsilon_{i,d,h}
$$

ここで、

- $g_i$: 系列固有成分
- $a_{i,d}$: day 成分
- $c_{i,h}$: hour 成分
- $u_{i,d,h}$: day-hour interaction 成分
- $\varepsilon_{i,d,h}$: ノイズ

である。

## 2. なぜ制約が必要か

この分解は、そのままだと一意ではない。

例えば任意の定数 $\alpha$ に対して、

$$
g_i' = g_i + \alpha
$$

$$
a_{i,d}' = a_{i,d} - \alpha
$$

と置いても、

$$
g_i' + a_{i,d}' = g_i + a_{i,d}
$$

なので、和は変わらない。

したがって、各成分の担当範囲を固定する制約が必要である。

## 3. 直交制約

以下の制約を置く。

### 3.1 day 成分

day 成分は日方向に変動するが、平均は 0 とする。

$$
\frac{1}{D}\sum_{d=1}^{D} a_{i,d}=0
$$

### 3.2 hour 成分

hour 成分は時間方向に変動するが、平均は 0 とする。

$$
\frac{1}{H}\sum_{h=1}^{H} c_{i,h}=0
$$

### 3.3 interaction 成分

interaction 成分は day の主効果も hour の主効果も持たない。

$$
\frac{1}{D}\sum_{d=1}^{D} u_{i,d,h}=0
\quad \forall h
$$

$$
\frac{1}{H}\sum_{h=1}^{H} u_{i,d,h}=0
\quad \forall d
$$

この制約により、$u$ は「day だけ」「hour だけ」の効果を含めない。

## 4. 命題 1: ANOVA 分解の一意性

### 命題

観測された残差行列 $R_i \in \mathbb{R}^{D\times H}$ に対して、以下の制約を満たす分解

$$
R_i = g_i \mathbf{1}\mathbf{1}^\top
+ a_i \mathbf{1}^\top
+ \mathbf{1} c_i^\top
+ U_i
$$

は一意に定まる。

制約は、

$$
\sum_d a_{i,d}=0
$$

$$
\sum_h c_{i,h}=0
$$

$$
\sum_d U_{i,d,h}=0
,\quad
\sum_h U_{i,d,h}=0
$$

である。

### 構成

各成分は次で与えられる。

$$
g_i = \bar r_i
$$

$$
a_{i,d} = \bar r_{i,d,\cdot}-\bar r_i
$$

$$
c_{i,h} = \bar r_{i,\cdot,h}-\bar r_i
$$

$$
u_{i,d,h}
=
r_{i,d,h}
-g_i
-a_{i,d}
-c_{i,h}
$$

ここで、

$$
\bar r_i
=
\frac{1}{DH}
\sum_{d,h}r_{i,d,h}
$$

$$
\bar r_{i,d,\cdot}
=
\frac{1}{H}
\sum_h r_{i,d,h}
$$

$$
\bar r_{i,\cdot,h}
=
\frac{1}{D}
\sum_d r_{i,d,h}
$$

である。

### 証明スケッチ

この定義で各制約が満たされることは、和を取れば確認できる。

また、同じ制約を満たす別の分解があると仮定する。
差分を取ると、ゼロ行列を

$$
0=\Delta g+\Delta a_d+\Delta c_h+\Delta u_{d,h}
$$

と分解することになる。

両辺を $d,h$ 方向に平均すると $\Delta g=0$ が分かる。
次に hour 平均を取ると $\Delta a_d=0$、day 平均を取ると $\Delta c_h=0$ が分かる。
最後に $\Delta u_{d,h}=0$ となる。

したがって分解は一意である。

## 5. 提案モデル

提案モデルは、残差予測を成分別に出す。

$$
\hat r_{i,d,h}
=
\hat g_i
+ \hat a_{i,d}
+ \hat c_{i,h}
+ \hat u_{i,d,h}
$$

各成分は、対応する latent から出力する。

$$
\hat g_i=f_g(z^g_i)
$$

$$
\hat a_{i,d}=f_a(z^a_{i,d})
$$

$$
\hat c_{i,h}=f_c(z^c_{i,h})
$$

$$
\hat u_{i,d,h}=f_u(z^u_{i,d,h})
$$

ただし、出力には制約を入れる。

$$
\sum_d \hat a_{i,d}=0
$$

$$
\sum_h \hat c_{i,h}=0
$$

$$
\sum_d \hat u_{i,d,h}=0
,\quad
\sum_h \hat u_{i,d,h}=0
$$

## 6. 損失関数

観測マスクを $m_{i,d,h}$ とする。

基本の再構成損失は、

$$
\mathcal L_{\mathrm{rec}}
=
\frac{
\sum_{i,d,h}
m_{i,d,h}
\left(r_{i,d,h}-\hat r_{i,d,h}\right)^2
}{
\sum_{i,d,h}m_{i,d,h}
}
$$

とする。

制約損失は、

$$
\mathcal L_{\mathrm{day}}
=
\sum_i
\left(
\frac{1}{D}
\sum_d \hat a_{i,d}
\right)^2
$$

$$
\mathcal L_{\mathrm{hour}}
=
\sum_i
\left(
\frac{1}{H}
\sum_h \hat c_{i,h}
\right)^2
$$

$$
\mathcal L_{\mathrm{int}}
=
\sum_{i,h}
\left(
\frac{1}{D}
\sum_d \hat u_{i,d,h}
\right)^2
+
\sum_{i,d}
\left(
\frac{1}{H}
\sum_h \hat u_{i,d,h}
\right)^2
$$

とする。

全体の目的関数は、

$$
\mathcal L
=
\mathcal L_{\mathrm{rec}}
+\lambda_d\mathcal L_{\mathrm{day}}
+\lambda_h\mathcal L_{\mathrm{hour}}
+\lambda_u\mathcal L_{\mathrm{int}}
+\lambda_{\mathrm{leak}}\mathcal L_{\mathrm{leak}}
+\lambda_{\mathrm{swap}}\mathcal L_{\mathrm{swap}}
$$

である。

## 7. 命題 2: 出力成分の識別可能性

### 命題

次を仮定する。

1. 真の残差が

$$
r_{i,d,h}=g_i+a_{i,d}+c_{i,h}+u_{i,d,h}+\varepsilon_{i,d,h}
$$

で生成される。

2. 真の $g,a,c,u$ が直交制約を満たす。

3. モデルの各出力クラスが、真の $g,a,c,u$ を表現できる。

4. 学習により $\mathcal L_{\mathrm{rec}}$ と制約損失が十分小さくなる。

このとき、学習された出力成分

$$
\hat g,\hat a,\hat c,\hat u
$$

は、ノイズと近似誤差を除いて、真の ANOVA 成分

$$
g,a,c,u
$$

に近づく。

### 証明スケッチ

再構成誤差が十分小さいなら、

$$
\hat r \approx r
$$

である。

さらに、$\hat g,\hat a,\hat c,\hat u$ が直交制約を満たすなら、$\hat r$ の分解は命題 1 の ANOVA 分解に対応する。

命題 1 により、制約を満たす分解は一意である。

したがって、$\hat r$ が $r$ に近いほど、その成分分解も真の成分分解に近づく。

この主張は latent の一意性ではなく、**出力成分の一意性**に関するものである。

## 8. 収束について何を証明できるか

ここでいう「収束」は 3 つに分ける必要がある。

### 8.1 分解の収束

サンプル数が増え、残差の平均が安定すれば、経験的な ANOVA 分解は母集団の ANOVA 分解に近づく。

これは大数の法則に基づく。

主張できることは、

$$
\hat g^{\mathrm{ANOVA}}\to g^{\star}
$$

$$
\hat a^{\mathrm{ANOVA}}\to a^{\star}
$$

$$
\hat c^{\mathrm{ANOVA}}\to c^{\star}
$$

$$
\hat u^{\mathrm{ANOVA}}\to u^{\star}
$$

である。

### 8.2 推定量の収束

モデルクラスが十分に大きく、経験リスク最小化が適切に行われるなら、経験リスクの最小解は母集団リスクの最小解に近づく。

これは統計的一貫性の主張である。

ただし、ニューラルネットでは一般に強い大域最適性は保証しにくい。
そのため論文では、以下のように限定して述べるのが現実的である。

> 提案する出力制約により、最適化された解が従うべき成分空間を制限できる。
> その制約空間内で再構成誤差が小さい場合、出力成分は ANOVA 的な意味を持つ。

### 8.3 最適化の収束

SGD や Adam が大域最適解に必ず到達することは、一般には保証できない。

ただし、滑らかな損失、有限の勾配分散、適切な学習率などの仮定の下で、

$$
\mathbb{E}\|\nabla \mathcal L(\theta)\|^2 \to 0
$$

のような stationary point への収束は既存理論に沿って主張できる。

したがって、本研究で強く主張すべきなのは、

```text
最適化が必ず真の解に到達する
```

ではなく、

```text
到達した解が制約を満たす限り、出力成分には ANOVA 的な意味がある
```

である。

## 9. 論文で使う保証のレベル

本研究で現実的に主張できる保証は次の 3 段階である。

| レベル | 内容 | 強さ |
|---|---|---|
| Level 1 | 直交制約の下で ANOVA 分解は一意 | 強い |
| Level 2 | 提案モデルの出力制約は ANOVA 成分空間に対応する | 中程度 |
| Level 3 | 学習が十分進めば、出力成分は真の成分に近づく | 仮定付き |

この 3 段階なら、修士論文として過剰な主張を避けながら、数理的な論拠を作れる。

## 10. 実験で確認すべきこと

理論を実験で支えるには、以下を確認する。

1. 合成データで、真の $g,a,c,u$ を回復できるか。
2. 制約なしモデルより、制約ありモデルの成分回復誤差が小さいか。
3. interaction 成分が day/hour 主効果を持たないか。
4. FreshRetailNet で、成分 ablation が subset の意味と一致するか。
5. leakage probe が下がるか。
6. `b+\hat r` が全体または外れケースで改善するか。

## 11. まとめ

完全な latent 識別の保証は難しい。

しかし、出力空間を直交分解し、成分ごとに制約を入れることで、少なくとも出力された残差成分の意味は保証しやすくなる。

この方向は、元論文の「暗黙的な分離」の弱点を補う自然な拡張である。

