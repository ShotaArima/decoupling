# APIEMS 2026 Submission

更新日: 2026-06-16

## 投稿方式

APIEMS 2026 には次の 2 種類の投稿方式がある。

| 方式 | 内容 | この研究での向き |
| --- | --- | --- |
| Abstract-only | 発表審査のみ。会議 proceedings には掲載されない | 研究が論文化途中で、発表・フィードバック取得を優先する場合に向く |
| Full paper | 発表審査に加え、Best Paper Award と国際誌推薦の対象になる | 6月末時点で論文骨格が固まるなら第一候補 |

Submission ページでは、Full paper は会議 proceedings として別途出版されず、SCIE journal への推薦候補として評価される旨が書かれている。一方、CFP ページには Full papers が proceedings publication 対象のように読める記載がある。投稿前に `Submission` ページ、CFP PDF、EasyChair の投稿画面を再確認する。

## テンプレート

2026-06-16 時点で公式サイト上に見つかるテンプレートは `APIEMS FullPaperTemplate.docx` のみ。TeX / LaTeX テンプレートは公式 CFP ページ、Submission ページ、Web 検索では確認できなかった。

`APIEMS FullPaperTemplate.docx` は、本文中の表記から `APIEMS-2016` 用 Word テンプレートを流用したものに見える。テンプレートの指示は次の通り。

| 項目 | 指示 |
| --- | --- |
| 用紙 | A4, single-sided |
| ページ数 | 8 pages 以内 |
| 言語 | English |
| ページ番号 | Do not number the pages |
| Title | 20pt Times New Roman |
| Author details | 10pt Times New Roman |
| Section title | 11pt Times New Roman, bold |
| Body | 10pt Times New Roman |
| Abstract | 200 words 未満 |
| Keywords | maximum five keywords |
| 本文レイアウト | main part は 2 columns、column spacing 0.75 cm |
| 見出し番号 | decimal system で連番。第3レベルまで推奨 |
| 図表 | Figure caption は下、Table caption は上。可能ならページ上部 |
| 引用 | 著者年方式。例: `(Gusfield, 1997)`、`Gusfield (1997)` |
| 参考文献 | author name のアルファベット順。同一著者は発行年順 |

## テンプレート上の見出し例

テンプレートに含まれる見出し例:

- `1. INTRODUCTION`
- `2. HEADING`
- `2.1 Second-Level Heading`
- `2.1.1 Third-Level Heading`
- `3. MATHEMATICS`
- `4. THEOREMS AND LEMMATA`
- `5. FIGURES AND TABLES`
- `ACKNOWLEDGMENTS`
- `APPENDICES`
- `CITATIONS`
- `REFERENCES`

これは投稿論文の必須章立てというより、書式説明のためのサンプル構成である。

## この研究の章立て案

この研究を Full paper にする場合の章立て案:

| 章 | 内容 |
| --- | --- |
| 1. INTRODUCTION | 小売需要予測で残差を分解する動機、基準値からのズレを見る意義、貢献 |
| 2. RELATED WORK | 時系列予測、残差学習、解釈可能な需要分析、multi-granularity decomposition |
| 3. PROBLEM FORMULATION | `r = y - b`、day/hour/interaction 成分、平均ゼロ制約 |
| 4. PROPOSED METHOD | Encoder/Decoder、成分出力、loss、calibration |
| 5. EXPERIMENTS | synthetic と FreshRetailNet、baseline、評価指標、設定 |
| 6. RESULTS AND DISCUSSION | 成分回復、予測補正、baseline ごとの差、限界 |
| 7. CONCLUSION | 何が言えたか、適用条件、今後の課題 |
| ACKNOWLEDGMENTS | 必要なら研究費・共同研究先 |
| REFERENCES | 著者年方式 |

8 pages 制限を考えると、`RELATED WORK` は短くし、Method と Experiments を圧縮する。詳細な追加実験、ハイパーパラメータ、補助図は口頭発表資料または後続 journal version に回すのが現実的。

## 投稿作業

- EasyChair アカウントを確認する。
- 共著者の氏名、所属、メールアドレス、著者順を確定する。
- Abstract-only / Full paper のどちらで出すか決める。
- Full paper の場合は公式テンプレートをダウンロードして使う。
- 研究分野カテゴリを scope に合わせて選ぶ。
- 投稿後、Paper ID を記録する。
