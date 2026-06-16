# APIEMS 2026 Preparation

更新日: 2026-06-16

## 投稿方針

優先案は `Full paper`。理由は、Best Paper Award と国際誌推薦の対象になるため。ただし、会議 proceedings / Springer / EI indexing の扱いに公式ページ間の差があるため、実際の出版義務や著作権扱いは投稿前に確認する。

Abstract-only を選ぶ場合:

- 研究の完成度が不足していても発表機会を確保しやすい。
- ただし、国際誌推薦の対象外。
- 後続のジャーナル投稿に向けた発表・フィードバック取得の位置づけになる。

Full paper を選ぶ場合:

- 2026-07-31 までに本文・図表・実験結果・関連研究を最低限そろえる。
- Full paper template を使って整形する。
- 採択後、2026-09-30 までに早期登録する。

## 原稿作成

最低限そろえるもの:

- Title / Abstract / Keywords
- Introduction: なぜ売上そのものではなく残差を見るのか
- Problem formulation: `r = y - b`、4 成分分解、平均ゼロ制約
- Method: Encoder / Decoder / loss / calibration の説明
- Experiments: synthetic、FreshRetailNet、statistical validation
- Discussion: どの baseline で効き、どの baseline では残差構造が薄いか
- Limitations: 基準値選択依存、実データ interaction の弱さ、データセット一般化
- Figures / Tables: 既存の `2-Exp-23` 以降の論文用表を中心に選定

このリポジトリで対応する既存資料:

| 用途 | 参照 |
| --- | --- |
| 論文化スケジュール | `../../proposal/schedule.md` |
| 研究の方向性 | `../../proposal/paper_direction.md` |
| 定式化 | `../../proposal/formulation.md` |
| 実験結果集約 | `../../2-Exp-23_paper_tables.md` |
| FreshRetailNet 頑健性 | `../../2-Exp-24_freshretailnet_scale_sensitivity.md`, `../../2-Exp-25_freshretailnet_block_robustness.md` |

## 登録

登録費は公式 Registration ページでは USD 建て。

| 区分 | Early Bird: 2026-09-30 まで | Regular: 2026-10-31 まで | On-site |
| --- | ---: | ---: | ---: |
| Presenter | $600 | $650 | $700 |
| Presenter - Student | $350 | $400 | $450 |
| Retired Member | $350 | $400 | $450 |
| Participant / Organized Session Chair | $400 | $400 | $450 |
| Additional Fee for Each Extra Paper | $200 | $200 | $200 |

KIIE member は KIIE webpage、Non-KIIE member は EasyChair registration system で支払いと記載されている。

キャンセルは `admin@kiie.org` へ書面連絡。2026-10-31 までのキャンセルは 10% の cancellation fee を除いて返金、それ以降は返金不可。

## 渡航・宿泊

移動:

- 最寄り空港は Gimhae International Airport (PUS)。
- PUS から BPEX は公共交通で約60分、タクシーで約30〜40分、15,000〜20,000 KRW 程度。
- Incheon International Airport (ICN) からは AREX で Seoul Station、KTX で Busan Station、Busan Station から徒歩約10分が推奨ルート。
- Google Maps は韓国内の徒歩経路に弱い場合があるため、Naver Map または KakaoMap を準備する。

公式 Accommodation ページの推奨ホテル:

| ホテル | 目安距離 | メモ |
| --- | ---: | --- |
| Asti Hotel Busan | 0.9 km | Promotion code: `KIIE` |
| Ramada Encore Busan Station | 1.1 km | Promotion code: `KII26` |
| Commodore Hotel | 2.0 km | 代替候補 |
| Crown Harbor Hotel Busan | 2.0 km | 代替候補 |
| La Valse Hotel | 4.4 km | 代替候補 |
| Lotte Hotel Busan | 5.7 km | 代替候補 |
| ibis Ambassador Busan City Centre | 5.8 km | 代替候補 |

## 入国・招待状

APIEMS 2026 公式 Visa Information では、ビザ・入国要件は国籍、旅券種別、居住国により異なるため、各自で韓国大使館・領事館に確認するよう案内されている。

ビザ申請用の invitation letter が必要な場合は、登録と登録費支払い後に `admin@kiie.org` へ依頼する。件名は次の指定。

```text
Visa Invitation Letter Request - APIEMS 2026
```

依頼時に必要な情報:

- Full name as shown on passport
- Nationality
- Date of birth
- Gender
- Affiliation / organization
- Position / title
- Email address and contact number
- Passport number
- Expected dates of arrival in and departure from Korea
- Registration number or proof of registration
- Paper ID and title, if applicable
