# nec-daily-research

GitHub Actions で毎日 NEC 関連ニュースを収集し、OpenAI で 2 件に絞って Notion に保存します。

## Required GitHub Configuration

Repository secrets:

- `OPENAI_API_KEY`
- `NOTION_TOKEN`
- `NOTION_DATABASE_ID`

Optional repository variable:

- `OPENAI_MODEL`
  - 未設定時は `gpt-5.5` を使います
  - secret ではなく Actions variable で十分です

`NOTION_DATABASE_ID` には database ID または data source ID を指定できます。

## Required Notion Properties

対象の Notion data source には次のプロパティが必要です。

- `タイトル`: `title`
- `日付`: `date`
- `企業`: `rich_text`
- `カテゴリ`: `select`
- `要約`: `rich_text`
- `重要ポイント`: `rich_text`
- `自分への示唆`: `rich_text`
- `URL`: `url`
- `信頼度`: `select`

`カテゴリ` と `信頼度` の select options は、Actions 実行時に data source から読み取り、その候補だけを OpenAI に渡します。
