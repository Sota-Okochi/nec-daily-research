import os
import json
import re
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo
from urllib.parse import quote

import feedparser
import requests
from bs4 import BeautifulSoup
from openai import OpenAI


NOTION_VERSION = "2025-09-03"
DEFAULT_OPENAI_MODEL = "gpt-5.5"
JST = ZoneInfo("Asia/Tokyo")

REQUIRED_NOTION_PROPERTIES = {
    "タイトル": "title",
    "日付": "date",
    "企業": "rich_text",
    "カテゴリ": "multi_select",
    "要約": "rich_text",
    "重要ポイント": "rich_text",
    "自分への示唆": "rich_text",
    "URL": "url",
    "信頼度": "multi_select",
}


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    notion_token: str
    notion_database_id: str
    openai_model: str


def read_env(name: str, *, default: str | None = None, required: bool = False) -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value
    if default is not None:
        return default
    if required:
        raise RuntimeError(
            f"Environment variable '{name}' is required but was not set. "
            "GitHub Actions secrets/variables を確認してください。"
        )
    return ""


def load_settings() -> Settings:
    return Settings(
        openai_api_key=read_env("OPENAI_API_KEY", required=True),
        notion_token=read_env("NOTION_TOKEN", required=True),
        notion_database_id=read_env("NOTION_DATABASE_ID", required=True),
        openai_model=read_env("OPENAI_MODEL", default=DEFAULT_OPENAI_MODEL),
    )


def google_news_rss_url(query: str) -> str:
    return (
        "https://news.google.com/rss/search?"
        f"q={quote(query)}"
        "&hl=ja&gl=JP&ceid=JP:ja"
    )


def fetch_rss_candidates():
    queries = [
        "NEC 日本電気 when:7d",
        "NEC 日本電気 防衛 OR サイバー OR AI OR 通信 OR 宇宙 OR 6G when:7d",
        "NEC site:jpn.nec.com/press when:30d",
    ]

    candidates = []
    seen = set()

    for query in queries:
        feed_url = google_news_rss_url(query)
        feed = feedparser.parse(feed_url)

        for entry in feed.entries[:10]:
            title = getattr(entry, "title", "").strip()
            link = getattr(entry, "link", "").strip()
            summary = BeautifulSoup(getattr(entry, "summary", ""), "html.parser").get_text(" ", strip=True)
            published = getattr(entry, "published", "")

            if not title or not link:
                continue

            key = title.lower()
            if key in seen:
                continue
            seen.add(key)

            candidates.append({
                "title": title,
                "url": link,
                "summary": summary,
                "published": published,
                "query": query,
            })

    return candidates[:20]


def fetch_page_text(url: str) -> str:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; NEC-Daily-Research-Bot/1.0)"
        }
        res = requests.get(url, timeout=10, headers=headers)
        res.raise_for_status()

        soup = BeautifulSoup(res.text, "html.parser")

        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = soup.get_text("\n", strip=True)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text[:5000]

    except Exception:
        return ""


def build_prompt(candidates, category_options, confidence_options):
    today = datetime.now(JST).strftime("%Y-%m-%d")

    compact_candidates = []
    for i, c in enumerate(candidates, start=1):
        page_text = fetch_page_text(c["url"])
        compact_candidates.append({
            "id": i,
            "title": c["title"],
            "url": c["url"],
            "published": c["published"],
            "rss_summary": c["summary"],
            "page_text_excerpt": page_text[:2500],
        })

    return f"""
あなたは、NECに入社予定の大学院生向けに、毎朝読む企業・業界リサーチを作るアシスタントです。

目的:
- NECに関する幅広い情報を毎日2件だけ選ぶ
- 防衛、AI、通信、宇宙、サイバー、決算/IR、社会インフラ、官公庁案件などを優先
- ただし、与えられた候補に書かれていない事実は追加しない
- 不明な点は「不明」と書く
- 推測しない
- カテゴリは次の候補から必ず1つ選ぶ: {", ".join(category_options)}
- 信頼度は次の候補から必ず1つ選ぶ: {", ".join(confidence_options)}

今日の日付: {today}

候補記事:
{json.dumps(compact_candidates, ensure_ascii=False, indent=2)}

出力は必ずJSONのみ。
- 必ず2件選ぶ
- key_points は必ず3件にする
""".strip()


def build_response_schema(category_options, confidence_options):
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["items"],
        "properties": {
            "items": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "title",
                        "category",
                        "summary",
                        "key_points",
                        "insight",
                        "url",
                        "confidence",
                        "published",
                    ],
                    "properties": {
                        "title": {"type": "string", "minLength": 1},
                        "category": {
                            "type": "string",
                            "enum": category_options,
                        },
                        "summary": {"type": "string", "minLength": 1},
                        "key_points": {
                            "type": "array",
                            "minItems": 3,
                            "maxItems": 3,
                            "items": {
                                "type": "string",
                                "minLength": 1,
                            },
                        },
                        "insight": {"type": "string", "minLength": 1},
                        "url": {"type": "string", "minLength": 1},
                        "confidence": {
                            "type": "string",
                            "enum": confidence_options,
                        },
                        "published": {"type": "string", "minLength": 1},
                    },
                },
            }
        },
    }


def summarize_candidates(candidates, client, model, category_options, confidence_options):
    if not candidates:
        raise RuntimeError("ニュース候補が取得できませんでした。")

    prompt = build_prompt(candidates, category_options, confidence_options)

    response = client.responses.create(
        model=model,
        input=prompt,
        max_output_tokens=2000,
        text={
            "format": {
                "type": "json_schema",
                "name": "nec_daily_research",
                "schema": build_response_schema(category_options, confidence_options),
                "strict": True,
            }
        },
    )

    text = response.output_text.strip()
    data = json.loads(text)

    items = data.get("items", [])
    if len(items) < 2:
        raise RuntimeError(f"2件の記事が返ってきませんでした: {data}")

    return items[:2]


def notion_headers(notion_token: str):
    return {
        "Authorization": f"Bearer {notion_token}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }


def format_api_error(response: requests.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        body = response.text
    return f"{response.status_code} {body}"


def get_database_data_sources(database_id: str, notion_token: str):
    url = f"https://api.notion.com/v1/databases/{database_id}"
    res = requests.get(url, headers=notion_headers(notion_token), timeout=10)
    if res.ok:
        data_sources = res.json().get("data_sources", [])
        if not data_sources:
            raise RuntimeError("Notion databaseにdata_sourcesが見つかりません。")
        return data_sources
    return None, res


def get_data_source_schema(data_source_id: str, notion_token: str) -> dict:
    url = f"https://api.notion.com/v1/data_sources/{data_source_id}"
    res = requests.get(url, headers=notion_headers(notion_token), timeout=10)
    if not res.ok:
        raise RuntimeError(f"Notion data source取得失敗: {format_api_error(res)}")
    return res.json()


def resolve_data_source_schema(notion_database_id: str, notion_token: str) -> tuple[str, dict]:
    database_result = get_database_data_sources(notion_database_id, notion_token)
    if isinstance(database_result, tuple):
        _, db_res = database_result
        if db_res.status_code not in {400, 404}:
            raise RuntimeError(f"Notion database取得失敗: {format_api_error(db_res)}")
    else:
        data_sources = database_result
        data_source_id = data_sources[0]["id"]
        return data_source_id, get_data_source_schema(data_source_id, notion_token)

    try:
        data_source = get_data_source_schema(notion_database_id, notion_token)
        return notion_database_id, data_source
    except RuntimeError as exc:
        raise RuntimeError(
            "NOTION_DATABASE_ID は database ID または data source ID を指定してください。"
            "ID が誤っているか、対象 database/data source が Notion integration に共有されていません。"
        ) from exc


def get_property_schema(data_source: dict, property_name: str, expected_type: str) -> dict:
    properties = data_source.get("properties", {})
    prop = properties.get(property_name)
    if prop is None:
        raise RuntimeError(
            f"Notion data source に必須プロパティ '{property_name}' がありません。"
        )

    actual_type = prop.get("type")
    if actual_type != expected_type:
        raise RuntimeError(
            f"Notion プロパティ '{property_name}' の型が想定と異なります。"
            f" expected={expected_type}, actual={actual_type}"
        )
    return prop


def validate_required_properties(data_source: dict):
    for property_name, expected_type in REQUIRED_NOTION_PROPERTIES.items():
        get_property_schema(data_source, property_name, expected_type)


def get_multi_select_option_names(data_source: dict, property_name: str) -> list[str]:
    prop = get_property_schema(data_source, property_name, "multi_select")
    options = [option["name"] for option in prop["multi_select"].get("options", [])]
    if not options:
        raise RuntimeError(
            f"Notion multi_select プロパティ '{property_name}' に選択肢がありません。"
        )
    return options


def rich_text(content: str):
    return [{"type": "text", "text": {"content": content[:2000]}}]


def paragraph(content: str):
    return {
        "object": "block",
        "type": "paragraph",
        "paragraph": {
            "rich_text": rich_text(content)
        }
    }


def heading(content: str):
    return {
        "object": "block",
        "type": "heading_2",
        "heading_2": {
            "rich_text": rich_text(content)
        }
    }


def bulleted_item(content: str):
    return {
        "object": "block",
        "type": "bulleted_list_item",
        "bulleted_list_item": {
            "rich_text": rich_text(content)
        }
    }


def create_notion_page(data_source_id: str, notion_token: str, item: dict):
    today = datetime.now(JST).date().isoformat()

    key_points = item.get("key_points", [])
    key_points_text = "\n".join([f"- {p}" for p in key_points])

    payload = {
        "parent": {
            "type": "data_source_id",
            "data_source_id": data_source_id,
        },
        "properties": {
            "タイトル": {
                "title": rich_text(item.get("title", "NEC Daily Research"))
            },
            "日付": {
                "date": {"start": today}
            },
            "企業": {
                "rich_text": rich_text("NEC")
            },
            "カテゴリ": {
                "multi_select": [{"name": item.get("category", "その他")}]
            },
            "要約": {
                "rich_text": rich_text(item.get("summary", ""))
            },
            "重要ポイント": {
                "rich_text": rich_text(key_points_text)
            },
            "自分への示唆": {
                "rich_text": rich_text(item.get("insight", ""))
            },
            "URL": {
                "url": item.get("url", None)
            },
            "信頼度": {
                "multi_select": [{"name": item.get("confidence", "要確認")}]
            },
        },
        "children": [
            heading("要約"),
            paragraph(item.get("summary", "")),
            heading("重要ポイント"),
            *[bulleted_item(point) for point in key_points],
            heading("自分への示唆"),
            paragraph(item.get("insight", "")),
            heading("情報源"),
            paragraph(f"公開日: {item.get('published', '不明')}"),
            paragraph(f"URL: {item.get('url', '')}"),
        ],
    }

    res = requests.post(
        "https://api.notion.com/v1/pages",
        headers=notion_headers(notion_token),
        json=payload,
        timeout=15,
    )

    if res.status_code >= 400:
        raise RuntimeError(f"Notion API error: {format_api_error(res)}")

    return res.json()


def main():
    settings = load_settings()
    client = OpenAI(api_key=settings.openai_api_key)
    candidates = fetch_rss_candidates()
    data_source_id, data_source = resolve_data_source_schema(
        settings.notion_database_id,
        settings.notion_token,
    )
    validate_required_properties(data_source)
    category_options = get_multi_select_option_names(data_source, "カテゴリ")
    confidence_options = get_multi_select_option_names(data_source, "信頼度")
    items = summarize_candidates(
        candidates,
        client,
        settings.openai_model,
        category_options,
        confidence_options,
    )

    for item in items:
        create_notion_page(data_source_id, settings.notion_token, item)

    print(f"Created {len(items)} Notion pages.")


if __name__ == "__main__":
    main()
