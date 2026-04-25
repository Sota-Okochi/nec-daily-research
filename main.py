import os
import json
import re
from datetime import datetime
from zoneinfo import ZoneInfo
from urllib.parse import quote

import feedparser
import requests
from bs4 import BeautifulSoup
from openai import OpenAI


NOTION_VERSION = "2026-03-11"
JST = ZoneInfo("Asia/Tokyo")


OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
NOTION_TOKEN = os.environ["NOTION_TOKEN"]
NOTION_DATABASE_ID = os.environ["NOTION_DATABASE_ID"]
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.5")


client = OpenAI(api_key=OPENAI_API_KEY)


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


def build_prompt(candidates):
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

今日の日付: {today}

候補記事:
{json.dumps(compact_candidates, ensure_ascii=False, indent=2)}

出力は必ずJSONのみ。
以下の形式にしてください。

{{
  "items": [
    {{
      "title": "記事タイトル",
      "category": "防衛 / AI / 通信 / 宇宙 / セキュリティ / 決算/IR / 事業 / その他 のいずれか",
      "summary": "3〜5行程度の日本語要約",
      "key_points": ["重要ポイント1", "重要ポイント2", "重要ポイント3"],
      "insight": "研究・就活・NEC理解にどう活かせるか",
      "url": "記事URL",
      "confidence": "公式発表 / 報道 / 要確認 のいずれか",
      "published": "公開日。不明なら不明"
    }},
    {{
      "title": "記事タイトル",
      "category": "防衛 / AI / 通信 / 宇宙 / セキュリティ / 決算/IR / 事業 / その他 のいずれか",
      "summary": "3〜5行程度の日本語要約",
      "key_points": ["重要ポイント1", "重要ポイント2", "重要ポイント3"],
      "insight": "研究・就活・NEC理解にどう活かせるか",
      "url": "記事URL",
      "confidence": "公式発表 / 報道 / 要確認 のいずれか",
      "published": "公開日。不明なら不明"
    }}
  ]
}}
""".strip()


def summarize_candidates(candidates):
    if not candidates:
        raise RuntimeError("ニュース候補が取得できませんでした。")

    prompt = build_prompt(candidates)

    response = client.responses.create(
        model=OPENAI_MODEL,
        input=prompt,
    )

    text = response.output_text.strip()

    # JSON以外の文字が混じった場合に備え、最初と最後の波括弧で抽出
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise RuntimeError(f"OpenAI response is not JSON: {text}")

    data = json.loads(text[start:end])

    items = data.get("items", [])
    if len(items) < 2:
        raise RuntimeError(f"2件の記事が返ってきませんでした: {data}")

    return items[:2]


def notion_headers():
    return {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }


def get_data_source_id_from_database(database_id: str) -> str:
    url = f"https://api.notion.com/v1/databases/{database_id}"
    res = requests.get(url, headers=notion_headers(), timeout=10)
    res.raise_for_status()

    data = res.json()
    data_sources = data.get("data_sources", [])

    if not data_sources:
        raise RuntimeError("Notion databaseにdata_sourcesが見つかりません。")

    return data_sources[0]["id"]


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


def create_notion_page(data_source_id: str, item: dict):
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
                "select": {"name": item.get("category", "その他")}
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
                "select": {"name": item.get("confidence", "要確認")}
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
        ]
    }

    res = requests.post(
        "https://api.notion.com/v1/pages",
        headers=notion_headers(),
        json=payload,
        timeout=15,
    )

    if res.status_code >= 400:
        raise RuntimeError(f"Notion API error: {res.status_code} {res.text}")

    return res.json()


def main():
    candidates = fetch_rss_candidates()
    items = summarize_candidates(candidates)
    data_source_id = get_data_source_id_from_database(NOTION_DATABASE_ID)

    for item in items:
        create_notion_page(data_source_id, item)

    print(f"Created {len(items)} Notion pages.")


if __name__ == "__main__":
    main()