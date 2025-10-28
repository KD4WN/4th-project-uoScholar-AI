# src/uosai/crawler/crawler_utils.py
"""
공통 유틸리티 모듈
- DB 연결 및 CRUD
- OpenAI API (요약, 임베딩)
- Playwright 스크린샷
- HTML 파싱 유틸
"""

import os
import re
import sys
import traceback
from contextlib import contextmanager
from typing import Optional, List
from datetime import datetime
from io import BytesIO
import base64

import requests
from bs4 import BeautifulSoup
import mysql.connector
from mysql.connector import Error as MySQLError

from openai import OpenAI
from PIL import Image

from dotenv import load_dotenv
load_dotenv()

# Playwright
try:
    from playwright.sync_api import sync_playwright
    _PLAYWRIGHT_AVAILABLE = True
except Exception:
    _PLAYWRIGHT_AVAILABLE = False

# =========================
# 환경설정
# =========================
BASE_DIR = os.path.abspath(os.getcwd())
OUT_DIR = os.path.join(BASE_DIR, "screenshot")
os.makedirs(OUT_DIR, exist_ok=True)

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "charset": os.getenv("DB_CHARSET", "utf8mb4"),
    "autocommit": os.getenv("DB_AUTOCOMMIT", "False") == "True",
    "use_pure": os.getenv("DB_USE_PURE", "True") == "True",
    "connection_timeout": int(os.getenv("DB_CONN_TIMEOUT", 20)),
    "raise_on_warnings": os.getenv("DB_WARNINGS", "True") == "True",
}

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY").strip()
client = OpenAI(api_key=OPENAI_API_KEY)
SUMMARIZE_MODEL = "gpt-4o"

PLAYWRIGHT_TIMEOUT_MS = 90000
CONNECT_TIMEOUT = 10
READ_TIMEOUT = 20

# =========================
# 유틸리티 함수
# =========================

def log(msg: str) -> None:
    """로그 출력"""
    print(f"[crawler {datetime.now():%Y-%m-%d %H:%M:%S}] {msg}")


@contextmanager
def mysql_conn():
    """MySQL 연결 컨텍스트 매니저"""
    conn = mysql.connector.connect(**DB_CONFIG)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def parse_date_yyyy_mm_dd(text: str) -> Optional[str]:
    """YYYY-MM-DD 형식 날짜 추출"""
    m = re.search(r"(\d{4}-\d{2}-\d{2})", text or "")
    return m.group(1) if m else None


def extract_main_text_from_html(html: str, max_chars: int = 12000) -> str:
    """HTML에서 본문 텍스트만 추출"""
    soup = BeautifulSoup(html, "html.parser")

    # 본문 후보 셀렉터
    candidates = [
        "div.vw-cnt", "div.vw-con", "div.vw-bd", "div.board-view",
        "article", "div#content", "div#contents", "main"
    ]
    main = None
    for sel in candidates:
        node = soup.select_one(sel)
        if node and node.get_text(strip=True):
            main = node
            break
    if main is None:
        main = soup.body or soup

    # 불필요 영역 제거
    kill_selectors = [
        ".related", ".relate", ".attach", ".file", ".files",
        ".prev", ".next", "footer", "#footer", ".sns", ".share",
        ".copyright", ".copy", ".address", ".addr"
    ]
    for ks in kill_selectors:
        for n in main.select(ks):
            n.decompose()

    text = main.get_text("\n", strip=True)

    # 푸터/주소/카피라이트 문구 제거
    drop_patterns = [
        r"서울시립대학교\s*.+?\d{2,3}-\d{3,4}-\d{4}",
        r"Copyright.+?All rights reserved\.?",
        r"이전글.*", r"다음글.*", r"관련\s?게시물.*",
    ]
    for pat in drop_patterns:
        text = re.sub(pat, "", text, flags=re.I | re.S)

    # 공백 정리
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # 과도한 길이 제한
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[... 본문 일부 생략 ...]"

    return text


# =========================
# Playwright 스크린샷
# =========================

def html_to_images_playwright(
    url: str,
    viewport_width: int = 1200,
    slice_height: int = 1920,
    timeout_ms: int = PLAYWRIGHT_TIMEOUT_MS,
    debug_full_image_path: Optional[str] = None,
    full_image_format: str = "png",
) -> List[Image.Image]:
    """페이지 전체를 스크린샷 찍고 slice_height 간격으로 분할"""
    if not _PLAYWRIGHT_AVAILABLE:
        print("❌ Playwright 미설치/임포트 실패")
        return []

    imgs: List[Image.Image] = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=[
                    "--disable-web-security",
                    "--hide-scrollbars",
                    "--disable-blink-features=AutomationControlled",
                ]
            )
            page = browser.new_page(
                viewport={"width": viewport_width, "height": slice_height},
                device_scale_factor=2.0,
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                extra_http_headers={
                    "Accept-Language": "ko-KR,ko;q=0.9",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                }
            )
            page.goto(url, wait_until="networkidle", timeout=timeout_ms)

            try:
                page.wait_for_selector("div.vw-tibx", timeout=timeout_ms)
            except Exception:
                pass

            for _ in range(6):
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(700)

            page.wait_for_load_state("domcontentloaded")
            page.wait_for_timeout(500)

            # 전체 페이지 스크린샷
            if full_image_format.lower() == "png":
                buf = page.screenshot(full_page=True, type="png")
            else:
                buf = page.screenshot(full_page=True, type="jpeg", quality=85)
            browser.close()

        # 전체 페이지 저장 (디버그)
        if debug_full_image_path:
            try:
                with open(debug_full_image_path, "wb") as f:
                    f.write(buf)
                print(f"💾 Full screenshot saved: {debug_full_image_path}")
            except Exception as e:
                print(f"⚠️ Full screenshot save failed: {e}")

        # 슬라이스 분할
        full_img = Image.open(BytesIO(buf)).convert("RGB")
        W, H = full_img.size
        y = 0
        while y < H:
            crop = full_img.crop((0, y, W, min(y + slice_height, H)))
            imgs.append(crop)
            y += slice_height

    except Exception as e:
        print(f"❌ HTML→이미지 캡처 실패: {e}")

    return imgs


# =========================
# OpenAI 요약
# =========================

def pil_to_data_url(pil_image: Image.Image, fmt="JPEG", quality=80) -> str:
    """PIL 이미지를 Data URL로 변환"""
    bio = BytesIO()
    pil_image.save(bio, format=fmt, quality=quality, optimize=True)
    b64 = base64.b64encode(bio.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{b64}"


def summarize_with_text_and_images(html_text: str, images: List[Image.Image]) -> str:
    """HTML 본문 텍스트 + 이미지로 요약 생성"""
    merge_prompt = f"""
아래는 대학 공지사항의 'HTML 본문 텍스트'입니다. 이 텍스트를 **우선 근거**로 삼고,
추가로 제공되는 '페이지 전체 캡처 이미지들'에서만 보이는 표/포스터/스캔된 문장 등 누락 정보를 **보완**하여
내용을 덧붙여주세요.

- 본문과 무관한 사이드/푸터/주소/카피라이트/관련 게시물 등은 제외하세요.
- 수치는 원문 그대로 보존
- 날짜 및 시간은 원문 그대로 보존
- 기관/부서, 장소, 전화, 메일은 원문 표기 그대로 사용(추측 금지)
- "제공된 HTML 본문 텍스트와 추가 이미지 정보를 바탕으로 한 공지사항은 다음과 같습니다:" 와 같은, 공지 사항의 내용 이외의 다른 멘트는 절대 추가하면 안됨. 정확히 공지사항 내용'만' 포함해야함.

[HTML 본문 텍스트 시작]
{html_text}
[HTML 본문 텍스트 끝]
""".strip()

    contents = [{"type": "input_text", "text": merge_prompt}]
    for img in images:
        contents.append({
            "type": "input_image",
            "image_url": pil_to_data_url(img, fmt="JPEG", quality=75)
        })
    try:
        resp = client.responses.create(
            model=SUMMARIZE_MODEL,
            input=[{"role": "user", "content": contents}],
            temperature=0.2,
        )
        return (resp.output_text or "").strip()
    except Exception as e:
        print(f"❌ 텍스트+이미지 요약 실패: {type(e).__name__}: {e}")
        traceback.print_exc(limit=2, file=sys.stdout)
        return ""


# =========================
# DB CRUD
# =========================

UPSERT_SQL = """
INSERT INTO notice
    (category, post_number, title, link, summary, embedding_vector, posted_date, department, view_count)
VALUES
    (%s, %s, %s, %s, %s, %s, %s, %s, %s) AS new
ON DUPLICATE KEY UPDATE
    title = new.title,
    link = new.link,
    summary = new.summary,
    embedding_vector = new.embedding_vector,
    posted_date = new.posted_date,
    department = new.department,
    view_count = new.view_count
"""

EXISTS_SQL = "SELECT posted_date FROM notice WHERE category=%s AND post_number=%s LIMIT 1"


def _normalize_post_number(post_number) -> int:
    """post_number를 정수로 정규화 (slug는 CRC32 해시)"""
    if isinstance(post_number, int):
        return post_number

    import zlib
    hash_val = zlib.crc32(post_number.encode('utf-8')) & 0x7fffffff
    return hash_val


def get_existing_posted_date(category: str, post_number) -> Optional[str]:
    """기존 게시물의 posted_date 조회"""
    post_num = _normalize_post_number(post_number)

    with mysql_conn() as conn:
        cur = conn.cursor()
        cur.execute(EXISTS_SQL, (category, post_num))
        row = cur.fetchone()
        cur.close()
        return row[0] if row else None


def upsert_notice(row: dict):
    """공지사항 DB 업서트"""
    with mysql_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            UPSERT_SQL,
            (
                row["category"],
                row["post_number"],
                row["title"],
                row["link"],
                row.get("summary") or None,
                row.get("embedding_vector") or None,
                row["posted_date"],
                row.get("department") or None,
                row.get("view_count") or 0,
            ),
        )
        cur.close()


def update_view_count(category: str, post_number, view_count: int):
    """조회수만 업데이트"""
    post_num = _normalize_post_number(post_number)
    with mysql_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE notice SET view_count = %s WHERE category = %s AND post_number = %s",
            (view_count, category, post_num)
        )
        cur.close()


def _ymd(x: Optional[object]) -> Optional[str]:
    """날짜 객체를 YYYY-MM-DD 문자열로 변환"""
    if x is None:
        return None
    if isinstance(x, (datetime, )):
        return x.strftime("%Y-%m-%d")
    from datetime import date
    if isinstance(x, date):
        return x.strftime("%Y-%m-%d")
    s = str(x).strip()
    return s[:10]
