# src/uosai/crawler/notice_crawler.py

import os
import time
import re
from contextlib import contextmanager
from typing import Optional, Dict, List

import base64
from io import BytesIO
from collections import OrderedDict
from urllib.parse import urlencode

import requests
from bs4 import BeautifulSoup
import mysql.connector
from mysql.connector import Error as MySQLError

from openai import OpenAI
from PIL import Image  # 이미지 처리
from datetime import date, datetime
import sys, traceback

from dotenv import load_dotenv
load_dotenv()

# Playwright
try:
    from playwright.sync_api import sync_playwright
    _PLAYWRIGHT_AVAILABLE = True
except Exception:
    _PLAYWRIGHT_AVAILABLE = False

# =========================
# 0) 환경설정
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

#################################################################################
# 카테고리 ↔ list_id 매핑 (포털 공통)
CATEGORIES: Dict[str, str] = {
    "COLLEGE_ENGINEERING": "20013DA1",
    "COLLEGE_HUMANITIES": "human01",
    "COLLEGE_SOCIAL_SCIENCES": "econo01",
    "COLLEGE_URBAN_SCIENCE": "urbansciences01",
    "COLLEGE_ARTS_SPORTS": "artandsport01",
    "COLLEGE_BUSINESS": "20008N2",
    "COLLEGE_NATURAL_SCIENCES": "scien01",
    "COLLEGE_LIBERAL_CONVERGENCE": "clacds01",
    "GENERAL": "FA1",
    "ACADEMIC": "FA2",
}

CRAWL_VIEW_URL = "https://www.uos.ac.kr/korNotice/view.do?identified=anonymous&"
CRAWL_LIST_URL = "https://www.uos.ac.kr/korNotice/list.do?identified=anonymous&"
SAVE_VIEW_URL = "https://www.uos.ac.kr/korNotice/view.do"

#################################################################################
# 학과별 독립 URL 설정 (각 학과마다 다른 base_url과 파싱 로직)
#################################################################################
DEPT_CONFIGS = {
    "DEPT_CHEMICAL_ENGINEERING": {
        "category": "COLLEGE_ENGINEERING",
        "department": "화학공학과",
        "list_url": "https://cheme.uos.ac.kr/bbs/board.php?bo_table=notice",
        "id_param": "wr_id",        # URL 파라미터명
        "list_params": {"bo_table": "notice"},  # 목록 조회용 기본 파라미터
        "url_type": "query",  # query 파라미터 방식
        "selectors": {
            "title": ["#bo_v_title .bo_v_tit", "#bo_v_title"],
            "content": ["#bo_v_atc", ".board_view", ".view_content", "#bo_v"],
            "date_info": ["#bo_v_info", ".bo_v_info", ".view_info", ".board_view .info"],
            "view_count": "strong > i.fa-eye",  # 조회수 아이콘
        }
    },
    "DEPT_LIFE_SCIENCE": {
        "category": "COLLEGE_NATURAL_SCIENCES",
        "department": "생명과학과",
        "list_url": "https://lifesci.uos.ac.kr/community/notice",
        "id_param": "bbsidx",
        "list_params": {},  # 기본 파라미터 없음
        "url_type": "query",  # query 파라미터 방식
        "selectors": {
            "title": ["h1.bbstitle"],
            "content": [],  # extract_main_text_from_html 사용
            "date_info": ["div.writer"],
            "view_count": "div.writer",  # 텍스트에서 파싱
        }
    },
    "DEPT_ECONOMICS": {
        "category": "COLLEGE_SOCIAL_SCIENCES",
        "department": "경제학부",
        "list_url": "https://econ.uos.ac.kr/notices/undergraduate",
        "id_param": "post_id",  # 더미 (실제로는 경로에서 추출)
        "list_params": {},
        "url_type": "path",  # 경로(path) 방식 (예: /notices/19184)
        "detail_url_template": "https://econ.uos.ac.kr/notices/{post_id}",  # 상세 URL 템플릿
        "selectors": {
            "title": ["h2.uos-post-header__title", ".uos-post-header__title"],
            "content": ["div.uos-post__content", ".uos-post__content"],
            "date_info": ["span.uos-meta-section__date-value", ".uos-meta-section__date-value"],
            "view_count": None,  # 조회수 없음
        }
    },
    "DEPT_ARCHITECTURE": {
        "category": "COLLEGE_URBAN_SCIENCE",
        "department": "건축학부",
        "list_url": "https://uosarch.ac.kr/board/notice/",
        "id_param": "post_slug",  # WordPress slug 방식
        "list_params": {},
        "url_type": "slug",  # slug 방식 (WordPress)
        "detail_url_base": "https://uosarch.ac.kr/uosarch_notice/",  # 상세 URL 베이스
        "selectors": {
            "title": ["h2.__post-title", ".__post-title"],
            "content": ["div.__post-content", ".__post-content"],
            "date_info": ["div.__post-date", ".__post-date", "div.__post-meta"],
            "view_count": "div.__post-view",  # Views 168 형태
        }
    },
    "DEPT_LANDSCAPE_ARCHITECTURE": {
        "category": "COLLEGE_URBAN_SCIENCE",
        "department": "조경학과",
        "list_url": "https://lauos.or.kr/notice",
        "id_param": "uid",  # URL 파라미터명
        "list_params": {},
        "url_type": "query",  # query 파라미터 방식
        "detail_url_template": "https://lauos.or.kr/notice?uid={post_id}&mod=document",
        "selectors": {
            "title": ["h1"],  # h1 태그에서 제목 추출 (2번째 h1)
            "content": ["div.kboard-content", "div.content-view"],
            "date_info": ["div.detail-value"],  # 작성일/조회수 정보
            "view_count": "div.detail-value",  # 텍스트에서 파싱
        }
    },
}
#################################################################################

# 몇 개 크롤링할 건지 
REQUEST_SLEEP = 1.0
PLAYWRIGHT_TIMEOUT_MS = 90000
RECENT_WINDOW = 50

# =========================
# 1) 유틸
# =========================

# 로그 출력
def log(msg: str) -> None:
    print(f"[indexer {datetime.now():%Y-%m-%d %H:%M:%S}] {msg}")

@contextmanager
def mysql_conn():
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
    m = re.search(r"(\d{4}-\d{2}-\d{2})", text or "")
    return m.group(1) if m else None


def extract_main_text_from_html(html: str, max_chars: int = 12000) -> str:
    """
    공지의 '본문' 컨테이너에서 텍스트만 추출.
    사이드/푸터/관련글/주소/카피라이트 등은 제거하고,
    길이가 너무 길면 max_chars로 잘라 모델 입력을 안정화.
    """
    soup = BeautifulSoup(html, "html.parser")

    # 본문 후보 셀렉터 (사이트 맞게 필요시 추가)
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

    # 흔한 푸터/주소/카피라이트 문구 제거
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
# 2) Playwright로 HTML → 이미지 캡처
# =========================
def html_to_images_playwright(
    url: str,
    viewport_width: int = 1200,
    slice_height: int = 1920,
    timeout_ms: int = PLAYWRIGHT_TIMEOUT_MS,
    debug_full_image_path: Optional[str] = None,  # 전체 페이지 1장 저장 경로
    full_image_format: str = "png",               # "png"|"jpeg"
) -> List[Image.Image]:
    """
    페이지 전체를 full_page 스크린샷으로 찍은 뒤,
    slice_height 간격으로 끝까지 전부 잘라서 반환.
    (max_slices 제한 없음)
    debug_full_image_path가 주어지면 전체 스크린샷 원본을 파일로 저장.
    """
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
            # User-Agent 및 기타 헤더 설정
            page = browser.new_page(
                viewport={"width": viewport_width, "height": slice_height},
                device_scale_factor=2.0,
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                extra_http_headers={
                    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
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

        # 전체 페이지 한 장 저장(테스트/디버그)
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
# 3) OpenAI: 이미지/텍스트 요약 + 임베딩
# =========================
def pil_to_data_url(pil_image: Image.Image, fmt="JPEG", quality=80) -> str:
    bio = BytesIO()
    pil_image.save(bio, format=fmt, quality=quality, optimize=True)
    b64 = base64.b64encode(bio.getvalue()).decode("utf-8")

    return f"data:image/{fmt.lower()};base64,{b64}"

def summarize_with_text_and_images(html_text: str, images: List[Image.Image]) -> str:
    """
    HTML 본문 텍스트를 우선 근거로 삼고,
    이미지(포스터/표 등)에만 있는 누락 정보를 보강하도록 지시.
    """
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
            "image_url": pil_to_data_url(img, fmt="JPEG", quality=75)  # JPEG로 압축
        })
    try:
        resp = client.responses.create(
            model=SUMMARIZE_MODEL,   # "gpt-4o" 권장
            input=[{"role": "user", "content": contents}],
            temperature=0.2,
        )
        return (resp.output_text or "").strip()
    except Exception as e:
        print(f"❌ 텍스트+이미지 요약 실패: {type(e).__name__}: {e}")
        traceback.print_exc(limit=2, file=sys.stdout)
        return ""

# =========================
# 4) HTML 파싱 (상세)
# =========================
CONNECT_TIMEOUT = 10    # 서버 TCP 연결까지 기다릴 최대 시간
READ_TIMEOUT    = 20   # 실제 응답(HTML)을 받는 시간

def fetch_notice_html(list_id: str, seq: int) -> Optional[str]:
    try:
        params = {
            "list_id": list_id,
            "seq": str(seq),
            "sort": "1",
            "pageIndex": "1",
            "searchCnd": "",
            "searchWrd": "",
            "cate_id": "",
            "viewAuth": "Y",
            "writeAuth": "Y",
            "board_list_num": "10",
            "lpageCount": "12",
            "menuid": "",
        }
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(CRAWL_VIEW_URL, params=params, headers=headers, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
        if r.status_code != 200:
            print(f"❌ HTTP {r.status_code} for seq={seq}")
            return None
        return r.text
    except Exception as e:
        print(f"❌ 요청 실패 seq={seq}: {e}")
        return None


def parse_notice_fields(html: str, seq: int) -> Optional[dict]:
    soup = BeautifulSoup(html, "html.parser")
    title_el = soup.select_one("div.vw-tibx h4") if soup else None
    title = title_el.get_text(strip=True) if title_el else ""
    if not title:
        return None  # 게시물 없음

    spans = soup.select("div.vw-tibx div.zl-bx div.da span")
    department = spans[1].get_text(strip=True) if len(spans) >= 3 else ""
    date_text = spans[2].get_text(strip=True) if len(spans) >= 3 else ""
    dt = parse_date_yyyy_mm_dd(date_text) or datetime.now().strftime("%Y-%m-%d")

    post_number_el = soup.select_one("input[name=seq]")
    post_number = int(post_number_el["value"]) if post_number_el and post_number_el.get("value") else int(seq)

    content_text = soup.get_text("\n", strip=True)
    return {
        "title": title,
        "department": department,
        "posted_date": dt,
        "post_number": post_number,
        "content_text": content_text,
    }


# =========================
# 5) DB 업서트
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

def get_existing_posted_date(category: str, post_number) -> Optional[str]:
    """
    post_number는 int 또는 str(slug)일 수 있음
    str(slug)인 경우 해시값으로 변환하여 저장
    """
    # post_number를 정수로 변환 (slug는 해시값으로)
    post_num = _normalize_post_number(post_number)

    with mysql_conn() as conn:
        cur = conn.cursor()
        cur.execute(EXISTS_SQL, (category, post_num))
        row = cur.fetchone()
        cur.close()
        return row[0] if row else None


def _normalize_post_number(post_number) -> int:
    """
    post_number를 정수로 정규화
    - 이미 int면 그대로 반환
    - str(slug)이면 32비트 해시값으로 변환
    """
    if isinstance(post_number, int):
        return post_number

    # 문자열인 경우 CRC32 해시 사용 (양수 보장)
    import zlib
    hash_val = zlib.crc32(post_number.encode('utf-8')) & 0x7fffffff
    return hash_val

def upsert_notice(row: dict):
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

def _ymd(x: Optional[object]) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, (datetime, date)):
        return x.strftime("%Y-%m-%d")
    s = str(x).strip()
    return s[:10]

# =========================
# 6) 파이프라인: 한 건 처리 (HTML 텍스트 + 이미지 동시 요약)
# =========================
def process_one(category_key: str, list_id: str, seq: int) -> str:
    # 1) HTML
    html = fetch_notice_html(list_id, seq)
    if not html:
        print(f"⚠️ Seq {seq}: HTML 로드 실패 → 스킵")
        return "skipped_error"

    # 2) 파싱
    parsed = parse_notice_fields(html, seq)
    if not parsed:
        print(f"Seq {seq}: 게시물 없음")
        return "not_found"

    post_number = parsed["post_number"]
    title = parsed["title"]
    department = parsed["department"]
    posted_date = parsed["posted_date"]

    # 3) 링크
    crawl_link = f"{CRAWL_VIEW_URL}list_id={list_id}&seq={seq}"
    db_link    = f"{SAVE_VIEW_URL}?{urlencode({'list_id': list_id, 'seq': seq})}"

    # 4) 중복 체크
    prev_dt_raw = get_existing_posted_date(category_key, post_number)
    prev_dt = _ymd(prev_dt_raw)
    curr_dt = _ymd(posted_date)

    if prev_dt:
        if prev_dt == curr_dt:
            # 날짜까지 동일 → 스킵
            print(f"Seq {seq} (post_number={post_number}) 이미 존재 (posted_date={curr_dt}) → 스킵")
            return "stored"
        else:
            # 날짜가 다름 → 수정된 게시물로 간주
            print(f"Seq {seq} (post_number={post_number}) 날짜 변경 {prev_dt} → {curr_dt}, 업데이트 진행")

    # 4-1) HTML 본문 텍스트 추출
    html_text = extract_main_text_from_html(html)

    # 5) HTML → 전체 이미지 캡처 (슬라이스 포함)
    imgs = html_to_images_playwright(
        crawl_link,
        viewport_width=1200,
        slice_height=1800,
        debug_full_image_path=None,     # 전체 1장 저장
        full_image_format="png",
    )
    if not imgs:
        print(f"↳ Seq {seq}: 이미지 캡처 실패 → 스킵")
        return "skipped_error"

    # 6) 텍스트 + 이미지 동시 요약
    summary = summarize_with_text_and_images(html_text, imgs)
    if not summary:
        print(f"↳ Seq {seq}: 텍스트+이미지 요약 실패 → 스킵")
        return "skipped_error"

    print(summary)

    # 8) DB 업서트
    row = {
        "category": category_key,
        "post_number": post_number,
        "title": title,
        "link": db_link,
        "summary": summary,
        "embedding_vector": None,
        "posted_date": posted_date,
        "department": department,
        "viewCount" : "0"
    } 
    try:
        upsert_notice(row)
        print(f"✅ 저장 완료: [{category_key}] seq={seq}, post_number={post_number}, posted_date={posted_date}, title={title[:30]}...")
        return "stored"
    except MySQLError as e:
        print(f"❌ DB 저장 실패: {e.__class__.__name__}({getattr(e,'errno',None)}): {e}")
        tb = traceback.format_exc(limit=3)
        print(f"↳ Traceback(요약):\n{tb}")
        return "skipped_error"


# =========================
# 7) 목록 HTML에서 seq 추출
# =========================

def extract_seqs_skip_pinned(html: str) -> List[int]:
    """
    목록에서 '공지' 배지가 붙은 고정글을 제외하고 seq만 추출.
    - 고정글 마크업: <p class="num"><span class="cl">공지</span></p>
    - 일반글: <p class="num">1506</p> 처럼 숫자 표시
    """
    soup = BeautifulSoup(html, "html.parser")
    seqs: List[int] = []

    # li 단위로 훑되, p.num 안에 span.cl(=공지) 있으면 skip
    for li in soup.select("li"):
        num = li.select_one("p.num")
        if num and (num.select_one("span.cl") or "공지" in num.get_text(strip=True)):
            continue  # 🔸 고정글 스킵

        # li 안에서 view.do 링크 찾고 seq 추출
        hrefs = [a.get("href", "") for a in li.select("a[href]")]
        found = False
        for href in hrefs:
            m = re.search(r"(?:\?|&|&amp;)seq=(\d+)", href)
            if m:
                seqs.append(int(m.group(1)))
                found = True
                break
        if found:
            continue

        # href에 없으면 onclick 계열에서 보조 추출 (예: goDetail('xxx','15583') or goDetail('xxx',15583))
        txt = li.decode()
        m = re.search(r"\(\s*['\"][^'\"]*['\"]\s*,\s*'(\d+)'\s*\)", txt)
        if not m:
            m = re.search(r"\(\s*['\"][^'\"]*['\"]\s*,\s*(\d+)\s*\)", txt)
        if m:
            seqs.append(int(m.group(1)))

    # 순서 유지한 중복 제거
    return list(OrderedDict.fromkeys(seqs))


def extract_seqs_from_list_html(html: str) -> List[int]:
    seqs: List[int] = []
    for m in re.finditer(r"view\.do[^\"'>]*(?:\?|&|&amp;)seq=(\d+)", html):
        seqs.append(int(m.group(1)))
    for m in re.finditer(r"\(\s*['\"][^'\"]*['\"]\s*,\s*'(\d+)'\s*\)", html):
        seqs.append(int(m.group(1)))
    for m in re.finditer(r"\(\s*['\"][^'\"]*['\"]\s*,\s*(\d+)\s*\)", html):
        seqs.append(int(m.group(1)))
    return list(OrderedDict.fromkeys(seqs))


def collect_recent_seqs(list_id: str,
                        extra_params: Optional[Dict[str, str]] = None,
                        limit: int = RECENT_WINDOW,
                        max_pages: int = 10) -> List[int]:
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.uos.ac.kr/"}
    collected: List[int] = []
    seen = set()

    for page in range(1, max_pages + 1):
        params = {"list_id": list_id, "pageIndex": str(page), "searchCnd": "", "searchWrd": ""}
        if extra_params:
            params.update(extra_params)

        r = requests.get(CRAWL_LIST_URL, params=params, headers=headers, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
        if r.status_code != 200:
            print(f"❌ 목록 HTTP {r.status_code} (list_id={list_id}, page={page}, params={params})")
            break

        if page == 1:
            page_seqs = extract_seqs_skip_pinned(r.text)
        else:
            page_seqs = extract_seqs_from_list_html(r.text)

        new_count = 0
        for s in page_seqs:
            if s not in seen:
                seen.add(s)
                collected.append(s)
                new_count += 1
                if len(collected) >= limit:
                    return collected

        if new_count == 0:
            break

        time.sleep(0.2)

    return collected

# =========================
# 8) 학과별 통합 처리 함수들
# =========================

def _collect_arch_slugs_with_rest_api(base_url: str, limit: int = 100) -> List[str]:
    """
    건축학부 목록을 WordPress REST API로 수집
    (Playwright 없이 빠르고 안정적으로 동작)
    """
    # WordPress REST API 엔드포인트
    api_url = "https://uosarch.ac.kr/wp-json/wp/v2/uosarch_notice"

    collected = []
    per_page = 100  # 한 번에 최대 100개
    page = 1

    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        while len(collected) < limit:
            params = {
                "per_page": min(per_page, limit - len(collected)),
                "page": page,
                "order": "desc",
                "orderby": "date"
            }

            print(f"[건축학부] REST API 페이지 {page} 요청 중...")
            r = requests.get(api_url, params=params, headers=headers, timeout=10)

            if r.status_code != 200:
                print(f"[건축학부] API 요청 실패: HTTP {r.status_code}")
                break

            posts = r.json()

            if not posts:
                print(f"[건축학부] 페이지 {page}: 더 이상 게시물 없음")
                break

            # slug 추출
            for post in posts:
                slug = post.get("slug", "")
                if slug:
                    # URL 디코딩 (REST API는 URL 인코딩된 slug 반환)
                    from urllib.parse import unquote
                    slug = unquote(slug)
                    collected.append(slug)

            print(f"[건축학부] 페이지 {page}: {len(posts)}개 수집 (누적: {len(collected)}개)")

            if len(collected) >= limit:
                break

            # 다음 페이지로
            page += 1
            time.sleep(0.2)  # API 부하 방지

    except Exception as e:
        print(f"❌ [건축학부] REST API 크롤링 실패: {e}")
        traceback.print_exc()

    # limit까지만 자르기
    collected = collected[:limit]
    print(f"[건축학부] 총 {len(collected)}개 slug 수집 완료")
    return collected


def collect_recent_seqs_generic(dept_key: str, limit: int = 100, max_pages: int = 20) -> List:
    """학과별 독립 URL에서 게시물 ID/slug 수집 (통합)"""
    config = DEPT_CONFIGS.get(dept_key)
    if not config:
        print(f"❌ 설정되지 않은 학과: {dept_key}")
        return []

    list_url = config["list_url"]
    id_param = config["id_param"]
    list_params = config.get("list_params", {})
    url_type = config.get("url_type", "query")  # "query", "path", "slug"

    # 건축학부는 WordPress REST API 사용
    if dept_key == "DEPT_ARCHITECTURE":
        return _collect_arch_slugs_with_rest_api(list_url, limit)

    headers = {"User-Agent": "Mozilla/5.0"}
    collected = []
    seen = set()

    for page in range(1, max_pages + 1):
        # URL 구성 (url_type에 따라 다름)
        if url_type in ["path", "slug"]:
            # 경로/slug 방식: /notices/undergraduate/page/2/ or /board/notice/page/2/
            if page == 1:
                url = list_url
            else:
                url = f"{list_url.rstrip('/')}/page/{page}/"
        else:
            # 쿼리 파라미터 방식: ?page=2&bo_table=notice
            params = {"page": page}
            params.update(list_params)
            url = list_url

        # 요청
        if url_type in ["path", "slug"]:
            r = requests.get(url, headers=headers, timeout=(10, 20))
        else:
            r = requests.get(url, params=params, headers=headers, timeout=(10, 20))

        if r.status_code != 200:
            print(f"❌ {dept_key} 목록 요청 실패 page={page}: {r.status_code}")
            break

        soup = BeautifulSoup(r.text, "html.parser")

        # ID/slug 추출 (url_type에 따라 다름)
        page_items = []

        if url_type == "slug":
            # WordPress slug 방식: /uosarch_notice/slug-name/
            for a in soup.select('a[href*="/uosarch_notice/"]'):
                href = a.get("href", "")
                m = re.search(r"/uosarch_notice/([^/]+)/?$", href)
                if m:
                    slug = m.group(1)
                    # URL 인코딩된 slug 디코딩
                    from urllib.parse import unquote
                    slug = unquote(slug)
                    page_items.append(slug)
        elif url_type == "path":
            # 경로 방식: /notices/19184 같은 링크에서 ID 추출
            for a in soup.select('a[href*="/notices/"]'):
                href = a.get("href", "")
                m = re.search(r"/notices/(\d+)", href)
                if m:
                    page_items.append(int(m.group(1)))
        else:
            # 쿼리 파라미터 방식: ?wr_id=123 같은 링크에서 ID 추출
            for a in soup.select(f'a[href*="{id_param}="]'):
                href = a.get("href", "")
                m = re.search(rf"{id_param}=(\d+)", href)
                if m:
                    page_items.append(int(m.group(1)))

        # 중복 제거 + 순서 유지
        page_items = list(OrderedDict.fromkeys(page_items))

        new_cnt = 0
        for item in page_items:
            if item not in seen:
                seen.add(item)
                collected.append(item)
                new_cnt += 1
                if len(collected) >= limit:
                    return collected

        if new_cnt == 0:
            break

        time.sleep(0.2)

    return collected


def fetch_notice_html_generic(dept_key: str, post_id) -> Optional[str]:
    """학과별 독립 URL에서 개별 공지 HTML 가져오기 (통합)
    post_id는 int 또는 str(slug)일 수 있음
    """
    config = DEPT_CONFIGS.get(dept_key)
    if not config:
        return None

    url_type = config.get("url_type", "query")

    # URL 구성 (url_type에 따라 다름)
    if url_type == "slug":
        # WordPress slug 방식: https://uosarch.ac.kr/uosarch_notice/slug-name/
        from urllib.parse import quote
        detail_url_base = config.get("detail_url_base")
        slug = post_id if isinstance(post_id, str) else str(post_id)
        # slug가 이미 인코딩되어 있을 수 있으므로, 안전하게 처리
        url = f"{detail_url_base}{slug}/"
    elif url_type == "path":
        # 경로 방식: https://econ.uos.ac.kr/notices/19184
        detail_url_template = config.get("detail_url_template")
        if detail_url_template:
            url = detail_url_template.format(post_id=post_id)
        else:
            # 템플릿이 없으면 기본 패턴 사용
            base = config["list_url"].split("/notices/")[0]
            url = f"{base}/notices/{post_id}"
    else:
        # 쿼리 파라미터 방식
        detail_url_template = config.get("detail_url_template")
        if detail_url_template:
            # 템플릿이 있으면 템플릿 사용
            url = detail_url_template.format(post_id=post_id)
        else:
            # 템플릿이 없으면 기본 파라미터 방식
            list_url = config["list_url"]
            id_param = config["id_param"]
            list_params = config.get("list_params", {})

            params = {id_param: post_id}
            params.update(list_params)

            # 생명과학과는 md=v 파라미터 추가
            if dept_key == "DEPT_LIFE_SCIENCE":
                params["md"] = "v"

            # URL 조합
            if "?" in list_url:
                url = f"{list_url}&{urlencode(params)}"
            else:
                url = f"{list_url}?{urlencode(params)}"

    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=(10, 20))
    if r.status_code != 200:
        print(f"❌ {dept_key} 상세 요청 실패 post_id={post_id}, status={r.status_code}")
        return None
    return r.text


def parse_notice_fields_generic(dept_key: str, html: str, post_id: int) -> Optional[dict]:
    """학과별 독립 URL HTML 파싱 (통합)"""
    config = DEPT_CONFIGS.get(dept_key)
    if not config:
        return None

    soup = BeautifulSoup(html, "html.parser")
    selectors = config["selectors"]
    department = config["department"]

    # 제목 추출
    title = ""
    for sel in selectors.get("title", []):
        if sel == "title":
            # <title> 태그 특수 처리
            title_el = soup.find("title")
            if title_el:
                title = title_el.get_text(strip=True)
                # "- 서울시립대학교 경제학부" 같은 접미사 제거
                title = re.sub(r"\s*[-–—]\s*서울시립대학교.*$", "", title).strip()
                break
        else:
            # 조경학과는 h1 태그가 2개이므로 두 번째 것 사용
            if dept_key == "DEPT_LANDSCAPE_ARCHITECTURE" and sel == "h1":
                h1_elements = soup.select("h1")
                if len(h1_elements) >= 2:
                    title = h1_elements[1].get_text(" ", strip=True)
                    break
            else:
                title_el = soup.select_one(sel)
                if title_el:
                    title = title_el.get_text(" ", strip=True)
                    break

    # 본문 추출
    content_text = ""
    content_selectors = selectors.get("content", [])
    if content_selectors:
        for sel in content_selectors:
            content_el = soup.select_one(sel)
            if content_el:
                content_text = content_el.get_text("\n", strip=True)
                break
    if not content_text:
        content_text = extract_main_text_from_html(html)

    # 날짜/조회수 정보 추출
    date_text, view_count = "", 0
    date_info_selectors = selectors.get("date_info", [])

    for sel in date_info_selectors:
        info_el = soup.select_one(sel)
        if info_el:
            text = info_el.get_text(" ", strip=True)

            # 날짜 추출
            if not date_text:
                # yyyy-mm-dd 형식
                m_date = re.search(r"\d{4}-\d{2}-\d{2}", text)
                if m_date:
                    date_text = m_date.group()
                else:
                    # yyyy년 mm월 dd일 형식 (경제학부)
                    m_date_kr = re.search(r"(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일", text)
                    if m_date_kr:
                        yyyy = int(m_date_kr.group(1))
                        mm = int(m_date_kr.group(2))
                        dd = int(m_date_kr.group(3))
                        date_text = f"{yyyy:04d}-{mm:02d}-{dd:02d}"
                    else:
                        # "월 일, 년" 형식 (건축학부: "9월 22, 2025")
                        m_date_mon = re.search(r"(\d{1,2})월\s+(\d{1,2}),\s+(\d{4})", text)
                        if m_date_mon:
                            mm = int(m_date_mon.group(1))
                            dd = int(m_date_mon.group(2))
                            yyyy = int(m_date_mon.group(3))
                            date_text = f"{yyyy:04d}-{mm:02d}-{dd:02d}"
                        else:
                            # yy-mm-dd 형식 (화학공학과)
                            m_date_short = re.search(r'(?<!\d)(?P<yy>\d{2})-(?P<mm>\d{2})-(?P<dd>\d{2})(?!\d)', text)
                            if m_date_short:
                                yy = int(m_date_short['yy'])
                                mm = int(m_date_short['mm'])
                                dd = int(m_date_short['dd'])
                                yyyy = 2000 + yy
                                date_text = f"{yyyy:04d}-{mm:02d}-{dd:02d}"

            # 조회수 추출
            if view_count == 0:
                # "조회수 123" 형식
                m_view = re.search(r"조회수\s*([0-9,]+)", text)
                if m_view:
                    view_count = int(m_view.group(1).replace(",", ""))
                else:
                    # "Views 168" 형식 (건축학부)
                    m_view_en = re.search(r"Views?\s+(\d+)", text, re.I)
                    if m_view_en:
                        view_count = int(m_view_en.group(1))

    # 화학공학과 조회수 (아이콘으로 찾기)
    if dept_key == "DEPT_CHEMICAL_ENGINEERING" and view_count == 0:
        view_count_el = soup.select_one("strong > i.fa-eye")
        if view_count_el:
            raw_text = view_count_el.find_previous("strong").text.strip()
            m = re.search(r'\d+', raw_text)
            view_count = int(m.group()) if m else 0

    # 조경학과 특수 처리 (div.detail-value 배열에서 인덱스로 파싱)
    if dept_key == "DEPT_LANDSCAPE_ARCHITECTURE":
        detail_values = soup.select("div.detail-value")
        if len(detail_values) >= 3:
            # [1]번째: 날짜 (2025-10-10 16:17)
            if not date_text:
                date_val = detail_values[1].get_text(strip=True)
                m_date = re.search(r"(\d{4}-\d{2}-\d{2})", date_val)
                if m_date:
                    date_text = m_date.group(1)

            # [2]번째: 조회수 (19)
            if view_count == 0:
                view_val = detail_values[2].get_text(strip=True)
                if view_val.isdigit():
                    view_count = int(view_val)

    posted_date = date_text or datetime.now().strftime("%Y-%m-%d")

    return {
        "title": title,
        "department": department,
        "posted_date": posted_date,
        "post_number": post_id,
        "content_text": content_text,
        "view_count": view_count,
    }


def process_one_generic(dept_key: str, post_id: int) -> str:
    """학과별 독립 URL 공지사항 한 건 처리 (통합)"""
    config = DEPT_CONFIGS.get(dept_key)
    if not config:
        print(f"❌ 설정되지 않은 학과: {dept_key}")
        return "skipped_error"

    category = config["category"]
    department = config["department"]
    list_url = config["list_url"]
    id_param = config["id_param"]

    # HTML 가져오기
    html = fetch_notice_html_generic(dept_key, post_id)
    if not html:
        print(f"⚠️ {dept_key} {id_param}={post_id}: HTML 로드 실패 → 스킵")
        return "skipped_error"

    # 파싱
    parsed = parse_notice_fields_generic(dept_key, html, post_id)
    if not parsed:
        print(f"{dept_key} {id_param}={post_id}: 게시물 없음")
        return "not_found"

    post_number = parsed["post_number"]
    title = parsed["title"]
    posted_date = parsed["posted_date"]
    view_count = parsed.get("view_count", 0)

    # 링크 생성 (url_type에 따라 다름)
    url_type = config.get("url_type", "query")

    if url_type == "slug":
        # WordPress slug 방식: https://uosarch.ac.kr/uosarch_notice/slug-name/
        detail_url_base = config.get("detail_url_base")
        slug = post_id if isinstance(post_id, str) else str(post_id)
        crawl_link = f"{detail_url_base}{slug}/"
        db_link = crawl_link
    elif url_type == "path":
        # 경로 방식: https://econ.uos.ac.kr/notices/19184
        detail_url_template = config.get("detail_url_template")
        if detail_url_template:
            crawl_link = detail_url_template.format(post_id=post_id)
        else:
            base = list_url.split("/notices/")[0]
            crawl_link = f"{base}/notices/{post_id}"
        db_link = crawl_link
    else:
        # 쿼리 파라미터 방식 (화학공학과, 생명과학과, 조경학과)
        detail_url_template = config.get("detail_url_template")
        if detail_url_template:
            # 템플릿이 있으면 템플릿 사용 (조경학과)
            crawl_link = detail_url_template.format(post_id=post_id)
        else:
            # 템플릿이 없으면 기본 파라미터 방식
            list_params = config.get("list_params", {})
            params = {id_param: post_id}
            params.update(list_params)

            if dept_key == "DEPT_LIFE_SCIENCE":
                params["md"] = "v"

            if "?" in list_url:
                crawl_link = f"{list_url}&{urlencode(params)}"
            else:
                crawl_link = f"{list_url}?{urlencode(params)}"

        db_link = crawl_link

    # post_number 정규화 (slug -> hash)
    normalized_post_number = _normalize_post_number(post_number)

    # 중복 체크
    prev_dt_raw = get_existing_posted_date(category, post_number)
    prev_dt = _ymd(prev_dt_raw)
    curr_dt = _ymd(posted_date)

    if prev_dt:
        if prev_dt == curr_dt:
            # 날짜는 같지만 조회수는 업데이트 (가벼운 업데이트)
            print(f"{dept_key} {id_param}={post_id} (post_number={post_number}) 이미 존재 → 조회수만 업데이트")

            # 조회수만 업데이트
            with mysql_conn() as conn:
                cur = conn.cursor()
                cur.execute(
                    "UPDATE notice SET view_count = %s WHERE category = %s AND post_number = %s",
                    (view_count, category, normalized_post_number)
                )
                cur.close()

            print(f"✅ 조회수 업데이트 완료: [{department}] {id_param}={post_id}, viewCount={view_count}")
            return "stored"
        else:
            print(f"{dept_key} {id_param}={post_id} (post_number={post_number}) 날짜 변경 {prev_dt} → {curr_dt}, 업데이트 진행")

    # HTML 본문 텍스트 추출
    html_text = extract_main_text_from_html(html)

    # HTML → 전체 이미지 캡처
    imgs = html_to_images_playwright(
        crawl_link,
        viewport_width=1200,
        slice_height=1800,
        debug_full_image_path=None,
        full_image_format="png",
    )
    if not imgs:
        print(f"↳ {dept_key} {id_param}={post_id}: 이미지 캡처 실패 → 스킵")
        return "skipped_error"

    # 텍스트 + 이미지 동시 요약
    summary = summarize_with_text_and_images(html_text, imgs)
    if not summary:
        print(f"↳ {dept_key} {id_param}={post_id}: 텍스트+이미지 요약 실패 → 스킵")
        return "skipped_error"

    print(summary)

    # DB 업서트
    row = {
        "category": category,
        "post_number": normalized_post_number,  # 정규화된 값 사용
        "title": title,
        "link": db_link,
        "summary": summary,
        "embedding_vector": None,
        "posted_date": posted_date,
        "department": department,
        "view_count": view_count
    }
    try:
        upsert_notice(row)
        print(f"✅ 저장 완료: [{department}] {id_param}={post_id}, post_number={normalized_post_number} (원본:{post_number}), title={title}, posted_date={posted_date}, viewCount={view_count}")
        return "stored"
    except MySQLError as e:
        print(f"❌ DB 저장 실패: {e.__class__.__name__}({getattr(e,'errno',None)}): {e}")
        tb = traceback.format_exc(limit=3)
        print(f"↳ Traceback(요약):\n{tb}")
        return "skipped_error"

# =========================
# 9) 실행부
# =========================
def main() -> int:
    print(f"Screenshot directory: {OUT_DIR}")

    # 포털 공통 카테고리 처리
    portal_targets = [
        # "GENERAL",
        # "ACADEMIC",
        # "COLLEGE_ENGINEERING",
        # "COLLEGE_HUMANITIES",
        # "COLLEGE_SOCIAL_SCIENCES",
        # "COLLEGE_URBAN_SCIENCE",
        # "COLLEGE_ARTS_SPORTS",
        # "COLLEGE_BUSINESS",
        # "COLLEGE_NATURAL_SCIENCES",
        # "COLLEGE_LIBERAL_CONVERGENCE",
    ]

    for cat in portal_targets:
        list_id = CATEGORIES.get(cat)
        if not list_id or "TODO" in list_id.lower():
            print(f"⏭️  {cat}: list_id 미설정 → 건너뜀")
            continue

        seqs = collect_recent_seqs(list_id, extra_params=None, limit=RECENT_WINDOW, max_pages=10)

        if not seqs:
            print(f"⚠️ {cat}: 목록에서 seq를 찾지 못해 건너뜀")
            continue

        print(f"==== [{cat}] list_id={list_id}, {len(seqs)}개 수집됨 (목록 노출 항목만) ====")
        for seq in reversed(seqs):
            process_one(cat, list_id, seq)
            time.sleep(REQUEST_SLEEP)

    # 학과별 독립 URL 처리 (통합 방식)
    for dept_key in DEPT_CONFIGS.keys():
        config = DEPT_CONFIGS[dept_key]
        department = config["department"]

        #해당 부분이 학과별 독립 링크에서 각각 몇개씩 가져올지를 설정
        seqs = collect_recent_seqs_generic(dept_key, limit=1)

        if not seqs:
            print(f"⚠️ [{department}]: 목록에서 게시물을 찾지 못해 건너뜀")
            continue

        print(f"==== [{department}] {len(seqs)}개 수집됨 ====", flush=True)
        for post_id in reversed(seqs):
            process_one_generic(dept_key, post_id)
            time.sleep(REQUEST_SLEEP)

    return 0

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)