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
# 카테고리 ↔ list_id 매핑
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
#################################################################################

CRAWL_VIEW_URL = "https://www.uos.ac.kr/korNotice/view.do?identified=anonymous&"
CRAWL_LIST_URL = "https://www.uos.ac.kr/korNotice/list.do?identified=anonymous&"

SAVE_VIEW_URL = "https://www.uos.ac.kr/korNotice/view.do"

#################################################################################

CHEME_LIST_URL = "https://cheme.uos.ac.kr/bbs/board.php?bo_table=notice" #화학공학과
LIFE_SCI_LIST_URL = "https://lifesci.uos.ac.kr/community/notice"         #생명과학과

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
            browser = p.chromium.launch(headless=True, args=[
                "--disable-web-security",
                "--hide-scrollbars",
            ])
            page = browser.new_page(
                viewport={"width": viewport_width, "height": slice_height},
                device_scale_factor=2.0,
            )
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)

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
    (category, post_number, title, link, summary, embedding_vector, posted_date, department)
VALUES
    (%s, %s, %s, %s, %s, %s, %s, %s) AS new
ON DUPLICATE KEY UPDATE
    title = new.title,
    link = new.link,
    summary = new.summary,
    embedding_vector = new.embedding_vector,
    posted_date = new.posted_date,
    department = new.department
"""

EXISTS_SQL = "SELECT posted_date FROM notice WHERE category=%s AND post_number=%s LIMIT 1"

def get_existing_posted_date(category: str, post_number: int) -> Optional[str]:
    with mysql_conn() as conn:
        cur = conn.cursor()
        cur.execute(EXISTS_SQL, (category, post_number))
        row = cur.fetchone()
        cur.close()
        return row[0] if row else None

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
# 8-1) 화학공학과
# =========================

def collect_recent_seqs_cheme(limit: int = 100, max_pages: int = 20) -> List[int]:
    headers = {"User-Agent": "Mozilla/5.0"}
    collected: List[int] = []
    seen = set()

    for page in range(1, max_pages + 1):
        params = {"bo_table": "notice", "page": page}
        r = requests.get(CHEME_LIST_URL, params=params, headers=headers, timeout=(10, 20))
        if r.status_code != 200:
            print(f"❌ 화학공학과 목록 요청 실패 page={page}: {r.status_code}")
            break

        soup = BeautifulSoup(r.text, "html.parser")

        # wr_id 수집 (댓글 앵커 등 제외)
        page_ids: List[int] = []
        for a in soup.select("a[href*='wr_id=']"):
            href = a.get("href", "")
            m = re.search(r"wr_id=(\d+)", href)
            if m:
                wr_id = int(m.group(1))
                # (선택) 댓글 앵커, 파일 링크 등 제외 조건이 필요하면 여기서 필터
                page_ids.append(wr_id)

        # 중복 제거 + 순서 유지
        page_ids = list(OrderedDict.fromkeys(page_ids))

        # 새로 본 wr_id만 추가
        new_cnt = 0
        for wid in page_ids:
            if wid not in seen:
                seen.add(wid)
                collected.append(wid)
                new_cnt += 1
                if len(collected) >= limit:
                    return collected

        # 이 페이지에서 새로 얻은 게 없으면 중단
        if new_cnt == 0:
            break

        time.sleep(0.2) 

    return collected

def fetch_notice_html_cheme(wr_id: int) -> Optional[str]:
    """화학공학과 개별 공지 HTML 가져오기"""
    url = f"{CHEME_LIST_URL}&wr_id={wr_id}"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=(10, 20))
    if r.status_code != 200:
        print(f"❌ 화학공학과 상세 요청 실패 wr_id={wr_id}, status={r.status_code}")
        return None
    return r.text

def parse_date_any(text: str) -> Optional[str]:
    if not text:
        return None
    t = text.strip()
    # 예: 25-09-24, 25-09-24 11:02, (25-09-24) 등 변형도 허용
    m = re.search(r'(?<!\d)(?P<yy>\d{2})-(?P<mm>\d{2})-(?P<dd>\d{2})(?!\d)', t)
    if m:
        yy = int(m['yy']); mm = int(m['mm']); dd = int(m['dd'])
        yyyy = 2000 + yy          # 20xx로 해석
        return f"{yyyy:04d}-{mm:02d}-{dd:02d}"
    return None

def parse_notice_fields_cheme(html: str, wr_id: int) -> Optional[dict]:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")

    # ✅ 제목: h2#bo_v_title > span.bo_v_tit
    title_el = soup.select_one("#bo_v_title .bo_v_tit") or soup.select_one("#bo_v_title")
    title = title_el.get_text(" ", strip=True) if title_el else ""

    # ✅ 본문: section#bo_v_atc (gnuboard 본문 영역)
    content_el = soup.select_one("#bo_v_atc") or soup.select_one(".board_view, .view_content, #bo_v")
    content_text = content_el.get_text("\n", strip=True) if content_el else soup.get_text("\n", strip=True)

    # ✅ 날짜: section#bo_v_info 등
    date_el = soup.select_one("#bo_v_info, .bo_v_info, .view_info, .board_view .info")
    date_text = date_el.get_text(" ", strip=True) if date_el else datetime.now().strftime("%Y-%m-%d")

    # 조회수 추출 (예시: 33)
    view_count_el = soup.select_one("strong > i.fa-eye")  # 조회수에 해당하는 i 태그를 선택

    if view_count_el:
        raw_text = view_count_el.find_previous("strong").text.strip()
        m = re.search(r'\d+', raw_text)  # 숫자만 추출
        view_count = int(m.group()) if m else 0
    else:
        view_count = 0  

    return {
        "title": title,                         # ← 이제 깔끔한 제목
        "department": "화학공학과",
        "posted_date": parse_date_any(date_text) or datetime.now().strftime("%Y-%m-%d"),
        "post_number": wr_id,
        "content_text": content_text,
        "view_count": view_count  # 조회수 추가
    }

def process_one_cheme(wr_id: int) -> str:
    """화학공학과 공지사항 한 건 처리 (포털 방식과 동일하게)"""
    html = fetch_notice_html_cheme(wr_id)
    if not html:
        print(f"⚠️ wr_id={wr_id}: HTML 로드 실패 → 스킵")
        return "skipped_error"

    parsed = parse_notice_fields_cheme(html, wr_id)
    if not parsed:
        print(f"wr_id={wr_id}: 게시물 없음")
        return "not_found"

    post_number = parsed["post_number"]
    title = parsed["title"]
    department = parsed["department"]
    posted_date = parsed["posted_date"]
    view_count = parsed["view_count"]  

    # 링크
    crawl_link = f"{CHEME_LIST_URL}&wr_id={wr_id}"
    db_link    = crawl_link  # DB에 저장할 링크

    # 중복 체크
    prev_dt_raw = get_existing_posted_date("COLLEGE_ENGINEERING", post_number)
    prev_dt = _ymd(prev_dt_raw)
    curr_dt = _ymd(posted_date)

    if prev_dt:
        if prev_dt == curr_dt:
            print(f"wr_id={wr_id} (post_number={post_number}) 이미 존재 (posted_date={curr_dt}) → 스킵")
            return "stored"
        else:
            print(f"wr_id={wr_id} (post_number={post_number}) 날짜 변경 {prev_dt} → {curr_dt}, 업데이트 진행")

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
        print(f"↳ wr_id={wr_id}: 이미지 캡처 실패 → 스킵")
        return "skipped_error"

    # 텍스트 + 이미지 동시 요약
    summary = summarize_with_text_and_images(html_text, imgs)
    if not summary:
        print(f"↳ wr_id={wr_id}: 텍스트+이미지 요약 실패 → 스킵")
        return "skipped_error"

    print(summary)
    # DB 업서트
    row = {
        "category": "COLLEGE_ENGINEERING",
        "post_number": post_number,
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
        print(f"✅ 저장 완료: [화학공학과] wr_id={wr_id}, post_number={post_number}, title={title[:50]}, link={db_link}, posted_date={posted_date}, department={department}, viewCount={view_count}")
        return "stored"
    except MySQLError as e:
        print(f"❌ DB 저장 실패: {e.__class__.__name__}({getattr(e,'errno',None)}): {e}")
        tb = traceback.format_exc(limit=3)
        print(f"↳ Traceback(요약):\n{tb}")
        return "skipped_error"
    
# =========================
# 8-2) 생명과학과
# =========================

def collect_recent_seqs_lifesci(limit: int = 100, max_pages: int = 20) -> List[int]:
    headers = {"User-Agent": "Mozilla/5.0"}
    collected: List[int] = []
    seen = set()

    for page in range(1, max_pages + 1):
        params = {"page": page}
        r = requests.get(LIFE_SCI_LIST_URL, params=params, headers=headers, timeout=(10, 20))
        if r.status_code != 200:
            print(f"❌ 생명과학과 목록 요청 실패 page={page}: {r.status_code}")
            break

        soup = BeautifulSoup(r.text, "html.parser")

        # ✅ 리스트 구조나 배지 클래스에 의존하지 않고, bbsidx 링크만 수집
        page_ids: List[int] = []
        for a in soup.select('a[href*="bbsidx="]'):
            href = a.get("href", "")
            m = re.search(r"bbsidx=(\d+)", href)
            if m:
                page_ids.append(int(m.group(1)))

        # 중복 제거 + 순서 유지
        page_ids = list(OrderedDict.fromkeys(page_ids))

        new_cnt = 0
        for idx in page_ids:
            if idx not in seen:
                seen.add(idx)
                collected.append(idx)
                new_cnt += 1
                if len(collected) >= limit:
                    return collected

        if new_cnt == 0:
            break

        time.sleep(0.2)

    return collected

def fetch_notice_html_lifesci(bbsidx: int) -> Optional[str]:
    """생명과학과 개별 공지 HTML 가져오기 (화공과 fetch 함수와 구조 동일)"""
    # URL 구조: ...notice?md=v&bbsidx=11971
    url = f"{LIFE_SCI_LIST_URL}?md=v&bbsidx={bbsidx}"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=(10, 20))
    if r.status_code != 200:
        print(f"❌ 생명과학과 상세 요청 실패 bbsidx={bbsidx}, status={r.status_code}")
        return None
    return r.text

def parse_notice_fields_lifesci(html: str, bbsidx: int) -> Optional[dict]:
    soup = BeautifulSoup(html, "html.parser")

    # ✅ 제목: h1.bbstitle
    title_el = soup.select_one("h1.bbstitle")
    title = title_el.get_text(" ", strip=True) if title_el else ""

    # ✅ 날짜/조회수: div.writer 안의 텍스트에서 추출
    writer_el = soup.select_one("div.writer")
    date_text, view_count = "", 0
    if writer_el:
        text = writer_el.get_text(" ", strip=True)
        # 날짜 추출 (예: 2022-07-15)
        m_date = re.search(r"\d{4}-\d{2}-\d{2}", text)
        if m_date:
            date_text = m_date.group()
        # 조회수 추출 (예: 조회수 525)
        m_view = re.search(r"조회수\s*([0-9,]+)", text)
        if m_view:
            view_count = int(m_view.group(1).replace(",", ""))

    # ✅ 본문 추출: extract_main_text_from_html 이용
    content_text = extract_main_text_from_html(html)

    # ✅ 날짜 없으면 오늘 날짜로 대체
    posted_date = parse_date_yyyy_mm_dd(date_text) or datetime.now().strftime("%Y-%m-%d")

    return {
        "title": title,
        "department": "생명과학과",
        "posted_date": posted_date,
        "post_number": bbsidx,
        "content_text": content_text,
        "view_count": view_count,
    }

def process_one_lifesci(bbsidx: int) -> str:
    """생명과학과 공지사항 한 건 처리 (화공과 process 함수와 구조 동일)"""
    html = fetch_notice_html_lifesci(bbsidx)
    if not html:
        print(f"⚠️ bbsidx={bbsidx}: HTML 로드 실패 → 스킵")
        return "skipped_error"

    parsed = parse_notice_fields_lifesci(html, bbsidx)
    if not parsed:
        print(f"bbsidx={bbsidx}: 게시물 없음")
        return "not_found"

    post_number = parsed["post_number"]
    title = parsed["title"]
    department = parsed["department"]
    posted_date = parsed["posted_date"]
    view_count = parsed["view_count"] 

    # 링크
    crawl_link = f"{LIFE_SCI_LIST_URL}?md=v&bbsidx={bbsidx}"
    db_link = crawl_link 

    # 중복 체크: 자연과학대학 카테고리 사용 (COLLEGE_NATURAL_SCIENCES)
    prev_dt_raw = get_existing_posted_date("COLLEGE_NATURAL_SCIENCES", post_number)
    prev_dt = _ymd(prev_dt_raw)
    curr_dt = _ymd(posted_date)

    if prev_dt:
        if prev_dt == curr_dt:
            print(f"bbsidx={bbsidx} (post_number={post_number}) 이미 존재 (posted_date={curr_dt}) → 스킵")
            return "stored"
        else:
            print(f"bbsidx={bbsidx} (post_number={post_number}) 날짜 변경 {prev_dt} → {curr_dt}, 업데이트 진행")

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
        print(f"↳ bbsidx={bbsidx}: 이미지 캡처 실패 → 스킵")
        return "skipped_error"

    # 텍스트 + 이미지 동시 요약
    summary = summarize_with_text_and_images(html_text, imgs)
    if not summary:
        print(f"↳ bbsidx={bbsidx}: 텍스트+이미지 요약 실패 → 스킵")
        return "skipped_error"

    print(summary)
    
    # DB 업서트
    row = {
        "category": "COLLEGE_NATURAL_SCIENCES",
        "post_number": post_number,
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
        print(f"✅ 저장 완료: [생명과학과] bbsidx={bbsidx}, post_number={post_number}, title={title[:50]}, link={db_link}, posted_date={posted_date}, viewcount={view_count}, department={department}")
        return "stored"
    except MySQLError as e:
        print(f"❌ DB 저장 실패: {e.__class__.__name__}: {e}")
        traceback.print_exc(limit=3, file=sys.stdout)
        return "skipped_error"

# =========================
# 9) 실행부
# =========================
def main() -> int:
    print(f"Screenshot directory: {OUT_DIR}")

    targets = [
        "GENERAL",
        "ACADEMIC",
        "COLLEGE_ENGINEERING",
        "COLLEGE_HUMANITIES",
        "COLLEGE_SOCIAL_SCIENCES",
        "COLLEGE_URBAN_SCIENCE",
        "COLLEGE_ARTS_SPORTS",
        "COLLEGE_BUSINESS",
        "COLLEGE_NATURAL_SCIENCES",
        "COLLEGE_LIBERAL_CONVERGENCE"
    ]

    for cat in targets:
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

    # # 🔹 화학공학과 공지 처리
    seqs = collect_recent_seqs_cheme(limit=100)
    
    print(f"==== [화학공학과] {len(seqs)}개 수집됨 ====", flush=True)
    for wr_id in reversed(seqs):
        process_one_cheme(wr_id)
        time.sleep(REQUEST_SLEEP)

    # 🔹 생명과학과 공지 처리
    seqs = collect_recent_seqs_lifesci(limit=100)
    
    print(f"==== [생명과학과] {len(seqs)}개 수집됨 ====", flush=True)
    for wr_id in reversed(seqs):
        process_one_lifesci(wr_id)
        time.sleep(REQUEST_SLEEP)

    return 0

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)
