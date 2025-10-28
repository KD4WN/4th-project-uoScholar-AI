# src/uosai/crawler/portal_crawler.py
"""
서울시립대 포털 공통 크롤러
- URL: https://www.uos.ac.kr/korNotice/
- 카테고리: 일반공지, 학사공지, 각 단과대학 공지
"""

import time
import re
from typing import List, Optional, Dict
from collections import OrderedDict
from urllib.parse import urlencode
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from mysql.connector import Error as MySQLError

from crawler_utils import (
    log, mysql_conn, parse_date_yyyy_mm_dd, extract_main_text_from_html,
    html_to_images_playwright, summarize_with_text_and_images,
    get_existing_posted_date, upsert_notice,
    _normalize_post_number, _ymd,
    CONNECT_TIMEOUT, READ_TIMEOUT
)

# =========================
# 설정
# =========================

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

CRAWL_VIEW_URL = "https://www.uos.ac.kr/korNotice/view.do?identified=anonymous&"
CRAWL_LIST_URL = "https://www.uos.ac.kr/korNotice/list.do?identified=anonymous&"
SAVE_VIEW_URL = "https://www.uos.ac.kr/korNotice/view.do"

REQUEST_SLEEP = 1.0
RECENT_WINDOW = 50
MAX_PAGES = 10

# =========================
# 목록 수집
# =========================

def extract_seqs_skip_pinned(html: str) -> List[int]:
    """
    목록에서 '공지' 배지가 붙은 고정글을 제외하고 seq만 추출
    """
    soup = BeautifulSoup(html, "html.parser")
    seqs: List[int] = []

    # li 단위로 훑되, p.num 안에 span.cl(=공지) 있으면 skip
    for li in soup.select("li"):
        num = li.select_one("p.num")
        if num and (num.select_one("span.cl") or "공지" in num.get_text(strip=True)):
            continue  # 고정글 스킵

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

        # href에 없으면 onclick 계열에서 보조 추출
        txt = li.decode()
        m = re.search(r"\(\s*['\"][^'\"]*['\"]\s*,\s*'(\d+)'\s*\)", txt)
        if not m:
            m = re.search(r"\(\s*['\"][^'\"]*['\"]\s*,\s*(\d+)\s*\)", txt)
        if m:
            seqs.append(int(m.group(1)))

    # 순서 유지한 중복 제거
    return list(OrderedDict.fromkeys(seqs))


def extract_seqs_from_list_html(html: str) -> List[int]:
    """목록 HTML에서 seq 추출 (일반 페이지용)"""
    seqs: List[int] = []
    for m in re.finditer(r"view\.do[^\"'>]*(?:\?|&|&amp;)seq=(\d+)", html):
        seqs.append(int(m.group(1)))
    for m in re.finditer(r"\(\s*['\"][^'\"]*['\"]\s*,\s*'(\d+)'\s*\)", html):
        seqs.append(int(m.group(1)))
    for m in re.finditer(r"\(\s*['\"][^'\"]*['\"]\s*,\s*(\d+)\s*\)", html):
        seqs.append(int(m.group(1)))
    return list(OrderedDict.fromkeys(seqs))


def collect_recent_seqs(
    list_id: str,
    extra_params: Optional[Dict[str, str]] = None,
    limit: int = RECENT_WINDOW,
    max_pages: int = MAX_PAGES
) -> List[int]:
    """포털 공통 목록에서 seq 수집"""
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.uos.ac.kr/"}
    collected: List[int] = []
    seen = set()

    for page in range(1, max_pages + 1):
        params = {"list_id": list_id, "pageIndex": str(page), "searchCnd": "", "searchWrd": ""}
        if extra_params:
            params.update(extra_params)

        try:
            r = requests.get(CRAWL_LIST_URL, params=params, headers=headers, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
        except Exception as e:
            log(f"❌ 목록 요청 실패 (list_id={list_id}, page={page}): {e}")
            break

        if r.status_code != 200:
            log(f"❌ 목록 HTTP {r.status_code} (list_id={list_id}, page={page})")
            break

        # 1페이지는 고정글 제외, 나머지는 일반 추출
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

        log(f"[포털] list_id={list_id} 페이지 {page}: {len(page_seqs)}개 발견, {new_count}개 신규 (누적: {len(collected)}개)")

        if new_count == 0:
            break

        time.sleep(0.2)

    return collected


# =========================
# 상세 페이지 파싱
# =========================

def fetch_notice_html(list_id: str, seq: int) -> Optional[str]:
    """게시물 HTML 가져오기"""
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
            log(f"❌ HTTP {r.status_code} for seq={seq}")
            return None
        return r.text
    except Exception as e:
        log(f"❌ 요청 실패 seq={seq}: {e}")
        return None


def parse_notice_fields(html: str, seq: int) -> Optional[dict]:
    """HTML 파싱하여 공지사항 필드 추출"""
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
# 한 건 처리
# =========================

def process_one(category_key: str, list_id: str, seq: int) -> str:
    """포털 공통 공지사항 한 건 처리"""

    # 1) HTML 가져오기
    html = fetch_notice_html(list_id, seq)
    if not html:
        log(f"⚠️ Seq {seq}: HTML 로드 실패 → 스킵")
        return "skipped_error"

    # 2) 파싱
    parsed = parse_notice_fields(html, seq)
    if not parsed:
        log(f"Seq {seq}: 게시물 없음")
        return "not_found"

    post_number = parsed["post_number"]
    title = parsed["title"]
    department = parsed["department"]
    posted_date = parsed["posted_date"]

    # 3) 링크
    crawl_link = f"{CRAWL_VIEW_URL}list_id={list_id}&seq={seq}"
    db_link = f"{SAVE_VIEW_URL}?{urlencode({'list_id': list_id, 'seq': seq})}"

    # 4) 중복 체크
    prev_dt_raw = get_existing_posted_date(category_key, post_number)
    prev_dt = _ymd(prev_dt_raw)
    curr_dt = _ymd(posted_date)

    if prev_dt:
        if prev_dt == curr_dt:
            log(f"Seq {seq} (post_number={post_number}) 이미 존재 (posted_date={curr_dt}) → 스킵")
            return "stored"
        else:
            log(f"Seq {seq} (post_number={post_number}) 날짜 변경 {prev_dt} → {curr_dt}, 업데이트 진행")

    # 5) HTML 본문 텍스트 추출
    html_text = extract_main_text_from_html(html)

    # 6) 이미지 캡처
    imgs = html_to_images_playwright(
        crawl_link,
        viewport_width=1200,
        slice_height=1800,
        debug_full_image_path=None,
        full_image_format="png",
    )
    if not imgs:
        log(f"↳ Seq {seq}: 이미지 캡처 실패 → 스킵")
        return "skipped_error"

    # 7) 텍스트 + 이미지 요약
    summary = summarize_with_text_and_images(html_text, imgs)
    if not summary:
        log(f"↳ Seq {seq}: 요약 실패 → 스킵")
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
        "view_count": 0
    }
    try:
        upsert_notice(row)
        log(f"✅ 저장 완료: [{category_key}] seq={seq}, post_number={post_number}, title={title[:30]}...")
        return "stored"
    except MySQLError as e:
        log(f"❌ DB 저장 실패: {e.__class__.__name__}: {e}")
        return "skipped_error"


# =========================
# 메인 실행
# =========================

def main():
    """포털 공통 크롤러 메인"""
    log("==== 포털 공통 크롤링 시작 ====")

    portal_targets = [
        "GENERAL",
        "ACADEMIC",
        "COLLEGE_ENGINEERING",
        "COLLEGE_HUMANITIES",
        "COLLEGE_SOCIAL_SCIENCES",
        "COLLEGE_URBAN_SCIENCE",
        "COLLEGE_ARTS_SPORTS",
        "COLLEGE_BUSINESS",
        "COLLEGE_NATURAL_SCIENCES",
        "COLLEGE_LIBERAL_CONVERGENCE",
    ]

    for cat in portal_targets:
        list_id = CATEGORIES.get(cat)
        if not list_id or "TODO" in list_id.upper():
            log(f"⏭️  {cat}: list_id 미설정 → 건너뜀")
            continue

        seqs = collect_recent_seqs(list_id, extra_params=None, limit=RECENT_WINDOW, max_pages=MAX_PAGES)

        if not seqs:
            log(f"⚠️ {cat}: 목록에서 seq를 찾지 못해 건너뜀")
            continue

        log(f"==== [{cat}] list_id={list_id}, {len(seqs)}개 수집됨 ====")
        for seq in reversed(seqs):
            process_one(cat, list_id, seq)
            time.sleep(REQUEST_SLEEP)

    log("==== 포털 공통 크롤링 완료 ====")
    return 0


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
