# src/uosai/crawler/architecture_crawler.py
"""
건축학부 공지사항 크롤러
- URL: https://uosarch.ac.kr/board/notice/
- 특징: WordPress 기반, slug 방식
- REST API 사용: https://uosarch.ac.kr/wp-json/wp/v2/uosarch_notice
"""

import time
import re
from typing import List, Optional
from urllib.parse import unquote
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from mysql.connector import Error as MySQLError

from crawler_utils import (
    log, mysql_conn, extract_main_text_from_html,
    html_to_images_playwright, summarize_with_text_and_images,
    get_existing_posted_date, upsert_notice, update_view_count,
    _normalize_post_number, _ymd,
    CONNECT_TIMEOUT, READ_TIMEOUT
)

# =========================
# 설정
# =========================
CATEGORY = "COLLEGE_URBAN_SCIENCE"
DEPARTMENT = "건축학부"
LIST_URL = "https://uosarch.ac.kr/board/notice/"
REST_API_URL = "https://uosarch.ac.kr/wp-json/wp/v2/uosarch_notice"
DETAIL_URL_BASE = "https://uosarch.ac.kr/uosarch_notice/"

REQUEST_SLEEP = 1.0
RECENT_WINDOW = 50
MAX_PAGES = 20

# =========================
# 목록 수집 (REST API 사용)
# =========================

def collect_recent_post_slugs(limit: int = RECENT_WINDOW) -> List[str]:
    """
    WordPress REST API를 통해 게시물 slug 수집
    (Playwright 없이 빠르고 안정적)
    """
    collected = []
    per_page = 100
    page = 1
    headers = {'User-Agent': 'Mozilla/5.0'}

    try:
        while len(collected) < limit:
            params = {
                "per_page": min(per_page, limit - len(collected)),
                "page": page,
                "order": "desc",
                "orderby": "date"
            }

            log(f"[{DEPARTMENT}] REST API 페이지 {page} 요청 중...")
            r = requests.get(REST_API_URL, params=params, headers=headers, timeout=10)

            if r.status_code != 200:
                log(f"❌ [{DEPARTMENT}] API 요청 실패: HTTP {r.status_code}")
                break

            posts = r.json()

            if not posts:
                log(f"[{DEPARTMENT}] 페이지 {page}: 더 이상 게시물 없음")
                break

            # slug 추출
            for post in posts:
                slug = post.get("slug", "")
                if slug:
                    # URL 디코딩
                    slug = unquote(slug)
                    collected.append(slug)

            log(f"[{DEPARTMENT}] API 응답: {len(posts)}개 수집 (누적: {len(collected)}개)")

            if len(collected) >= limit:
                break

            page += 1
            time.sleep(0.2)

    except Exception as e:
        log(f"❌ [{DEPARTMENT}] REST API 크롤링 실패: {e}")
        import traceback
        traceback.print_exc()

    # limit까지만 자르기
    collected = collected[:limit]
    log(f"[{DEPARTMENT}] 총 {len(collected)}개 slug 수집 완료")
    return collected


# =========================
# 상세 페이지 파싱
# =========================

def fetch_notice_html(slug: str) -> Optional[str]:
    """게시물 HTML 가져오기"""
    url = f"{DETAIL_URL_BASE}{slug}/"
    headers = {'User-Agent': 'Mozilla/5.0'}

    try:
        r = requests.get(url, headers=headers, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
        if r.status_code != 200:
            log(f"❌ [{DEPARTMENT}] 상세 요청 실패 slug={slug}, status={r.status_code}")
            return None
        return r.text
    except Exception as e:
        log(f"❌ [{DEPARTMENT}] 요청 실패 slug={slug}: {e}")
        return None


def parse_notice_fields(html: str, slug: str) -> Optional[dict]:
    """HTML 파싱하여 공지사항 필드 추출"""
    soup = BeautifulSoup(html, 'html.parser')

    # 제목 추출
    title = ""
    title_el = soup.select_one('h2.__post-title')
    if not title_el:
        title_el = soup.select_one('.__post-title')
    if title_el:
        title = title_el.get_text(' ', strip=True)

    if not title:
        log(f"[{DEPARTMENT}] slug={slug}: 제목 없음 (게시물 없거나 삭제됨)")
        return None

    # 본문 추출
    content_text = ""
    content_el = soup.select_one('div.__post-content')
    if not content_el:
        content_el = soup.select_one('.__post-content')
    if content_el:
        content_text = content_el.get_text('\n', strip=True)

    if not content_text:
        content_text = extract_main_text_from_html(html)

    # 날짜 추출
    date_text = ""
    date_el = soup.select_one('div.__post-date')
    if not date_el:
        date_el = soup.select_one('.__post-date')
    if not date_el:
        date_el = soup.select_one('div.__post-meta')

    if date_el:
        text = date_el.get_text(' ', strip=True)
        # "9월 22, 2025" 형식
        m = re.search(r'(\d{1,2})월\s+(\d{1,2}),\s+(\d{4})', text)
        if m:
            mm = int(m.group(1))
            dd = int(m.group(2))
            yyyy = int(m.group(3))
            date_text = f"{yyyy:04d}-{mm:02d}-{dd:02d}"

    if not date_text:
        date_text = datetime.now().strftime("%Y-%m-%d")

    # 조회수 추출
    view_count = 0
    view_el = soup.select_one('div.__post-view')
    if view_el:
        text = view_el.get_text(' ', strip=True)
        # "Views 168" 형식
        m = re.search(r'Views?\s+(\d+)', text, re.I)
        if m:
            view_count = int(m.group(1))

    return {
        "title": title,
        "department": DEPARTMENT,
        "posted_date": date_text,
        "post_number": slug,  # slug를 post_number로 사용
        "content_text": content_text,
        "view_count": view_count,
    }


# =========================
# 한 건 처리
# =========================

def process_one(slug: str) -> str:
    """건축학부 공지사항 한 건 처리"""

    # 1) HTML 가져오기
    html = fetch_notice_html(slug)
    if not html:
        log(f"⚠️ [{DEPARTMENT}] slug={slug}: HTML 로드 실패 → 스킵")
        return "skipped_error"

    # 2) 파싱
    parsed = parse_notice_fields(html, slug)
    if not parsed:
        log(f"[{DEPARTMENT}] slug={slug}: 게시물 없음")
        return "not_found"

    post_number = parsed["post_number"]  # slug
    title = parsed["title"]
    posted_date = parsed["posted_date"]
    view_count = parsed.get("view_count", 0)

    # 3) 링크 생성
    crawl_link = f"{DETAIL_URL_BASE}{slug}/"
    db_link = crawl_link

    # 4) post_number 정규화 (slug → hash)
    normalized_post_number = _normalize_post_number(post_number)

    # 5) 중복 체크
    prev_dt_raw = get_existing_posted_date(CATEGORY, post_number)
    prev_dt = _ymd(prev_dt_raw)
    curr_dt = _ymd(posted_date)

    if prev_dt:
        if prev_dt == curr_dt:
            # 날짜 동일 → 조회수만 업데이트
            log(f"[{DEPARTMENT}] slug={slug} 이미 존재 → 조회수만 업데이트")
            update_view_count(CATEGORY, normalized_post_number, view_count)
            log(f"✅ [{DEPARTMENT}] 조회수 업데이트: slug={slug}, viewCount={view_count}")
            return "stored"
        else:
            log(f"[{DEPARTMENT}] slug={slug} 날짜 변경 {prev_dt} → {curr_dt}, 업데이트 진행")

    # 6) HTML 본문 텍스트 추출
    html_text = extract_main_text_from_html(html)

    # 7) 이미지 캡처
    imgs = html_to_images_playwright(
        crawl_link,
        viewport_width=1200,
        slice_height=1800,
        debug_full_image_path=None,
        full_image_format="png",
    )
    if not imgs:
        log(f"↳ [{DEPARTMENT}] slug={slug}: 이미지 캡처 실패 → 스킵")
        return "skipped_error"

    # 8) 텍스트 + 이미지 요약
    summary = summarize_with_text_and_images(html_text, imgs)
    if not summary:
        log(f"↳ [{DEPARTMENT}] slug={slug}: 요약 실패 → 스킵")
        return "skipped_error"

    print(summary)

    # 9) DB 업서트
    row = {
        "category": CATEGORY,
        "post_number": normalized_post_number,
        "title": title,
        "link": db_link,
        "summary": summary,
        "embedding_vector": None,
        "posted_date": posted_date,
        "department": DEPARTMENT,
        "view_count": view_count
    }
    try:
        upsert_notice(row)
        log(f"✅ [{DEPARTMENT}] 저장 완료: slug={slug}, post_number={normalized_post_number}, title={title}, posted_date={posted_date}")
        return "stored"
    except MySQLError as e:
        log(f"❌ [{DEPARTMENT}] DB 저장 실패: {e.__class__.__name__}: {e}")
        return "skipped_error"


# =========================
# 메인 실행
# =========================

def main():
    """건축학부 크롤러 메인"""
    log(f"==== [{DEPARTMENT}] 크롤링 시작 ====")

    # 1) 목록 수집
    slugs = collect_recent_post_slugs(limit=RECENT_WINDOW)

    if not slugs:
        log(f"⚠️ [{DEPARTMENT}]: 목록에서 게시물을 찾지 못해 종료")
        return 0

    log(f"==== [{DEPARTMENT}] {len(slugs)}개 수집됨 ====")

    # 2) 각 게시물 처리 (역순: 오래된 것부터)
    for slug in reversed(slugs):
        process_one(slug)
        time.sleep(REQUEST_SLEEP)

    log(f"==== [{DEPARTMENT}] 크롤링 완료 ====")
    return 0


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
