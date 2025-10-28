# src/uosai/crawler/chemical_crawler.py
"""
화학공학과 공지사항 크롤러
- URL: https://cheme.uos.ac.kr/bbs/board.php?bo_table=notice
- 특징: 그누보드(Gnuboard) 기반, 쿼리 파라미터 방식
- 상세 URL: ?bo_table=notice&wr_id=123
"""

import time
import re
from typing import List, Optional
from collections import OrderedDict
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
CATEGORY = "COLLEGE_ENGINEERING"
DEPARTMENT = "화학공학과"
LIST_URL = "https://cheme.uos.ac.kr/bbs/board.php"
LIST_PARAMS = {"bo_table": "notice"}

REQUEST_SLEEP = 1.0
RECENT_WINDOW = 50
MAX_PAGES = 20

# =========================
# 목록 수집
# =========================

def collect_recent_post_ids(limit: int = RECENT_WINDOW, max_pages: int = MAX_PAGES) -> List[int]:
    """화학공학과 공지사항 목록에서 게시물 ID (wr_id) 수집"""
    headers = {'User-Agent': 'Mozilla/5.0'}
    collected = []
    seen = set()

    for page in range(1, max_pages + 1):
        # URL 구성 (쿼리 파라미터)
        params = {"bo_table": "notice", "page": page}

        # 요청
        try:
            r = requests.get(LIST_URL, params=params, headers=headers, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
        except Exception as e:
            log(f"❌ [{DEPARTMENT}] 목록 요청 실패 page={page}: {e}")
            break

        if r.status_code != 200:
            log(f"❌ [{DEPARTMENT}] 목록 HTTP {r.status_code} page={page}")
            break

        soup = BeautifulSoup(r.text, 'html.parser')

        # ID 추출 (wr_id 파라미터)
        page_items = []
        for a in soup.select('a[href*="wr_id="]'):
            href = a.get('href', '')
            m = re.search(r'wr_id=(\d+)', href)
            if m:
                post_id = int(m.group(1))
                page_items.append(post_id)

        # 중복 제거 (순서 유지)
        page_items = list(OrderedDict.fromkeys(page_items))

        new_cnt = 0
        for post_id in page_items:
            if post_id not in seen:
                seen.add(post_id)
                collected.append(post_id)
                new_cnt += 1
                if len(collected) >= limit:
                    log(f"[{DEPARTMENT}] 목록 수집 완료: {len(collected)}개")
                    return collected

        log(f"[{DEPARTMENT}] 페이지 {page}: {len(page_items)}개 발견, {new_cnt}개 신규 (누적: {len(collected)}개)")

        # 신규 항목이 없으면 종료
        if new_cnt == 0:
            log(f"[{DEPARTMENT}] 페이지 {page}에서 신규 항목 없음 → 수집 종료")
            break

        time.sleep(0.2)

    log(f"[{DEPARTMENT}] 총 {len(collected)}개 수집 완료")
    return collected


# =========================
# 상세 페이지 파싱
# =========================

def fetch_notice_html(post_id: int) -> Optional[str]:
    """게시물 HTML 가져오기"""
    params = {"bo_table": "notice", "wr_id": post_id}
    headers = {'User-Agent': 'Mozilla/5.0'}

    try:
        r = requests.get(LIST_URL, params=params, headers=headers, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
        if r.status_code != 200:
            log(f"❌ [{DEPARTMENT}] 상세 요청 실패 wr_id={post_id}, status={r.status_code}")
            return None
        return r.text
    except Exception as e:
        log(f"❌ [{DEPARTMENT}] 요청 실패 wr_id={post_id}: {e}")
        return None


def parse_notice_fields(html: str, post_id: int) -> Optional[dict]:
    """HTML 파싱하여 공지사항 필드 추출"""
    soup = BeautifulSoup(html, 'html.parser')

    # 제목 추출
    title = ""
    title_el = soup.select_one('#bo_v_title .bo_v_tit')
    if not title_el:
        title_el = soup.select_one('#bo_v_title')
    if title_el:
        title = title_el.get_text(' ', strip=True)

    if not title:
        log(f"[{DEPARTMENT}] wr_id={post_id}: 제목 없음 (게시물 없거나 삭제됨)")
        return None

    # 본문 추출
    content_text = ""
    content_el = soup.select_one('#bo_v_atc')
    if not content_el:
        content_el = soup.select_one('.board_view')
        if not content_el:
            content_el = soup.select_one('.view_content')
            if not content_el:
                content_el = soup.select_one('#bo_v')

    if content_el:
        content_text = content_el.get_text('\n', strip=True)

    if not content_text:
        content_text = extract_main_text_from_html(html)

    # 날짜/조회수 추출
    date_text = ""
    view_count = 0

    info_el = soup.select_one('#bo_v_info')
    if not info_el:
        info_el = soup.select_one('.bo_v_info')
        if not info_el:
            info_el = soup.select_one('.view_info')
            if not info_el:
                info_el = soup.select_one('.board_view .info')

    if info_el:
        text = info_el.get_text(' ', strip=True)

        # 날짜 추출: yy-mm-dd 형식
        m_date = re.search(r'(?<!\d)(?P<yy>\d{2})-(?P<mm>\d{2})-(?P<dd>\d{2})(?!\d)', text)
        if m_date:
            yy = int(m_date['yy'])
            mm = int(m_date['mm'])
            dd = int(m_date['dd'])
            yyyy = 2000 + yy
            date_text = f"{yyyy:04d}-{mm:02d}-{dd:02d}"

    # 조회수 추출 (아이콘으로 찾기)
    view_icon_el = soup.select_one('strong > i.fa-eye')
    if view_icon_el:
        strong_el = view_icon_el.find_parent('strong')
        if strong_el:
            raw_text = strong_el.get_text(strip=True)
            m = re.search(r'\d+', raw_text)
            if m:
                view_count = int(m.group())

    if not date_text:
        date_text = datetime.now().strftime("%Y-%m-%d")

    return {
        "title": title,
        "department": DEPARTMENT,
        "posted_date": date_text,
        "post_number": post_id,
        "content_text": content_text,
        "view_count": view_count,
    }


# =========================
# 한 건 처리
# =========================

def process_one(post_id: int) -> str:
    """화학공학과 공지사항 한 건 처리"""

    # 1) HTML 가져오기
    html = fetch_notice_html(post_id)
    if not html:
        log(f"⚠️ [{DEPARTMENT}] wr_id={post_id}: HTML 로드 실패 → 스킵")
        return "skipped_error"

    # 2) 파싱
    parsed = parse_notice_fields(html, post_id)
    if not parsed:
        log(f"[{DEPARTMENT}] wr_id={post_id}: 게시물 없음")
        return "not_found"

    post_number = parsed["post_number"]
    title = parsed["title"]
    posted_date = parsed["posted_date"]
    view_count = parsed.get("view_count", 0)

    # 3) 링크 생성
    from urllib.parse import urlencode
    params = {"bo_table": "notice", "wr_id": post_id}
    crawl_link = f"{LIST_URL}?{urlencode(params)}"
    db_link = crawl_link

    # 4) post_number 정규화
    normalized_post_number = _normalize_post_number(post_number)

    # 5) 중복 체크
    prev_dt_raw = get_existing_posted_date(CATEGORY, post_number)
    prev_dt = _ymd(prev_dt_raw)
    curr_dt = _ymd(posted_date)

    if prev_dt:
        if prev_dt == curr_dt:
            # 날짜 동일 → 조회수만 업데이트
            log(f"[{DEPARTMENT}] wr_id={post_id} 이미 존재 → 조회수만 업데이트")
            update_view_count(CATEGORY, normalized_post_number, view_count)
            log(f"✅ [{DEPARTMENT}] 조회수 업데이트: wr_id={post_id}, viewCount={view_count}")
            return "stored"
        else:
            log(f"[{DEPARTMENT}] wr_id={post_id} 날짜 변경 {prev_dt} → {curr_dt}, 업데이트 진행")

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
        log(f"↳ [{DEPARTMENT}] wr_id={post_id}: 이미지 캡처 실패 → 스킵")
        return "skipped_error"

    # 8) 텍스트 + 이미지 요약
    summary = summarize_with_text_and_images(html_text, imgs)
    if not summary:
        log(f"↳ [{DEPARTMENT}] wr_id={post_id}: 요약 실패 → 스킵")
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
        log(f"✅ [{DEPARTMENT}] 저장 완료: wr_id={post_id}, title={title}, posted_date={posted_date}")
        return "stored"
    except MySQLError as e:
        log(f"❌ [{DEPARTMENT}] DB 저장 실패: {e.__class__.__name__}: {e}")
        return "skipped_error"


# =========================
# 메인 실행
# =========================

def main():
    """화학공학과 크롤러 메인"""
    log(f"==== [{DEPARTMENT}] 크롤링 시작 ====")

    # 1) 목록 수집
    post_ids = collect_recent_post_ids(limit=RECENT_WINDOW, max_pages=MAX_PAGES)

    if not post_ids:
        log(f"⚠️ [{DEPARTMENT}]: 목록에서 게시물을 찾지 못해 종료")
        return 0

    log(f"==== [{DEPARTMENT}] {len(post_ids)}개 수집됨 ====")

    # 2) 각 게시물 처리 (역순: 오래된 것부터)
    for post_id in reversed(post_ids):
        process_one(post_id)
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
