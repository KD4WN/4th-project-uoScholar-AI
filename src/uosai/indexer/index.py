# src/uosai/indexer/index.py
import os, sys, time, traceback
from datetime import datetime, timedelta

# 공통 유틸
from uosai.common.utils import fetch_all_rows, fetch_rows_since, row_to_doc, split_docs, upsert_docs

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "200"))
BATCH_SLEEP_SEC = float(os.getenv("BATCH_SLEEP_SEC", "0.8"))  # 레이트리밋 대응

# 증분 업데이트 설정
INDEX_MODE = os.getenv("INDEX_MODE", "incremental")  # "incremental" 또는 "full"
INCREMENTAL_DAYS = int(os.getenv("INCREMENTAL_DAYS", "7"))  # 증분 업데이트 시 최근 N일

def log(msg: str) -> None:
    print(f"[indexer {datetime.now():%Y-%m-%d %H:%M:%S}] {msg}")

def main() -> int:
    """인덱싱 메인 함수

    환경 변수:
        INDEX_MODE: "incremental" (증분 업데이트, 기본값) 또는 "full" (전체 리빌드)
        INCREMENTAL_DAYS: 증분 업데이트 시 최근 N일 데이터 처리 (기본값: 7)

    WUs 절약:
        - incremental: 최근 N일 데이터만 upsert (일일 약 30-300 WUs)
        - full: 전체 삭제 후 재삽입 (약 6,000 WUs, 월 1회 권장)
    """
    mode = INDEX_MODE.lower()

    if mode == "full":
        log("=== FULL REBUILD MODE ===")
        log("WARNING: This will consume significant WUs!")
        rows = fetch_all_rows()
        rebuild = True
    elif mode == "incremental":
        log(f"=== INCREMENTAL UPDATE MODE (last {INCREMENTAL_DAYS} days) ===")
        since_date = (datetime.now() - timedelta(days=INCREMENTAL_DAYS)).strftime("%Y-%m-%d")
        log(f"Fetching notices since: {since_date}")
        rows = fetch_rows_since(since_date)
        rebuild = False
    else:
        log(f"ERROR: Invalid INDEX_MODE={INDEX_MODE}. Use 'incremental' or 'full'")
        return 1

    if not rows:
        log("No rows found")
        return 0

    docs = split_docs([row_to_doc(r) for r in rows])
    log(f"Rows={len(rows)} → Chunks={len(docs)}")

    total = 0
    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i:i+BATCH_SIZE]
        # full 모드일 때만 첫 배치에서 전체 삭제
        should_rebuild = rebuild and (i == 0)
        n = upsert_docs(batch, rebuild=should_rebuild)
        total += n
        log(f"Upsert batch {i//BATCH_SIZE+1}: {n} chunks (cum {total})")
        if i + BATCH_SIZE < len(docs) and BATCH_SLEEP_SEC > 0:
            time.sleep(BATCH_SLEEP_SEC)

    mode_label = "Full rebuild" if rebuild else "Incremental update"
    log(f"{mode_label} done: chunks={total}")
    return total

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)
