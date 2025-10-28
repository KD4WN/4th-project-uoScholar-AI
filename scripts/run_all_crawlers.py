# scripts/run_all_crawlers.py
"""
통합 크롤러 실행 스크립트
모든 크롤러를 순차적으로 실행합니다.

실행 순서:
1. 포털 공통 (일반공지, 학사공지, 단과대학 공지)
2. 화학공학과
3. 생명과학과
4. 경제학부
5. 건축학부
6. 조경학과
"""

import sys
import os
import pathlib
from datetime import datetime

# src 디렉토리를 sys.path에 추가
project_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "src" / "uosai" / "crawler"))

# UTF-8 출력 설정 (Windows 환경)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

def log(msg: str):
    """로그 출력"""
    print(f"[run_all {datetime.now():%Y-%m-%d %H:%M:%S}] {msg}")

def main():
    """모든 크롤러 실행"""
    log("="*80)
    log("통합 크롤러 시작")
    log("="*80)

    # 크롤러 목록 (name, module_name, enabled)
    crawlers = [
        ("포털 공통", "uosai.crawler.portal_crawler", False),
        ("화학공학과", "uosai.crawler.chemical_crawler", False),  # 주석 처리됨
        ("생명과학과", "uosai.crawler.life_science_crawler", False),  # 주석 처리됨
        ("경제학부", "uosai.crawler.economics_crawler", True),
        ("건축학부", "uosai.crawler.architecture_crawler", False),
        ("조경학과", "uosai.crawler.landscape_crawler", False),
    ]

    results = {}

    for name, module_name, enabled in crawlers:
        if not enabled:
            log(f"⏭️  [{name}] 크롤러는 현재 비활성화됨 → 건너뜀")
            results[name] = "비활성화"
            continue

        log(f"\n{'='*80}")
        log(f"[{name}] 크롤러 시작...")
        log(f"{'='*80}")

        try:
            # 동적 import
            parts = module_name.split('.')
            module = __import__(module_name, fromlist=[parts[-1]])
            result = module.main()
            results[name] = "성공" if result == 0 else "실패"
            log(f"[{name}] 크롤러 완료: {results[name]}")

        except Exception as e:
            log(f"❌ [{name}] 크롤러 실행 중 에러: {type(e).__name__}: {e}")
            results[name] = "에러"
            import traceback
            traceback.print_exc()

    # 최종 결과 출력
    log(f"\n{'='*80}")
    log("전체 크롤링 결과")
    log(f"{'='*80}")

    for name, result in results.items():
        if result == "비활성화":
            status_icon = "⏭️ "
        elif result == "성공":
            status_icon = "✅"
        else:
            status_icon = "❌"
        log(f"{status_icon} [{name}]: {result}")

    log(f"{'='*80}")
    log("통합 크롤러 종료")
    log(f"{'='*80}")

    # 활성화된 크롤러 중 하나라도 실패하면 exit code 1
    active_results = {k: v for k, v in results.items() if v != "비활성화"}
    if any(r != "성공" for r in active_results.values()):
        return 1
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        log("\n사용자에 의해 중단됨")
        sys.exit(130)
    except Exception as e:
        log(f"치명적 에러: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
