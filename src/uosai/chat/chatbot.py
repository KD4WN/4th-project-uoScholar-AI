"""
스트리밍 API 테스트 스크립트 (멀티턴 대화 히스토리 테스트 포함)
"""
import requests
import json
import sys
import time
from typing import List, Dict, Optional

def test_streaming(query: str, conversation_history: Optional[List[Dict]] = None, port: int = 8000):
    """스트리밍 API 테스트 (시간 측정 + 히스토리 지원)"""

    url = f"http://localhost:{port}/chat/stream"

    if conversation_history is None:
        conversation_history = []

    payload = {
        "query": query,
        "conversation_history": conversation_history
    }

    print(f"질문: {query}")
    if conversation_history:
        print(f"히스토리: {len(conversation_history)}개 메시지")
    print("-" * 50)
    print("응답 (실시간):")
    print()

    # 시간 측정 변수
    start_time = time.time()
    first_response_time = None
    first_token_time = None
    end_time = None

    try:
        response = requests.post(
            url,
            json=payload,
            stream=True,
            timeout=60
        )

        if response.status_code != 200:
            print(f"에러: HTTP {response.status_code}")
            print(response.text)
            return None, None

        full_response = ""
        notice_data = None
        turn = 0
        status_messages = []

        # 스트리밍 데이터 읽기
        for line in response.iter_lines():
            if line:
                # 첫 응답 시간 기록
                if first_response_time is None:
                    first_response_time = time.time()

                line_str = line.decode('utf-8')

                if line_str.startswith('data: '):
                    data_str = line_str[6:]

                    try:
                        data = json.loads(data_str)

                        if data['type'] == 'status':
                            status_msg = data['content']
                            if status_msg == 'searching':
                                status_messages.append(f"🔍 검색 중...")
                            elif status_msg == 'found':
                                status_messages.append(f"✅ 공지 발견!")

                        elif data['type'] == 'token':
                            # 첫 토큰 시간 기록
                            if first_token_time is None:
                                first_token_time = time.time()
                                # 상태 메시지가 있었다면 표시
                                if status_messages:
                                    print(f"[{' → '.join(status_messages)}]\n", flush=True)

                            # 실시간으로 토큰 출력
                            print(data['content'], end='', flush=True)
                            full_response += data['content']

                        elif data['type'] == 'content':
                            # 전체 내용 출력
                            if first_token_time is None:
                                first_token_time = time.time()
                            print(data['content'])
                            full_response = data['content']

                        elif data['type'] == 'done':
                            # 완료 신호
                            turn = data['turn']
                            notice_data = data['notice']
                            end_time = time.time()
                            print()

                        elif data['type'] == 'error':
                            print(f"\n❌ 에러: {data['content']}")
                            end_time = time.time()

                    except json.JSONDecodeError as e:
                        print(f"\nJSON 파싱 에러: {e}")

        print()
        print("-" * 50)

        # 시간 측정 결과 출력
        print("\n⏱️  [시간 측정 결과]")
        if first_response_time:
            print(f"  첫 응답 시작: {(first_response_time - start_time):.3f}초")
        if first_token_time:
            print(f"  첫 토큰 도착: {(first_token_time - start_time):.3f}초")
        if end_time:
            print(f"  전체 완료:    {(end_time - start_time):.3f}초")

        print(f"\nTurn: {turn}")

        if notice_data:
            print("\n[찾은 공지사항]")
            print(f"제목: {notice_data.get('title', 'N/A')}")
            print(f"주관: {notice_data.get('department', 'N/A')}")
            print(f"게시일: {notice_data.get('posted_date', 'N/A')}")
            print(f"링크: {notice_data.get('link', 'N/A')}")
            print(f"점수: {notice_data.get('score', 'N/A'):.3f}")
        else:
            print("\n[공지사항 없음]")

        return full_response, notice_data

    except requests.exceptions.RequestException as e:
        print(f"요청 에러: {e}")
        return None, None
    except KeyboardInterrupt:
        print("\n\n중단됨")
        return None, None

def test_multi_turn_conversation(port: int = 8000):
    """멀티턴 대화 테스트"""

    print("=" * 70)
    print("🔄 멀티턴 대화 테스트 (히스토리 누적)")
    print("=" * 70)
    print()

    # 대화 시나리오
    turns = [
        ("25학년도 계절학기 신청 일정 알려줘", "턴1: 학과 지정"),
        ("안녕", "턴2: 구체화 - "),
        ("24학년도 계절학기 신청 일정도 알려줘", "턴3: 구체화 - 시기")
    ]

    conversation_history = []

    for i, (query, description) in enumerate(turns, 1):
        print(f"\n{'='*70}")
        print(f"📝 {description}")
        print(f"{'='*70}\n")

        response, notice = test_streaming(query, conversation_history, port)

        if response:
            # 대화 히스토리에 추가
            conversation_history.append({
                "role": "user",
                "content": query,
                "timestamp": None
            })
            conversation_history.append({
                "role": "assistant",
                "content": response,
                "timestamp": None
            })

        print()

    print("\n" + "="*70)
    print("✅ 멀티턴 대화 테스트 완료")
    print(f"총 {len(conversation_history)//2}개 턴 진행")
    print("="*70)

if __name__ == "__main__":
    print("=" * 70)
    print("🚀 스트리밍 API 테스트")
    print("=" * 70)
    print()

    # 포트 선택 (기본: 8001 - fast 버전)
    port = 8000
    if len(sys.argv) > 1 and sys.argv[1] == "--normal":
        port = 8000
        print("📌 기존 버전 테스트 (포트 8000)\n")
    else:
        print("📌 빠른 버전 테스트 (포트 8001)\n")
        print("   기존 버전 테스트: python test_streaming.py --normal\n")

    # 테스트 모드 선택
    if len(sys.argv) > 1 and "--multi" in sys.argv:
        # 멀티턴 테스트
        test_multi_turn_conversation(port)
    elif len(sys.argv) > 1 and sys.argv[-1] not in ["--normal", "--multi"]:
        # 커스텀 쿼리
        test_query = " ".join([arg for arg in sys.argv[1:] if not arg.startswith("--")])
        print("=" * 70)
        print("💬 단일 쿼리 테스트")
        print("=" * 70)
        print()
        test_streaming(test_query, port=port)
    else:
        # 기본 테스트
        print("=" * 70)
        print("💬 단일 쿼리 테스트")
        print("=" * 70)
        print()
        test_query = "마스터즈에 대해 알려줘"
        test_streaming(test_query, port=port)

        print("\n\n")

        # 멀티턴 테스트 자동 실행
        test_multi_turn_conversation(port)

    print("\n\n" + "="*70)
    print("📋 사용법")
    print("="*70)
    print("""
기본 테스트:
  python test_streaming.py

멀티턴만 테스트:
  python test_streaming.py --multi

커스텀 질문:
  python test_streaming.py "장학금 정보 알려줘"

기존 버전 테스트:
  python test_streaming.py --normal

기존 버전 멀티턴:
  python test_streaming.py --normal --multi
""")
