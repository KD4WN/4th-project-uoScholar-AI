# chatbot.py - 대화형 공지 추천 챗봇
import os
import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
from math import sqrt
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
from pinecone import Pinecone
import cohere

# ===== 환경 설정 =====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "uos-notices")
PINECONE_NS = os.getenv("PINECONE_NAMESPACE")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# 대화 설정
MAX_CONVERSATION_TURNS = None  # 무제한 대화 턴
TOP_K = int(os.getenv("TOP_K", "12"))

# 검색 설정 (Cohere Reranker 사용)
USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"
INITIAL_SEARCH_K = int(os.getenv("INITIAL_SEARCH_K", "20"))  # 초기 검색 개수 (속도 최적화)
FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", "5"))  # Reranker 후 최종 개수
RERANK_THRESHOLD = float(os.getenv("RERANK_THRESHOLD", "0.1"))  # Reranker 점수 임계값 (더 관대하게)

# 입력 검증 설정
MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", "500"))  # 쿼리 최대 길이
MAX_CONVERSATION_HISTORY = int(os.getenv("MAX_CONVERSATION_HISTORY", "20"))  # 최대 대화 턴 수
MAX_MESSAGE_LENGTH = int(os.getenv("MAX_MESSAGE_LENGTH", "2000"))  # 개별 메시지 최대 길이

# API 타임아웃 설정 (초 단위)
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "30"))  # LLM API 타임아웃
VECTORSTORE_TIMEOUT = int(os.getenv("VECTORSTORE_TIMEOUT", "10"))  # 벡터 검색 타임아웃
COHERE_TIMEOUT = int(os.getenv("COHERE_TIMEOUT", "15"))  # Cohere Reranker 타임아웃

# ===== 전역 객체 =====
_llm = None
_embeddings = None
_vectorstore = None
_pc_client = None
_cohere_client = None


# 불용어 설정
STOPWORDS = {
    "공지","안내","프로그램","워크숍","행사","공지사항","공지요","문의","신청",
    "관련","관련된","있어","있나요","혹시","좀","요","거","것","같아","싶어","겨","부터","까지"
}

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# ===== 입력 검증 =====
class ValidationError(Exception):
    """입력 검증 실패 시 발생하는 예외"""
    pass

def validate_input(query: str, conversation_history: List[Dict[str, str]]) -> None:
    """입력값 검증 (DoS 공격 방지, 프롬프트 인젝션 방지)"""

    # 1. 쿼리 길이 검증
    if not query or not query.strip():
        raise ValidationError("질문 내용이 비어있습니다.")

    if len(query) > MAX_QUERY_LENGTH:
        raise ValidationError(f"질문이 너무 깁니다. (최대 {MAX_QUERY_LENGTH}자)")

    # 2. 대화 내역 개수 제한
    if len(conversation_history) > MAX_CONVERSATION_HISTORY:
        raise ValidationError(f"대화 내역이 너무 깁니다. (최대 {MAX_CONVERSATION_HISTORY}턴)")

    # 3. 개별 메시지 길이 검증
    for i, msg in enumerate(conversation_history):
        content = msg.get("content", "")
        if len(content) > MAX_MESSAGE_LENGTH:
            raise ValidationError(f"대화 내역의 {i+1}번째 메시지가 너무 깁니다. (최대 {MAX_MESSAGE_LENGTH}자)")

    # 4. Role 검증 (user 또는 assistant만 허용)
    for i, msg in enumerate(conversation_history):
        role = msg.get("role", "")
        if role not in ["user", "assistant"]:
            raise ValidationError(f"올바르지 않은 role 값입니다: {role}")

    # 5. 프롬프트 인젝션 패턴 감지 (기본적인 방어)
    suspicious_patterns = [
        "ignore previous instructions",
        "ignore all previous",
        "disregard previous",
        "you are now",
        "new instructions:",
        "system:",
        "assistant:",
        "<|im_start|>",
        "<|im_end|>",
        "```python",  # 코드 실행 시도
    ]

    combined_text = query.lower() + " " + " ".join([m.get("content", "").lower() for m in conversation_history])

    for pattern in suspicious_patterns:
        if pattern in combined_text:
            logging.warning(f"Suspicious pattern detected: {pattern}")
            # 로그만 남기고 차단하지는 않음 (false positive 방지)

    # 6. 반복 문자 패턴 감지 (DoS 공격)
    if len(set(query)) < 5 and len(query) > 50:  # 5개 미만의 고유 문자로 50자 이상
        raise ValidationError("비정상적인 입력 패턴이 감지되었습니다.")

def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=CHAT_MODEL,
            temperature=0.1,
            api_key=OPENAI_API_KEY,
            request_timeout=LLM_TIMEOUT,  # 타임아웃 설정
            max_retries=2  # 최대 재시도 횟수
        )
    return _llm

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(
            model=EMBED_MODEL,
            api_key=OPENAI_API_KEY
        )
        logging.info("OpenAI embedding model loaded: %s", EMBED_MODEL)
    return _embeddings

def get_cohere_client():
    """Cohere 클라이언트 초기화"""
    global _cohere_client
    if _cohere_client is None and COHERE_API_KEY:
        _cohere_client = cohere.Client(COHERE_API_KEY)
        logging.info("Cohere client initialized")
    return _cohere_client

def get_vectorstore():
    global _vectorstore, _pc_client
    if _vectorstore is None:
        if _pc_client is None:
            _pc_client = Pinecone(api_key=PINECONE_API_KEY)
            logging.info("Pinecone client initialized")

        _vectorstore = PineconeVectorStore.from_existing_index(
            index_name=PINECONE_INDEX,
            embedding=get_embeddings(),
            namespace=PINECONE_NS
        )
        logging.info("PineconeVectorStore ready: %s (ns=%s)", PINECONE_INDEX, PINECONE_NS)
    return _vectorstore


# ===== 대화 요구사항 분석 =====
def extract_requirements(conversation_history: List[Dict[str, str]]) -> Dict[str, Any]:
    """대화 히스토리에서 사용자 요구사항 추출"""

    # 모든 사용자 메시지를 합쳐서 분석
    user_messages = [msg["content"] for msg in conversation_history if msg["role"] == "user"]
    full_conversation = " ".join(user_messages)

    prompt = f"""다음 대화에서 사용자가 찾고 있는 공지사항의 요구사항을 JSON 형태로 추출해주세요.

대화 내용: {full_conversation}

먼저, 이 대화가 "대학 공지사항을 찾는 질문"인지 판단하세요.
- 공지사항 관련: 장학금, 학사일정, 수강신청, 취업, 행사, 프로그램, 신청, 모집 등
- 공지사항 무관: 인사말, 잡담, 일상 대화, 공지와 관련 없는 질문

다음 형태의 JSON으로 응답해주세요:
{{
    "is_notice_related": true 또는 false,
    "category": "장학금|학사일정|수강신청|취업|행사|기타",
    "keywords": ["핵심키워드1", "핵심키워드2", ...],
    "target_audience": "학부생|대학원생|전체|특정학과",
    "urgency": "높음|보통|낮음",
    "specific_requirements": "구체적인 요구사항 요약"
}}

JSON만 응답하세요:"""

    llm = get_llm()
    response = llm.invoke(prompt)

    try:
        import json
        # JSON 응답에서 불필요한 텍스트 제거
        content = response.content.strip()
        # JSON 부분만 추출 (```json으로 감싸진 경우 처리)
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            content = content[start:end].strip()
        elif "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            content = content[start:end].strip()

        requirements = json.loads(content)

        # is_notice_related 필드 확인 (없으면 기본값 true)
        if "is_notice_related" not in requirements:
            requirements["is_notice_related"] = True

        # 필수 필드 검증 및 보완
        if not requirements.get("keywords"):
            # 키워드가 없으면 사용자 메시지에서 추출
            important_words = []
            for msg in user_messages:
                words = [w for w in msg.split() if len(w) > 1 and w not in STOPWORDS]
                important_words.extend(words[:2])  # 각 메시지에서 2개씩
            requirements["keywords"] = important_words[:5]  # 최대 5개

        if not requirements.get("specific_requirements"):
            requirements["specific_requirements"] = full_conversation

        return requirements

    except Exception as parse_error:
        logging.warning("JSON parsing failed: %s", parse_error)
        # JSON 파싱 실패 시 더 정교한 기본값 생성
        important_words = []
        for msg in user_messages:
            # 불용어 제거하고 의미있는 단어 추출
            words = [w for w in msg.split() if len(w) > 1 and w not in STOPWORDS]
            important_words.extend(words)

        # 중복 제거하고 상위 5개 선택
        unique_keywords = list(dict.fromkeys(important_words))[:5]

        return {
            "is_notice_related": True,  # 파싱 실패 시 기본값
            "category": "기타",
            "keywords": unique_keywords,
            "target_audience": "전체",
            "urgency": "보통",
            "specific_requirements": full_conversation[:200]  # 너무 길지 않게 제한
        }

# ===== 일반 대화 응답 생성 =====
def generate_casual_response(conversation_history: List[Dict[str, str]]) -> str:
    """공지사항과 관련 없는 질문에 대한 자연스러운 응답"""

    recent_conversation = ' '.join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-4:]])

    prompt = f"""당신은 서울시립대학교 공지사항 안내 챗봇입니다.

지금까지의 대화:
{recent_conversation}

사용자가 공지사항과 관련 없는 말을 했습니다. 다음 지침에 따라 응답하세요:

1. 인사말이면 친근하게 인사하고, 공지사항 관련 도움을 줄 수 있다고 안내
2. 일상 대화나 잡담이면 가볍게 응대하고, 공지사항이 필요하면 도와줄 수 있다고 안내
3. 챗봇이 할 수 없는 질문이면 정중히 범위를 설명하고, 공지사항 관련 질문을 유도
4. 2-3문장으로 간결하게 응답할 것
5. 너무 딱딱하지 않고 친근한 톤으로 작성할 것

자연스럽고 친근한 응답을 작성해주세요:"""

    llm = get_llm()
    response = llm.invoke(prompt)
    return response.content.strip()

# ===== 질문 생성 =====
def generate_clarifying_question(turn: int, conversation_history: List[Dict[str, str]]) -> str:
    """대화 턴에 관계없이 상황에 맞는 명확화 질문 생성"""

    user_messages = [msg["content"] for msg in conversation_history if msg["role"] == "user"]

    # 전체 대화 맥락을 고려한 질문 생성
    recent_conversation = ' '.join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-6:]])

    prompt = f"""지금까지의 대화:
{recent_conversation}

현재까지의 대화를 바탕으로 사용자가 원하는 공지사항을 더 구체적으로 파악하기 위한 질문을 생성해주세요.

조건:
- 1문장으로만 작성할 것
- 너무 공식적이지 않고, 자연스럽고 친근한 톤으로 질문할 것
- 필요하다면 카테고리, 대상, 시기, 추가 정보 중 하나만 골라 질문할 것
- 불필요하게 여러 항목을 한 번에 묻지 말 것

다음과 같은 방향으로 질문할 수 있습니다:
1. 카테고리 확인 (장학금, 학사일정, 수강신청, 취업정보, 행사 등)
2. 대상 확인 (학부생, 대학원생, 특정 학과)
3. 시기 확인 (언제 필요한지)
4. 추가 정보 요청

1,2,3,4번의 내용 중 스스로 판단하여 적절한 질문을 생성해주세요. 모든 항목이 포함될 필요는 없습니다.

친근하고 자연스러운 톤으로 1문장으로 질문해주세요:"""

    llm = get_llm()
    response = llm.invoke(prompt)
    return response.content.strip()


def cosine_sim(a, b, eps: float = 1e-10) -> float:
    """코사인 유사도 계산"""
    dot = sum(x*y for x, y in zip(a, b))
    na = sqrt(sum(x*x for x in a)) + eps
    nb = sqrt(sum(y*y for y in b)) + eps
    return dot / (na * nb)



# ===== 최종 공지 추천 (고급 검색 로직 적용) =====
def rerank_documents(query: str, docs: List[Document]) -> List[Tuple[Document, float]]:
    """입력 쿼리와 문서들을 Cohere Reranker로 재정렬 (날짜 정보 포함)"""

    if not docs or not USE_RERANKER:
        # Reranker 비활성화 시 기본 코사인 유사도 사용 (배치 처리로 최적화)
        embeddings = get_embeddings()
        query_embedding = embeddings.embed_query(query)

        # 배치로 문서 임베딩 계산 (더 빠름)
        doc_texts = [doc.page_content for doc in docs]
        doc_embeddings = embeddings.embed_documents(doc_texts)

        results = []
        for doc, doc_embedding in zip(docs, doc_embeddings):
            score = cosine_sim(query_embedding, doc_embedding)
            results.append((doc, float(score)))

        return sorted(results, key=lambda x: x[1], reverse=True)

    cohere_client = get_cohere_client()
    if not cohere_client:
        logging.warning("Cohere client not available, using cosine similarity")
        # 직접 코사인 유사도 계산 (재귀 호출 방지)
        embeddings = get_embeddings()
        query_embedding = embeddings.embed_query(query)

        # 배치로 문서 임베딩 계산 (더 빠름)
        doc_texts = [doc.page_content for doc in docs]
        doc_embeddings = embeddings.embed_documents(doc_texts)

        results = []
        for doc, doc_embedding in zip(docs, doc_embeddings):
            score = cosine_sim(query_embedding, doc_embedding)
            results.append((doc, float(score)))

        return sorted(results, key=lambda x: x[1], reverse=True)

    try:
        # 현재 날짜 정보를 쿼리에 추가
        now = datetime.now()
        enhanced_query = f"{query} (현재 날짜: {now.strftime('%Y년 %m월')})"

        # 문서 텍스트 준비 (날짜 정보 포함)
        documents = []
        for doc in docs:
            metadata = doc.metadata or {}
            title = metadata.get("title", "")
            content = doc.page_content[:500]  # 너무 긴 문서 제한
            posted_date = metadata.get("posted_date", "")

            # 날짜 정보를 텍스트에 명시
            if posted_date:
                date_info = f"[게시일: {posted_date}]"
            else:
                date_info = ""

            # 날짜 + 제목 + 내용 결합
            if title:
                combined_text = f"{date_info} {title}\n{content}"
            else:
                combined_text = f"{date_info} {content}"

            documents.append(combined_text)

        # Cohere Rerank API 호출
        response = cohere_client.rerank(
            model="rerank-multilingual-v3.0",  # 다국어 지원 모델
            query=enhanced_query,  # 날짜 정보가 포함된 쿼리
            documents=documents,
            top_n=min(FINAL_TOP_K, len(docs))
        )

        # 결과 처리
        reranked_results = []
        for result in response.results:
            original_doc = docs[result.index]
            relevance_score = result.relevance_score
            reranked_results.append((original_doc, float(relevance_score)))

        logging.info("[RERANK] Reranked %d documents, top score: %.3f",
                    len(reranked_results),
                    reranked_results[0][1] if reranked_results else 0.0)

        return reranked_results

    except Exception as e:
        logging.error("Cohere reranking failed: %s", e)
        # Fallback to cosine similarity (직접 계산으로 재귀 방지)
        try:
            embeddings = get_embeddings()
            query_embedding = embeddings.embed_query(query)

            # 배치로 문서 임베딩 계산
            doc_texts = [doc.page_content for doc in docs]
            doc_embeddings = embeddings.embed_documents(doc_texts)

            results = []
            for doc, doc_embedding in zip(docs, doc_embeddings):
                score = cosine_sim(query_embedding, doc_embedding)
                results.append((doc, float(score)))

            return sorted(results, key=lambda x: x[1], reverse=True)
        except Exception as fallback_error:
            logging.error("Fallback cosine similarity also failed: %s", fallback_error)
            return []

def find_best_notice(requirements: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """요구사항을 바탕으로 가장 적합한 공지 1개 추천 (Cohere Reranker 사용)"""

    try:
        # 1) 검색 쿼리 생성 - specific_req 직접 사용 (clean 과정 제거)
        specific_req = requirements.get('specific_requirements', '')

        # specific_req를 그대로 사용 (가장 정확한 쿼리)
        if specific_req and specific_req.strip():
            search_query = specific_req.strip()
        else:
            # fallback: keywords 사용
            keywords = requirements.get('keywords', [])
            if keywords:
                search_query = ' '.join(keywords)
            else:
                search_query = ""

        logging.info("[RECOMMEND] specific_req=%s", specific_req[:100])

        if not search_query:
            logging.warning("[RECOMMEND] Empty search query")
            return None

        logging.info("[RECOMMEND] search_query=%r", search_query)

        # 2) 벡터스토어 검증
        try:
            vectorstore = get_vectorstore()
            if not vectorstore:
                logging.error("[RECOMMEND] Vectorstore not available")
                return None
        except Exception as vs_error:
            logging.error("[RECOMMEND] Vectorstore initialization failed: %s", vs_error)
            return None

        # 3) 초기 코사인 유사도 검색
        try:
            docs = vectorstore.similarity_search(
                search_query,
                k=INITIAL_SEARCH_K
            )
        except Exception as search_error:
            logging.error("[RECOMMEND] Similarity search failed: %s", search_error)
            return None

        if not docs:
            logging.info("[RECOMMEND] No documents found for query")
            return None

        # 4) Cohere Reranker로 재정렬 (안전한 호출)
        try:
            reranked_results = rerank_documents(search_query, docs)
        except Exception as rerank_error:
            logging.error("[RECOMMEND] Reranking failed: %s", rerank_error)
            return None

        if not reranked_results:
            logging.info("[RECOMMEND] No results after reranking")
            return None

        # 5) 최고 점수 문서 선택
        best_doc, best_score = reranked_results[0]

        # 상위 결과들 로깅 (디버깅용)
        logging.info("[RECOMMEND] Top 3 results:")
        for i, (doc, score) in enumerate(reranked_results[:3]):
            title = doc.metadata.get("title", "제목없음")[:50]
            logging.info("  %d. Score: %.3f | Title: %s", i+1, score, title)

        # 6) 임계값 확인
        if best_score < RERANK_THRESHOLD:
            logging.info("[RECOMMEND] Score below threshold: %.3f < %.3f", best_score, RERANK_THRESHOLD)
            return None

        # 7) 결과 구성 (안전한 메타데이터 추출)
        try:
            metadata = best_doc.metadata or {}
            content = best_doc.page_content or ""

            return {
                "content": content,
                "score": float(best_score),
                "title": metadata.get("title", "제목 없음"),
                "link": metadata.get("link", ""),
                "posted_date": metadata.get("posted_date", ""),
                "department": metadata.get("department", ""),
                "category": metadata.get("category", ""),
                "doc_id": metadata.get("doc_id", "")
            }
        except Exception as result_error:
            logging.error("[RECOMMEND] Result construction failed: %s", result_error)
            return None

    except Exception as e:
        logging.error("[RECOMMEND] Unexpected error in find_best_notice: %s", e, exc_info=True)
        return None

def generate_final_recommendation(requirements: Dict[str, Any], notice: Dict[str, Any], conversation_history: List[Dict[str, str]]) -> str:
    """최종 추천 메시지 생성"""

    conversation_summary = " ".join([msg["content"] for msg in conversation_history if msg["role"] == "user"])

    # 현재 날짜 정보 추가
    now = datetime.now()
    current_date_str = now.strftime("%Y년 %m월 %d일")

    prompt = f"""**현재 날짜: {current_date_str}**

사용자가 다음과 같이 질문했습니다: "{conversation_summary}"

이에 대한 답변으로 적합한 공지사항을 찾았습니다:

**공지사항 정보:**
- 제목: {notice.get('title')}
- 주관: {notice.get('department')}
- 게시일: {notice.get('posted_date')}
- 내용: {notice.get('content')[:1000]}

**지침:**
1. 기간에 대한 답변을 할 경우, **현재 날짜를 고려하여** 모집/신청 기간이 지났는지, 진행 중인지, 예정인지 명확히 알려주세요
2. 사용자의 구체적인 질문에 직접적으로 답변하세요
3. 공지사항의 내용에서 사용자가 궁금해하는 부분을 중점적으로 설명하세요
4. 형식적인 "행사:", "장소:" 같은 구조화된 답변 금지
5. 자연스럽고 대화적인 톤으로 작성하세요
6. 사용자가 알고 싶어하는 핵심 정보(언제, 어디서, 누가, 어떻게)를 자연스럽게 포함하세요
7. 마지막에 링크: {notice.get('link')} 제공

자연스럽고 친근한 답변을 작성해주세요:"""

    llm = get_llm()
    response = llm.invoke(prompt)
    return response.content.strip()

# ===== 앱 초기화 및 모델 프리로딩 =====
def preload_models():
    """앱 시작 시 모든 모델을 미리 로드하여 첫 요청 속도 개선"""
    try:
        logging.info("Preloading models...")
        # 임베딩 모델 로드
        get_embeddings()
        # 벡터스토어 초기화
        get_vectorstore()
        # Cohere 클라이언트 초기화
        get_cohere_client()
        # LLM 초기화
        get_llm()
        logging.info("All models preloaded successfully!")
    except Exception as e:
        logging.error("Error preloading models: %s", e)

def cleanup_resources():
    """앱 종료 시 리소스 정리"""
    global _llm, _embeddings, _vectorstore, _pc_client, _cohere_client
    try:
        logging.info("Cleaning up resources...")
        # 전역 객체 정리
        _llm = None
        _embeddings = None
        _vectorstore = None
        _pc_client = None
        _cohere_client = None
        logging.info("Resources cleaned up successfully!")
    except Exception as e:
        logging.error("Error cleaning up resources: %s", e)

# ===== FastAPI 앱 lifespan =====
@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 실행되는 lifespan 핸들러"""
    # 시작 시
    preload_models()
    yield
    # 종료 시
    cleanup_resources()

# ===== FastAPI 앱 =====
app = FastAPI(
    title="Notice Recommendation Chatbot",
    version="1.0.0",
    lifespan=lifespan
)

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[str] = None

    class Config:
        # Pydantic v2 호환성
        str_strip_whitespace = True

class ChatRequest(BaseModel):
    query: str
    conversation_history: List[ChatMessage] = []  # 클라이언트가 전체 대화 내역 전송

    class Config:
        str_strip_whitespace = True

class ChatResponse(BaseModel):
    response: str
    turn: int
    recommended_notice: Optional[Dict[str, Any]] = None

@app.get("/health")
def health_check():
    """헬스체크"""
    try:
        _ = get_vectorstore()
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """대화형 공지 추천 엔드포인트 (Stateless)"""
    try:
        # 클라이언트로부터 받은 대화 내역을 딕셔너리로 변환
        messages = [msg.dict() for msg in request.conversation_history]

        # 입력 검증
        try:
            validate_input(request.query, messages)
        except ValidationError as ve:
            logging.warning(f"Validation failed: {ve}")
            return ChatResponse(
                response=str(ve),
                turn=0,
                recommended_notice=None
            )

        # 현재 사용자 메시지 추가
        messages.append({
            "role": "user",
            "content": request.query,
            "timestamp": datetime.now().isoformat()
        })

        current_turn = len([m for m in messages if m["role"] == "user"])

        # 매 턴마다 요구사항 추출
        requirements = extract_requirements(messages)

        # 공지사항 관련 질문인지 확인
        is_notice_related = requirements.get("is_notice_related", True)

        if not is_notice_related:
            # 공지사항과 관련 없는 질문: 일반 대화 응답
            casual_response = generate_casual_response(messages)

            return ChatResponse(
                response=casual_response,
                turn=current_turn,
                recommended_notice=None
            )

        # 공지사항 관련 질문: 최적 공지 찾기
        best_notice = find_best_notice(requirements)

        if best_notice:
            # 공지를 찾았을 때: 최종 추천 메시지 생성
            final_response = generate_final_recommendation(requirements, best_notice, messages)

            return ChatResponse(
                response=final_response,
                turn=current_turn,
                recommended_notice=best_notice
            )
        else:
            # 공지를 찾지 못했을 때: 명확화 질문 생성
            clarifying_question = generate_clarifying_question(current_turn, messages)

            return ChatResponse(
                response=clarifying_question,
                turn=current_turn,
                recommended_notice=None
            )

    except Exception as e:
        logging.error(f"Chat error: {e}", exc_info=True)
        return ChatResponse(
            response="죄송합니다. 일시적인 오류가 발생했습니다. 다시 시도해주세요.",
            turn=0,
            recommended_notice=None
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
