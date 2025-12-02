# common.py : 공용 유틸 함수 정의 (lazy DB pool)
import os
from typing import List, Dict, Any
from dotenv import load_dotenv; load_dotenv()

from mysql.connector import pooling, Error as MySQLError
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# ===== Helpers =====
def _env_bool(val: str | None, default: bool) -> bool:
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "t", "yes", "y", "on"}

# ===== Embedding / Pinecone Env =====
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX", "uos-notices")

_raw_ns = os.getenv("PINECONE_NAMESPACE", "").strip()
PINECONE_NS = None if _raw_ns == "" else _raw_ns

PINECONE_CLOUD   = os.getenv("PINECONE_CLOUD")   # 최초 생성 시 필요
PINECONE_REGION  = os.getenv("PINECONE_REGION")  # 최초 생성 시 필요

CHUNK_SIZE     = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP  = int(os.getenv("CHUNK_OVERLAP", "150"))
MAX_DOC_LEN    = int(os.getenv("MAX_DOC_LEN", "12000"))

# OpenAI 임베딩 차원
EMBED_DIM = 1536  # text-embedding-3-small

# ===== DB (lazy pool) =====
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "charset": "utf8mb4",
    # 에러 메시지 가시성↑ (C-extension 대신 파이썬 구현)
    "use_pure": _env_bool(os.getenv("DB_USE_PURE"), True),
    "raise_on_warnings": _env_bool(os.getenv("DB_WARNINGS"), True),
    "connection_timeout": 10,
}

_POOL: pooling.MySQLConnectionPool | None = None

def get_pool() -> pooling.MySQLConnectionPool:
    """지연 초기화로 커넥션 풀 생성"""
    global _POOL
    if _POOL is None:
        try:
            _POOL = pooling.MySQLConnectionPool(
                pool_name="ragpool",
                pool_size=5,
                **DB_CONFIG
            )
        except MySQLError as e:
            print(f"[DB POOL INIT ERROR] {type(e).__name__}: {e}")
            print(f"[DB CHECK] host={DB_CONFIG.get('host')} port={DB_CONFIG.get('port')} "
                  f"db={DB_CONFIG.get('database')} user={DB_CONFIG.get('user')} "
                  f"use_pure={DB_CONFIG.get('use_pure')}")
            raise
    return _POOL

def get_conn():
    return get_pool().get_connection()

# ===== DB Queries =====
def fetch_rows_since(since: str) -> List[Dict[str, Any]]:
    sql = """
    SELECT category, post_number, title, link, summary, posted_date, department
    FROM notice
    WHERE title IS NOT NULL AND title <> ''
      AND summary IS NOT NULL AND summary <> ''
      AND posted_date >= %s
    """
    conn = get_conn()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(sql, (since,))
        rows = cur.fetchall()
        cur.close()
        return rows
    finally:
        conn.close()

def fetch_all_rows() -> List[Dict[str, Any]]:
    sql = """
    SELECT category, post_number, title, link, summary, posted_date, department
    FROM notice
    WHERE title IS NOT NULL AND title <> '' AND summary IS NOT NULL AND summary <> ''
    """
    conn = get_conn()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(sql)
        rows = cur.fetchall()
        cur.close()
        return rows
    finally:
        conn.close()

# ===== Doc / Chunk =====
def row_to_doc(row: Dict[str, Any]) -> Document:
    """DB row → LangChain Document (summary = 본문, 나머지 = 메타데이터)"""
    title = (row.get("title") or "").strip()
    link  = row.get("link", "")
    dept  = row.get("department", "")
    date  = str(row.get("posted_date", ""))
    cat   = row.get("category", "")
    pno   = row.get("post_number")

    # summary가 본문
    full = (row.get("summary") or "").strip()
    if len(full) > MAX_DOC_LEN:
        full = full[:MAX_DOC_LEN] + "\n\n[... 본문 일부 생략 ...]"

    return Document(
        page_content=full,
        metadata={
            "title": title,
            "link": link,
            "department": dept,
            "posted_date": date,
            "category": cat,
            "post_number": pno,
        }
    )

def split_docs(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(docs)

# ===== 전역 임베딩 인스턴스 캐싱 =====
_EMBEDDING_INSTANCE = None

def get_embedding_instance():
    """임베딩 인스턴스를 재사용하기 위한 캐싱"""
    global _EMBEDDING_INSTANCE
    if _EMBEDDING_INSTANCE is None:
        _EMBEDDING_INSTANCE = OpenAIEmbeddings(model=EMBED_MODEL)
        print(f"[Vectorstore] Using OpenAI embedding model: {EMBED_MODEL}")
    return _EMBEDDING_INSTANCE

# ===== Pinecone =====
def ensure_pinecone_index(pc: Pinecone, index_name: str, dim: int):
    names = [idx.name for idx in pc.list_indexes()]
    if index_name not in names:
        if not PINECONE_CLOUD or not PINECONE_REGION:
            raise RuntimeError("PINECONE_CLOUD/REGION required for first-time index create.")
        pc.create_index(
            name=index_name,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )

def get_vectorstore() -> PineconeVectorStore:
    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY missing")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    ensure_pinecone_index(pc, PINECONE_INDEX, EMBED_DIM)

    # 캐싱된 임베딩 인스턴스 사용
    embeddings = get_embedding_instance()

    # PINECONE_NS 가 None이면 기본 네임스페이스(__default__) 사용
    return PineconeVectorStore(index_name=PINECONE_INDEX, embedding=embeddings, namespace=PINECONE_NS)


def upsert_docs(docs: List[Document], rebuild: bool = False) -> int:
    """Pinecone에 문서를 upsert합니다.

    Args:
        docs: 업서트할 문서 리스트
        rebuild: True면 전체 삭제 후 재삽입, False면 증분 업데이트 (기본값)

    Returns:
        업서트된 문서 개수

    Note:
        - rebuild=False (증분 모드): 동일 ID 문서는 덮어쓰기, 신규는 삽입 (WUs 절약)
        - rebuild=True (전체 리빌드): 모든 문서 삭제 후 재삽입 (월 1회 권장)
    """
    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY missing")

    pc = Pinecone(api_key=PINECONE_API_KEY)
    ensure_pinecone_index(pc, PINECONE_INDEX, EMBED_DIM)
    vs = get_vectorstore()

    # rebuild=True일 때만 전체 삭제 (WUs 대량 소모)
    if rebuild:
        idx = pc.Index(PINECONE_INDEX)
        ns_repr = "__default__" if PINECONE_NS is None else PINECONE_NS
        print(f"[pinecone] FULL REBUILD: delete_all namespace={ns_repr}")
        try:
            if PINECONE_NS is None:
                idx.delete(delete_all=True)  # 기본 네임스페이스
            else:
                idx.delete(delete_all=True, namespace=PINECONE_NS)
        except Exception as e:
            # 존재하지 않는 네임스페이스면 지울 게 없어서 404가 날 수 있음 — 경고만 출력
            print(f"[pinecone] delete_all warning: {e}")
    else:
        print(f"[pinecone] INCREMENTAL UPDATE: upserting {len(docs)} documents")

    # 업서트 (동일 ID면 덮어쓰기, 신규면 삽입)
    ids = []
    for i, d in enumerate(docs):
        m = d.metadata or {}
        pno = str(m.get("post_number", "none"))
        cat = str(m.get("category", "none"))
        ids.append(f"{cat}_{pno}_{i}")

    vs.add_documents(docs, ids=ids)
    return len(docs)

