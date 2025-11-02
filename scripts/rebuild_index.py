# Pinecone 인덱스를 완전히 삭제하고 새로 생성하는 스크립트
# 한국어 임베딩 모델로 전환할 때 사용


import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import os
from dotenv import load_dotenv
load_dotenv()

from pinecone import Pinecone, ServerlessSpec

def main():
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX = os.getenv("PINECONE_INDEX", "uos-notices")
    PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
    PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

    # 한국어 모델 차원 (768)
    EMBED_DIM = 768

    if not PINECONE_API_KEY:
        print("PINECONE_API_KEY가 설정되지 않았습니다.")
        return 1

    pc = Pinecone(api_key=PINECONE_API_KEY)

    # 1. 기존 인덱스 삭제
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX in existing_indexes:
        print(f" 기존 인덱스 '{PINECONE_INDEX}' 삭제 중")
        pc.delete_index(PINECONE_INDEX)
        print("인덱스 삭제 완료")
    else:
        print(f" 인덱스 '{PINECONE_INDEX}'가 존재하지 않습니다.")

    # 2. 새 인덱스 생성 (768차원)
    print(f"새 인덱스 '{PINECONE_INDEX}' 생성 중")
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=EMBED_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
    )
    print("인덱스 생성 완료")

    print("\nPinecone 인덱스 재구축 완료!")
    return 0

if __name__ == "__main__":
    sys.exit(main())