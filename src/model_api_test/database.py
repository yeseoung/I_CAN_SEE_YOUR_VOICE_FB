import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.orm import declarative_base, sessionmaker


# 프로젝트 최상위 폴더의 .env 파일 불러오기
BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")


# .env에서 PostgreSQL 접속 정보 가져오기
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")


# 필수 환경변수 확인
if not all([DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD]):
    raise ValueError("데이터베이스 접속 정보가 .env에 올바르게 설정되지 않았습니다.")


# PostgreSQL 접속 주소 생성
DATABASE_URL = URL.create(
    drivername="postgresql+psycopg",
    username=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=int(DB_PORT),
    database=DB_NAME,
)


# 데이터베이스 엔진 생성
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)


# 데이터베이스 세션 생성기
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


# 데이터베이스 테이블 모델의 기본 클래스
Base = declarative_base()


# API에서 사용할 데이터베이스 연결 함수
def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()