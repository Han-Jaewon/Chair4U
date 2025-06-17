# database.py (수정된 버전)

from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Numeric, CheckConstraint, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import os
from datetime import datetime

# 환경 변수에서 DB URL 가져오기 (또는 직접 설정)
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://chair4u_user:inisw06@localhost:5432/chair4u_db")

# SQLAlchemy 설정
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 데이터베이스 모델들
class Person(Base):
    __tablename__ = "persons"
    
    person_id = Column(Integer, primary_key=True, index=True)
    image_name = Column(String(255), nullable=False)
    human_height = Column(Integer, nullable=False)
    buttock_popliteal_length = Column(Numeric(10, 2), nullable=False)
    popliteal_height = Column(Numeric(10, 2), nullable=False)
    hip_breadth = Column(Numeric(10, 2), nullable=False)
    sitting_height = Column(Numeric(10, 2), nullable=False)
    shoulder_breadth = Column(Numeric(10, 2), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

class Chair(Base):
    __tablename__ = "chairs"
    
    chair_id = Column(Integer, primary_key=True, index=True)
    product_url = Column(Text)
    brand_name = Column(String(100), nullable=False)
    product_name = Column(String(255), nullable=False)
    price = Column(Integer)
    rating = Column(Numeric(3, 1))
    review_count = Column(Integer)
    has_headrest = Column(Boolean, default=False)
    has_armrest = Column(Boolean, default=False)
    has_lumbar_support = Column(Boolean, default=False)
    has_height_adjustment = Column(Boolean, default=False)
    has_tilting = Column(Boolean, default=False)
    backrest_type = Column(String(10))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    
    # 관계 설정
    specifications = relationship("ChairSpecification", back_populates="chair", uselist=False)

class ChairSpecification(Base):
    __tablename__ = "chair_specifications"
    
    spec_id = Column(Integer, primary_key=True, index=True)
    chair_id = Column(Integer, ForeignKey("chairs.chair_id"), nullable=False, unique=True)
    seat_height_min = Column(Integer)
    seat_height_max = Column(Integer)
    seat_width = Column(Integer)
    seat_depth = Column(Integer)
    backrest_width = Column(Integer)
    backrest_height = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    
    # 관계 설정
    chair = relationship("Chair", back_populates="specifications")

class OChair(Base):
    __tablename__ = "o_chairs"
    
    chair_id = Column(Integer, primary_key=True, index=True)
    product_url = Column(Text)
    brand_name = Column(String(100), nullable=False)
    product_name = Column(String(255), nullable=False)
    price = Column(Integer)
    rating = Column(Numeric(3, 1))
    review_count = Column(Integer)
    has_headrest = Column(Boolean, default=False)
    has_armrest = Column(Boolean, default=False)
    has_lumbar_support = Column(Boolean, default=False)
    has_height_adjustment = Column(Boolean, default=False)
    has_tilting = Column(Boolean, default=False)
    backrest_type = Column(String(10))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

    # 관계 설정
    specifications = relationship("OChairSpecification", back_populates="chair", uselist=False)

class OChairSpecification(Base):
    __tablename__ = "o_chair_specifications"
    
    spec_id = Column(Integer, primary_key=True, index=True)
    chair_id = Column(Integer, ForeignKey("o_chairs.chair_id"), nullable=False, unique=True)
    seat_height_min = Column(Integer)
    seat_height_max = Column(Integer)
    seat_width = Column(Integer)
    seat_depth = Column(Integer)
    backrest_width = Column(Integer)
    backrest_height = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

    # 관계 설정
    chair = relationship("OChair", back_populates="specifications")
    
    __table_args__ = (
        CheckConstraint(
            'seat_height_max IS NULL OR seat_height_min IS NULL OR seat_height_max >= seat_height_min',
            name='o_valid_seat_height'
        ),
    )

# 의존성 주입용 함수
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# DB 연결 테스트 함수
def test_db_connection():
    """DB 연결 테스트"""
    try:
        db = SessionLocal()
        
        # persons 테이블 테스트
        persons_count = db.query(Person).count()
        chairs_count = db.query(Chair).count()
        o_chairs_count = db.query(OChair).count()
        
        print(f"DB 연결 성공!")
        print(f"Persons: {persons_count}명")
        print(f"Chairs: {chairs_count}개")
        print(f"O_Chairs: {o_chairs_count}개")
        
        # 첫 번째 person 정보 확인
        if persons_count > 0:
            first_person = db.query(Person).first()
            print(f"첫 번째 Person ID: {first_person.person_id}")
            print(f"신장: {first_person.human_height}mm")
        
        # O_chairs 데이터 확인
        if o_chairs_count > 0:
            first_chair = db.query(OChair).first()
            print(f"첫 번째 O_Chair: {first_chair.brand_name} - {first_chair.product_name[:30]}...")
            
            # 관계 테스트
            if first_chair.specifications:
                print(f"좌석 높이: {first_chair.specifications.seat_height_min}-{first_chair.specifications.seat_height_max}mm")
        
        db.close()
        return True
        
    except Exception as e:
        print(f"DB 연결 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_db_connection()