# app/main.py
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional
from pydantic import BaseModel

from .aa_database import get_db, engine
from ..database import Base, Chair, ChairSpecification, Person, Recommendation
from .schemas import ChairResponse

# MigrationStatus 정의
class MigrationStatus(BaseModel):
    status: str
    message: str
    total_chairs: int
    total_specifications: int

# 테이블 생성
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Chair4U API", version="1.0.0")

@app.get("/")
def read_root():
    return {"message": "Chair4U Backend API", "status": "running"}

@app.get("/chairs", response_model=List[ChairResponse])
def get_chairs(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """모든 의자 목록 조회"""
    chairs = db.query(Chair).offset(skip).limit(limit).all()
    return chairs

@app.get("/chairs/{chair_id}", response_model=ChairResponse)
def get_chair(chair_id: int, db: Session = Depends(get_db)):
    """특정 의자 정보 조회"""
    chair = db.query(Chair).filter(Chair.chair_id == chair_id).first()
    if not chair:
        raise HTTPException(status_code=404, detail="Chair not found")
    return chair

@app.get("/stats")
def get_statistics(db: Session = Depends(get_db)):
    """데이터베이스 통계"""
    total_chairs = db.query(Chair).count()
    total_specs = db.query(ChairSpecification).count()
    total_persons = db.query(Person).count()
    total_recommendations = db.query(Recommendation).count()
    
    # 브랜드별 통계 (상위 10개)
    brand_stats = db.query(
        Chair.brand_name,
        func.count(Chair.chair_id).label('count')
    ).group_by(Chair.brand_name).order_by(func.count(Chair.chair_id).desc()).limit(10).all()
    
    # 가격 통계
    price_stats = db.query(
        func.min(Chair.price).label('min_price'),
        func.max(Chair.price).label('max_price'),
        func.avg(Chair.price).label('avg_price')
    ).filter(Chair.price.isnot(None)).first()
    
    return {
        "total_chairs": total_chairs,
        "total_specifications": total_specs,
        "total_persons": total_persons,
        "total_recommendations": total_recommendations,
        "brands": {brand: count for brand, count in brand_stats},
        "price_statistics": {
            "min": price_stats.min_price if price_stats and price_stats.min_price else None,
            "max": price_stats.max_price if price_stats and price_stats.max_price else None,
            "average": float(price_stats.avg_price) if price_stats and price_stats.avg_price else None
        }
    }

@app.get("/check-db")
def check_database(db: Session = Depends(get_db)):
    """데이터베이스 연결 및 데이터 확인"""
    try:
        # 연결 테스트
        chair_count = db.query(Chair).count()
        spec_count = db.query(ChairSpecification).count()
        
        # 샘플 의자 가져오기
        sample_chair = db.query(Chair).first()
        
        return {
            "status": "connected",
            "chair_count": chair_count,
            "specification_count": spec_count,
            "sample_chair": {
                "id": sample_chair.chair_id if sample_chair else None,
                "brand": sample_chair.brand_name if sample_chair else None,
                "product": sample_chair.product_name if sample_chair else None
            },
            "message": f"데이터베이스 연결 성공! 의자 {chair_count}개, 사양 {spec_count}개 확인됨"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/chairs/search")
def search_chairs(
    brand_name: Optional[str] = None,
    min_price: Optional[int] = None,
    max_price: Optional[int] = None,
    has_headrest: Optional[bool] = None,
    has_armrest: Optional[bool] = None,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """의자 검색"""
    query = db.query(Chair)
    
    if brand_name:
        query = query.filter(Chair.brand_name.contains(brand_name))
    if min_price:
        query = query.filter(Chair.price >= min_price)
    if max_price:
        query = query.filter(Chair.price <= max_price)
    if has_headrest is not None:
        query = query.filter(Chair.has_headrest == has_headrest)
    if has_armrest is not None:
        query = query.filter(Chair.has_armrest == has_armrest)
    
    chairs = query.limit(limit).all()
    
    return {
        "count": len(chairs),
        "chairs": [
            {
                "chair_id": chair.chair_id,
                "brand_name": chair.brand_name,
                "product_name": chair.product_name,
                "price": chair.price,
                "rating": chair.rating,
                "has_headrest": chair.has_headrest,
                "has_armrest": chair.has_armrest
            }
            for chair in chairs
        ]
    }