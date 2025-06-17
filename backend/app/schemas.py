# app/schemas.py
from pydantic import BaseModel, Field, validator
from typing import Optional
from datetime import datetime

# Person 관련 스키마
class PersonBase(BaseModel):
    image_name: str
    human_height: int = Field(..., description="키 (mm)")
    buttock_popliteal_length: float = Field(..., description="엉덩이-오금 길이 (mm)")
    popliteal_height: float = Field(..., description="오금 높이 (mm)")
    hip_breadth: float = Field(..., description="엉덩이 너비 (mm)")
    sitting_height: float = Field(..., description="앉은 키 (mm)")
    shoulder_breadth: float = Field(..., description="어깨 너비 (mm)")

class PersonCreate(PersonBase):
    pass

class PersonResponse(PersonBase):
    person_id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

# Chair 관련 스키마
class ChairSpecificationBase(BaseModel):
    seat_height_min: Optional[int] = None
    seat_height_max: Optional[int] = None
    seat_width: Optional[int] = None
    seat_depth: Optional[int] = None
    backrest_width: Optional[int] = None
    backrest_height: Optional[int] = None

class ChairBase(BaseModel):
    product_url: Optional[str] = None
    brand_name: str
    product_name: str
    price: Optional[int] = None
    rating: Optional[float] = None
    review_count: Optional[int] = None
    has_headrest: bool = False
    has_armrest: bool = False
    has_lumbar_support: bool = False
    has_height_adjustment: bool = False
    has_tilting: bool = False
    backrest_type: Optional[str] = None

class ChairCreate(ChairBase):
    """의자 생성 스키마"""
    @validator('brand_name', pre=True, always=True)
    def set_default_brand(cls, v):
        return v or '알 수 없음'
    
    @validator('product_name', pre=True, always=True)
    def set_default_product_name(cls, v):
        return v or '제품명 없음'
    
    @validator('rating', pre=True)
    def validate_rating(cls, v):
        if v is not None:
            # 0-100 범위를 0-5로 변환
            if v > 5:
                v = v / 20.0
            return round(max(0, min(5, v)), 1)
        return v

class ChairResponse(ChairBase):
    chair_id: int
    created_at: datetime
    specifications: Optional[ChairSpecificationBase] = None
    
    class Config:
        from_attributes = True

class ChairUpdate(BaseModel):
    """의자 업데이트 스키마"""
    product_url: Optional[str] = None
    brand_name: Optional[str] = None
    product_name: Optional[str] = None
    price: Optional[int] = None
    rating: Optional[float] = None
    review_count: Optional[int] = None
    has_headrest: Optional[bool] = None
    has_armrest: Optional[bool] = None
    has_lumbar_support: Optional[bool] = None
    has_height_adjustment: Optional[bool] = None
    has_tilting: Optional[bool] = None
    backrest_type: Optional[str] = None

# Recommendation 관련 스키마
class RecommendationCreate(BaseModel):
    person_id: int
    chair_id: int
    match_score: float = Field(..., ge=0, le=100)
    rank: int = Field(..., ge=1)

class RecommendationResponse(BaseModel):
    rec_id: int
    person_id: int
    chair_id: int
    match_score: float
    rank: int
    created_at: datetime
    
    class Config:
        from_attributes = True

# Migration 관련 스키마
class MigrationStatus(BaseModel):
    """마이그레이션 상태"""
    status: str
    message: str
    total_chairs: int
    total_specifications: int

# 통계 관련 스키마
class DatabaseStats(BaseModel):
    """데이터베이스 통계"""
    total_chairs: int
    total_specifications: int
    total_persons: int
    total_recommendations: int
    brands: dict
    price_statistics: dict