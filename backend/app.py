# app.py

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import logging
from datetime import datetime

# DB 관련 import
from database import get_db, Person, OChair, OChairSpecification, test_db_connection
from sqlalchemy.orm import Session

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic 모델들 먼저 정의
class UserPreferences(BaseModel):
    headrest: Optional[str] = "any"
    armrest: Optional[str] = "any"
    lumbar_support: Optional[str] = "any"
    height_adjustment: Optional[str] = "any"
    tilting: Optional[str] = "any"
    price_min: Optional[int] = None
    price_max: Optional[int] = None
    rating_min: Optional[float] = None

class RecommendationRequest(BaseModel):
    person_id: int
    preferences: Optional[UserPreferences] = UserPreferences()
    top_k: Optional[int] = 20

class ChairRecommendation(BaseModel):
    chair_id: int
    autoint_score: float
    compatibility_score: float
    final_score: float
    rank: int

class RecommendationResponse(BaseModel):
    person_id: int
    recommendations: List[ChairRecommendation]
    total_count: int
    model_used: str = "AutoInt + DRM"

# FastAPI 앱 생성 (Pydantic 모델 정의 후)
app = FastAPI(
    title="Chair Recommendation API",
    description="AutoInt 기반 의자 추천 시스템 (DB 연결)",
    version="2.0.0"
)

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # React 앱 주소들
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # OPTIONS 메서드 포함
    allow_headers=["*"],  # 모든 헤더 허용
)

# 전역 변수
recommendation_service = None
HAS_AI_MODEL = False  # AI 모델 사용 가능 여부

# AI 모델 import 시도
try:
    from web_autoint_model import create_autoint_recommendation_service, EnhancedChairRecommendationService
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    HAS_AI_MODEL = True
    logger.info("✅ AI 모델 모듈 import 성공")
except ImportError as e:
    logger.warning(f"⚠️ AI 모델 모듈 import 실패: {e}")
    HAS_AI_MODEL = False

# 모델 경로 설정
MODEL_PATHS = [
    "./saved_models/autoint_house/1/",  # 실제 경로 추가 # 2: 더미 데이터 # 1: 실제 데이터
    "./saved_models/",
    "../saved_models/",
    "../../saved_models/",
]

MODEL_PATH = None
for path in MODEL_PATHS:
    if os.path.exists(path):
        MODEL_PATH = path
        break

if MODEL_PATH is None:
    MODEL_PATH = "./saved_models/autoint_house/1/"

# 의자 필터링 함수
def filter_chairs_by_preferences(db: Session, preferences: UserPreferences):
    """사용자 선호도에 따라 의자 필터링"""
    
    logger.info(f"🎛️  필터링 시작 - 선호도: {preferences.dict()}")
    
    query = db.query(OChair)
    
    # Boolean 필터링
    if preferences.headrest == "required":
        query = query.filter(OChair.has_headrest.is_(True))
    elif preferences.headrest == "not_required":
        query = query.filter(OChair.has_headrest.is_(False))
    
    if preferences.armrest == "required":
        query = query.filter(OChair.has_armrest.is_(True))
    elif preferences.armrest == "not_required":
        query = query.filter(OChair.has_armrest.is_(False))
    
    if preferences.lumbar_support == "required":
        query = query.filter(OChair.has_lumbar_support.is_(True))
    elif preferences.lumbar_support == "not_required":
        query = query.filter(OChair.has_lumbar_support.is_(False))
    
    if preferences.height_adjustment == "required":
        query = query.filter(OChair.has_height_adjustment.is_(True))
    elif preferences.height_adjustment == "not_required":
        query = query.filter(OChair.has_height_adjustment.is_(False))
    
    if preferences.tilting == "required":
        query = query.filter(OChair.has_tilting.is_(True))
    elif preferences.tilting == "not_required":
        query = query.filter(OChair.has_tilting.is_(False))
    
    # 가격 필터링
    if preferences.price_min:
        query = query.filter(OChair.price >= preferences.price_min)
    if preferences.price_max:
        query = query.filter(OChair.price <= preferences.price_max)
    
    # 평점 필터링
    if preferences.rating_min:
        query = query.filter(OChair.rating >= preferences.rating_min)
    
    filtered_chairs = query.all()
    logger.info(f"🎯 필터링 결과: {len(filtered_chairs)}개 의자")
    
    return filtered_chairs

@app.on_event("startup")
async def startup_event():
    """앱 시작시 초기화"""
    global recommendation_service
    
    logger.info("Chair Recommendation API 시작 중...")
    
    # DB 연결 테스트
    logger.info("DB 연결 테스트 중...")
    if test_db_connection():
        logger.info("DB 연결 성공")
    else:
        logger.error("DB 연결 실패")
        return
    
    # AI 모델 로드 시도
    if HAS_AI_MODEL:
        try:
            logger.info("AI 모델 로딩 중...")
            
            from database import SessionLocal
            db = SessionLocal()
            
            # 전처리기 수정 - 31번째 피처 추가
            from web_preprocessor import WebDataPreprocessor
            prep = WebDataPreprocessor(db)
            original_create_feature_vector = prep.create_feature_vector
            
            def new_create_feature_vector(person_row, chair_row):
                values, indices = original_create_feature_vector(person_row, chair_row)
                values.append('1.0')  # 31번째 피처
                indices.append('30')
                return values, indices
            
            prep.create_feature_vector = new_create_feature_vector
            
            # 모델 설정
            model_config = {
                'feature_size': 31,  # 31개!
                'field_size': 24,
                'embedding_size': 16,
                'blocks': 2,
                'heads': 2,
                'block_shape': [16, 16],
                'has_residual': True,
                'deep_layers': None,
                'batch_size': 1024
            }
            
            # 정확한 경로 사용
            model_path = "./saved_models/autoint_house/1/"
            logger.info(f"모델 경로: {model_path}")
            
            # 커스텀 추천 서비스 생성
            from web_autoint_model import EnhancedChairRecommendationService
            recommendation_service = EnhancedChairRecommendationService(
                db_session=db,
                model_path=model_path,
                model_config=model_config,
                k=50
            )
            
            # 전처리기 교체
            recommendation_service.preprocessor = prep
            
            logger.info("모델 로드 성공")
            
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            import traceback
            traceback.print_exc()
            recommendation_service = None
    else:
        logger.warning("모델 없이 실행 (더미 추천 사용)")
    
    logger.info("API 초기화 완료")

@app.on_event("shutdown")
async def shutdown_event():
    """앱 종료시 리소스 정리"""
    global recommendation_service
    if recommendation_service and HAS_AI_MODEL:
        recommendation_service.cleanup()
        logger.info("리소스 정리 완료")

@app.get("/")
async def root():
    """API 루트 엔드포인트"""
    return {
        "message": "Chair Recommendation API",
        "version": "2.0.0",
        "status": "running",
        "ai_model": "loaded" if recommendation_service else "not_loaded",
        "docs_url": "/docs"
    }

@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {
        "status": "healthy",
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH),
        "ai_model_available": HAS_AI_MODEL,
        "ai_model_loaded": recommendation_service is not None
    }

@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_chairs(request: RecommendationRequest, db: Session = Depends(get_db)):
    """의자 추천 메인 엔드포인트"""
    
    logger.info(f"추천 요청: person_id={request.person_id}, top_k={request.top_k}")
    
    try:
        # 1. Person 존재 확인
        person = db.query(Person).filter(Person.person_id == request.person_id).first()
        if not person:
            raise HTTPException(status_code=404, detail=f"Person ID {request.person_id}를 찾을 수 없습니다")
        
        # 2. 사용자 선호도로 의자 필터링
        filtered_chairs = filter_chairs_by_preferences(db, request.preferences)
        
        if len(filtered_chairs) == 0:
            logger.warning("필터링 결과가 0개 -> 전체 의자 사용")
            filtered_chairs = db.query(OChair).limit(5000).all()
        
        # 3. 모델 사용 추천
        if recommendation_service and HAS_AI_MODEL:
            logger.info("모델 사용하여 추천")
            
            # DB 세션 업데이트
            recommendation_service.db_session = db
            recommendation_service.preprocessor.db_session = db
            
            # 필터링된 의자 ID들
            chair_ids = [chair.chair_id for chair in filtered_chairs[:5000]]
            
            # 추천 실행
            ai_recommendations = recommendation_service.recommend_chairs(
                person_id=request.person_id,
                chair_ids=chair_ids,
                top_k=request.top_k
            )
            
            # 응답 형식 변환
            recommendations = []
            for rec in ai_recommendations:
                recommendations.append(ChairRecommendation(
                    chair_id=rec['chair_id'],
                    autoint_score=round(rec['autoint_score'], 3),
                    compatibility_score=round(rec['compatibility_score'], 3),
                    final_score=round(rec['final_score'], 3),
                    rank=rec['rank']
                ))
            
            model_used = f"AutoInt AI (필터링: {len(filtered_chairs)}개)"
            
        else:
            logger.info("더미 추천 사용")
            recommendations = []
            
            for i, chair in enumerate(filtered_chairs[:request.top_k]):
                base_score = 0.95 - (i * 0.05)
                
                # 선호도 보너스
                preference_bonus = 0.0
                if chair.has_headrest and request.preferences.headrest == "required":
                    preference_bonus += 0.05
                if chair.has_armrest and request.preferences.armrest == "required":
                    preference_bonus += 0.05
                
                final_score = min(0.99, base_score + preference_bonus)
                
                recommendations.append(ChairRecommendation(
                    chair_id=chair.chair_id,
                    autoint_score=round(final_score, 3),
                    compatibility_score=round(base_score, 3),
                    final_score=round(final_score, 3),
                    rank=i + 1
                ))
            
            model_used = f"Dummy (필터링: {len(filtered_chairs)}개)"
        
        response = RecommendationResponse(
            person_id=request.person_id,
            recommendations=recommendations,
            total_count=len(recommendations),
            model_used=model_used
        )
        
        logger.info(f"추천 완료: {len(recommendations)}개 결과")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"추천 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"추천 처리 중 오류: {str(e)}")

@app.get("/model/info")
async def get_model_info():
    """모델 정보 조회"""
    return {
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH),
        "ai_module_available": HAS_AI_MODEL,
        "ai_model_loaded": recommendation_service is not None,
        "model_type": "AutoInt" if HAS_AI_MODEL else "None"
    }

@app.get("/persons")
async def get_persons(db: Session = Depends(get_db)):
    """등록된 사람들 목록 조회"""
    try:
        persons = db.query(Person).all()
        return {
            "total_count": len(persons),
            "persons": [
                {
                    "person_id": p.person_id,
                    "image_name": p.image_name,
                    "height": p.human_height
                } for p in persons
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB 조회 실패: {str(e)}")

@app.get("/chairs/{chair_id}")
async def get_chair_detail(chair_id: int, db: Session = Depends(get_db)):
    """특정 의자의 상세 정보 조회"""
    try:
        chair = db.query(OChair).filter(OChair.chair_id == chair_id).first()
        if not chair:
            raise HTTPException(status_code=404, detail=f"Chair ID {chair_id}를 찾을 수 없습니다")
        
        # ChairSpecification 정보도 함께 가져오기
        spec = db.query(OChairSpecification).filter(OChairSpecification.chair_id == chair_id).first()
        
        return {
            "chair_id": chair.chair_id,
            "product_url": chair.product_url,
            "brand_name": chair.brand_name,
            "product_name": chair.product_name,
            "price": chair.price,
            "rating": chair.rating,
            "review_count": chair.review_count,
            "has_headrest": chair.has_headrest,
            "has_armrest": chair.has_armrest,
            "has_lumbar_support": chair.has_lumbar_support,
            "has_height_adjustment": chair.has_height_adjustment,
            "has_tilting": chair.has_tilting,
            "backrest_type": chair.backrest_type,
            "specifications": {
                "seat_height_min": spec.seat_height_min if spec else None,
                "seat_height_max": spec.seat_height_max if spec else None,
                "seat_width": spec.seat_width if spec else None,
                "seat_depth": spec.seat_depth if spec else None,
                "backrest_width": spec.backrest_width if spec else None,
                "backrest_height": spec.backrest_height if spec else None,
            } if spec else None
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"의자 조회 중 오류: {e}")
        raise HTTPException(status_code=500, detail=f"의자 조회 중 오류: {str(e)}")

@app.get("/chairs/sample")
async def get_sample_chairs(limit: int = 10, db: Session = Depends(get_db)):
    """샘플 의자들 조회"""
    try:
        chairs = db.query(OChair).limit(limit).all()
        return {
            "total_shown": len(chairs),
            "chairs": [
                {
                    "chair_id": c.chair_id,
                    "brand_name": c.brand_name,
                    "product_name": c.product_name,
                    "price": c.price
                } for c in chairs
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB 조회 실패: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)