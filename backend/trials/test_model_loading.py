# test_complete_integration_v2.py
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from database import SessionLocal, OChair
from web_preprocessor import WebDataPreprocessor
from web_data_formatter import WebDataFormatter
from web_scaler import ScaledDataFormatter
from web_autoint_model import WebAutoIntService

print("=== Chair4U AI 모델 통합 테스트 V2 ===\n")

# 1. DB 연결 및 의자 ID 확인
print("1. DB 연결 및 데이터 확인...")
db = SessionLocal()

# 실제 존재하는 의자 ID 가져오기
actual_chairs = db.query(OChair).limit(10).all()
if not actual_chairs:
    print("   ❌ DB에 의자 데이터가 없습니다!")
    db.close()
    exit(1)

chair_ids_to_test = [c.chair_id for c in actual_chairs[:5]]
print(f"   테스트할 의자 ID: {chair_ids_to_test}")

# 2. 전처리기 초기화
print("\n2. 전처리기 초기화...")
preprocessor = WebDataPreprocessor(db)

# 3. 피처 매핑 수정 (31개로)
print("3. 피처 매핑 수정...")
original_create_feature_vector = preprocessor.create_feature_vector

def new_create_feature_vector(person_row, chair_row):
    values, indices = original_create_feature_vector(person_row, chair_row)
    # 31번째 피처 추가
    values.append('1.0')
    indices.append('30')
    return values, indices

preprocessor.create_feature_vector = new_create_feature_vector

# 4. 테스트 데이터 전처리
print("\n4. 테스트 데이터 전처리...")
try:
    test_data = preprocessor.process_for_prediction(
        person_id=1, 
        chair_ids=chair_ids_to_test
    )
    
    if not test_data:
        print("   ❌ 전처리 결과가 비어있습니다.")
        # 디버깅 정보
        person_df = preprocessor.load_person_data_from_db(1)
        chair_df = preprocessor.load_chair_data_from_db(chair_ids_to_test)
        print(f"   Person 데이터: {person_df.shape}")
        print(f"   Chair 데이터: {chair_df.shape}")
        db.close()
        exit(1)
        
    print(f"   ✅ 전처리 성공: {len(test_data)}개 의자")
    print(f"   피처 수: {len(test_data[0]['indices'])}")
    
except Exception as e:
    print(f"   ❌ 전처리 실패: {e}")
    import traceback
    traceback.print_exc()
    db.close()
    exit(1)

# 5. 데이터 포맷팅
print("\n5. 데이터 포맷팅...")
formatter = ScaledDataFormatter()
train_x, train_i, chair_ids = formatter.format_with_scaling(test_data)
print(f"   train_x shape: {train_x.shape}")
print(f"   train_i shape: {train_i.shape}")

# 6. AutoInt 모델 로드
print("\n6. AutoInt 모델 로드...")
model_config = {
    'feature_size': 31,
    'field_size': 24,
    'embedding_size': 16,
    'blocks': 2,
    'heads': 2,
    'block_shape': [16, 16],
    'has_residual': True,
    'deep_layers': None,
    'batch_size': 1024
}

autoint_service = WebAutoIntService(
    model_path="./saved_models/autoint_house/1/",
    model_config=model_config
)

success = autoint_service.initialize()
if success:
    print("   ✅ 모델 로드 성공!")
else:
    print("   ❌ 모델 로드 실패!")

# 7. 예측 수행
print("\n7. 예측 수행...")
try:
    predictions = autoint_service.predict_for_recommendation(train_x, train_i)
    print(f"   ✅ 예측 성공: {len(predictions)}개 점수")
    
    # 결과 정리
    print("\n=== 추천 결과 ===")
    results = []
    for i, (chair_id, score) in enumerate(zip(chair_ids, predictions)):
        # 의자 정보 가져오기
        chair = db.query(OChair).filter(OChair.chair_id == chair_id).first()
        results.append({
            'chair_id': chair_id,
            'brand': chair.brand_name if chair else 'Unknown',
            'product': chair.product_name if chair else 'Unknown',
            'score': float(score)
        })
    
    # 점수로 정렬
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # 순위 출력
    for rank, result in enumerate(results, 1):
        print(f"{rank}위: {result['brand']} - {result['product']} (ID: {result['chair_id']}, 점수: {result['score']:.4f})")
        
except Exception as e:
    print(f"   ❌ 예측 실패: {e}")
    import traceback
    traceback.print_exc()

# 8. 정리
print("\n8. 리소스 정리...")
autoint_service.cleanup()
db.close()
print("   ✅ 완료!")