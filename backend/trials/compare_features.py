# compare_features.py
import os
import pandas as pd
import numpy as np
from web_preprocessor import WebDataPreprocessor
from database import SessionLocal

# DB 전처리기
db = SessionLocal()
web_prep = WebDataPreprocessor(db)

print("=== 웹 전처리 테스트 ===")
try:
    # 테스트 데이터로 피처 확인
    test_data = web_prep.process_for_prediction(person_id=1, chair_ids=[1, 2, 3])
    if test_data:
        print(f"전처리 성공! {len(test_data)}개 결과")
        print(f"values 길이: {len(test_data[0]['values'])}")
        print(f"indices 길이: {len(test_data[0]['indices'])}")
        print(f"수치형 피처: {len(test_data[0]['values'][:18])}")
        print(f"범주형 피처: {len(test_data[0]['indices'][18:])}")
        
        # 실제 값 샘플 출력
        print("\n=== 샘플 데이터 ===")
        print(f"처음 5개 values: {test_data[0]['values'][:5]}")
        print(f"처음 5개 indices: {test_data[0]['indices'][:5]}")
        print(f"compatibility_score: {test_data[0]['compatibility_score']}")
        
        # 전체 피처 수 계산
        total_features = len(test_data[0]['indices'])
        print(f"\n총 피처 수: {total_features}")
        print(f"최대 인덱스: {max(int(i) for i in test_data[0]['indices'])}")
        
except Exception as e:
    print(f"전처리 실패: {e}")
    import traceback
    traceback.print_exc()

# 원본 CSV 전처리 방식 확인
print("\n=== 원본 데이터 구조 ===")
if os.path.exists("../person.csv"):
    person_df = pd.read_csv("../person.csv")
    print(f"Person shape: {person_df.shape}")
    print(f"Person 컬럼: {list(person_df.columns)}")
    
if os.path.exists("../chair.xlsx"):
    chair_df = pd.read_excel("../chair.xlsx")
    print(f"Chair shape: {chair_df.shape}")
    print(f"Chair 컬럼: {list(chair_df.columns)}")

# 원본 학습 코드의 피처 생성 방식 찾기
print("\n=== 원본 피처 수 추정 ===")
if os.path.exists("../person.csv") and os.path.exists("../chair.xlsx"):
    # Person: 6개 (image-name 제외)
    person_features = ['human-height', 'A_Buttock-popliteal length', 
                      'B_Popliteal-height', 'C_Hip-breadth', 
                      'F_Sitting-height', 'G_Shoulder-breadth']
    
    # Chair 수치형: 6개
    chair_numerical = ['h8_지면-좌석 높이_MIN', 'h8_지면-좌석 높이_MAX',
                      'b3_좌석 가로 길이', 't4_좌석 세로 길이 일반',
                      'b4_등받이 가로 길이', 'h7_등받이 세로 길이']
    
    # Chair 범주형: 6개
    chair_categorical = ['헤드레스트 유무', '팔걸이 유무', '요추지지대 유무',
                        '높이 조절 레버 유무', '틸팅 여부', '등받이 곧/꺾']
    
    print(f"Person 피처: {len(person_features)}개")
    print(f"Chair 수치형: {len(chair_numerical)}개")
    print(f"Chair 범주형: {len(chair_categorical)}개 (각각 0/1 = 12개)")
    print(f"상호작용 피처: 6개 (추정)")
    print(f"예상 총 피처: {6 + 6 + 12 + 6} = 30개")
    print(f"하지만 모델은 31개 피처 사용...")
    
    # 누락된 피처 찾기
    print("\n=== 가능한 31번째 피처 ===")
    print("1. bias 항 (상수 1)")
    print("2. E_Elbow-rest-height (사용 안 한다고 했지만...)")
    print("3. 추가 상호작용 피처")
    print("4. 가격이나 별점 같은 추가 피처")

# feature_size.npy 찾기
print("\n=== feature_size 파일 찾기 ===")
possible_paths = ["../", "../../", "../model_data/", "../saved_models/"]
for path in possible_paths:
    feature_size_path = os.path.join(path, "feature_size.npy")
    if os.path.exists(feature_size_path):
        feature_size = np.load(feature_size_path)
        print(f"✅ Found feature_size at {feature_size_path}: {feature_size}")

# DB 닫기
db.close()