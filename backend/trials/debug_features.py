# debug_features.py
from web_preprocessor import WebDataPreprocessor
from database import SessionLocal

db = SessionLocal()
prep = WebDataPreprocessor(db)

# feature_idx_map 확인
print("=== 현재 피처 매핑 ===")
for feat, idx in sorted(prep.feature_idx_map.items(), key=lambda x: x[1]):
    print(f"{idx}: {feat}")

print(f"\n총 피처 수: {len(prep.feature_idx_map)}")
print(f"최대 인덱스: {max(prep.feature_idx_map.values())}")

# 누락된 피처가 있는지 확인
# 원본 학습 시 추가 피처가 있었을 가능성