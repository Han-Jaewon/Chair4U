# check_original_model.py
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

# 체크포인트 변수 확인
checkpoint_path = "./saved_models/autoint_house/1/model.ckpt-36"
reader = tf.train.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()

print("=== 모델 구조 분석 ===")
for key in sorted(var_to_shape_map.keys()):
    shape = var_to_shape_map[key]
    print(f"{key}: {shape}")
    
    # 주요 변수 분석
    if "feature_embeddings" in key:
        print(f"  → 전체 피처 수: {shape[0]}")
        print(f"  → 임베딩 크기: {shape[1]}")
    elif "prediction" in key and "bias" not in key:
        print(f"  → 최종 레이어 입력 크기: {shape[0]}")

# 원본 학습 데이터 확인
import os
if os.path.exists("../model_data/"):
    print("\n=== 원본 데이터 구조 확인 ===")
    if os.path.exists("../model_data/feature_size.npy"):
        feature_size = np.load("../model_data/feature_size.npy")
        print(f"원본 feature_size: {feature_size}")