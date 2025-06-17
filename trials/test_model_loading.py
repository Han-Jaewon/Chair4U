# test_model_loading.py
# 저장된 모델이 제대로 로드되는지 테스트

import os
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def test_model_loading():
    """저장된 모델 로딩 테스트"""
    
    MODEL_PATH = './saved_models/autoint/1/'
    
    print("=== 모델 로딩 테스트 ===")
    
    # 1. 파일 존재 확인
    print("1. 체크포인트 파일 확인...")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 모델 경로를 찾을 수 없습니다: {MODEL_PATH}")
        return False
    
    files = os.listdir(MODEL_PATH)
    print(f"📁 파일 목록: {files}")
    
    required_patterns = ['checkpoint', '.data-', '.index', '.meta']
    found_patterns = []
    
    for pattern in required_patterns:
        found = any(pattern in f for f in files)
        found_patterns.append(found)
        status = "✅" if found else "❌"
        print(f"   {status} {pattern} 패턴 파일")
    
    if not all(found_patterns):
        print("❌ 필수 파일이 누락되었습니다.")
        return False
    
    # 2. TensorFlow로 모델 로딩 시도
    print("\n2. TensorFlow 모델 로딩 테스트...")
    
    try:
        # 세션 생성
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        
        # 체크포인트 상태 확인
        ckpt = tf.train.get_checkpoint_state(MODEL_PATH)
        
        if ckpt and ckpt.model_checkpoint_path:
            print(f"✅ 체크포인트 발견: {ckpt.model_checkpoint_path}")
            
            # 메타 그래프 로드
            saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
            print("✅ 메타 그래프 로드 성공")
            
            # 가중치 복원
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("✅ 모델 가중치 복원 성공")
            
            # 그래프 정보 확인
            graph = tf.get_default_graph()
            operations = [op.name for op in graph.get_operations()]
            
            # 중요한 텐서들 확인
            important_tensors = [
                'feat_index:0', 'feat_value:0', 'label:0', 
                'dropout_keep_prob:0', 'train_phase:0'
            ]
            
            print("\n📊 모델 구조 확인:")
            for tensor_name in important_tensors:
                try:
                    tensor = graph.get_tensor_by_name(tensor_name)
                    print(f"   ✅ {tensor_name}: {tensor.shape}")
                except:
                    print(f"   ❌ {tensor_name}: 찾을 수 없음")
            
            # 출력 텐서 확인
            try:
                output_tensor = graph.get_tensor_by_name('Sigmoid:0')  # 또는 다른 출력 이름
                print(f"   ✅ 출력 텐서: {output_tensor.shape}")
            except:
                print("   ⚠️  출력 텐서를 자동으로 찾지 못했습니다. 수동 확인 필요")
            
            sess.close()
            print("\n🎉 모델 로딩 테스트 성공!")
            return True
            
        else:
            print("❌ 체크포인트를 찾을 수 없습니다")
            return False
            
    except Exception as e:
        print(f"❌ 모델 로딩 실패: {e}")
        return False

def test_dummy_prediction():
    """더미 데이터로 예측 테스트"""
    
    MODEL_PATH = './saved_models/autoint/1/'
    
    print("\n=== 더미 예측 테스트 ===")
    
    try:
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        
        # 모델 로드
        ckpt = tf.train.get_checkpoint_state(MODEL_PATH)
        saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
        saver.restore(sess, ckpt.model_checkpoint_path)
        
        graph = tf.get_default_graph()
        
        # 입력 텐서들
        feat_index = graph.get_tensor_by_name('feat_index:0')
        feat_value = graph.get_tensor_by_name('feat_value:0')
        dropout_keep_prob = graph.get_tensor_by_name('dropout_keep_prob:0')
        train_phase = graph.get_tensor_by_name('train_phase:0')
        
        # 출력 텐서 (여러 가능성 시도)
        output_tensor = None
        possible_outputs = ['Sigmoid:0', 'pred:0', 'logits:0', 'out:0']
        
        for out_name in possible_outputs:
            try:
                output_tensor = graph.get_tensor_by_name(out_name)
                print(f"✅ 출력 텐서 발견: {out_name}")
                break
            except:
                continue
        
        if output_tensor is None:
            print("❌ 출력 텐서를 찾을 수 없습니다.")
            return False
        
        # 더미 데이터 생성 (field_size=24 가정)
        batch_size = 5
        field_size = 24
        
        dummy_feat_index = np.random.randint(0, 30, (batch_size, field_size))
        dummy_feat_value = np.random.random((batch_size, field_size)).astype(np.float32)
        dummy_dropout = [1.0, 1.0, 1.0]  # 추론 시에는 드롭아웃 비활성화
        
        # 예측 실행
        feed_dict = {
            feat_index: dummy_feat_index,
            feat_value: dummy_feat_value,
            dropout_keep_prob: dummy_dropout,
            train_phase: False
        }
        
        predictions = sess.run(output_tensor, feed_dict=feed_dict)
        
        print(f"✅ 예측 성공!")
        print(f"   입력 shape: {dummy_feat_index.shape}")
        print(f"   출력 shape: {predictions.shape}")
        print(f"   예측값 샘플: {predictions.flatten()[:3]}")
        print(f"   예측값 범위: [{predictions.min():.4f}, {predictions.max():.4f}]")
        
        sess.close()
        print("🎉 더미 예측 테스트 성공!")
        return True
        
    except Exception as e:
        print(f"❌ 예측 테스트 실패: {e}")
        return False

if __name__ == "__main__":
    print("=== AutoInt 모델 테스트 ===\n")
    
    # 1. 모델 로딩 테스트
    loading_success = test_model_loading()
    
    if loading_success:
        # 2. 예측 테스트
        prediction_success = test_dummy_prediction()
        
        if prediction_success:
            print("\n🚀 모델이 웹 서비스에 연결할 준비가 되었습니다!")
            print("\n다음 단계:")
            print("1. FastAPI 서버 설정")
            print("2. 데이터베이스 연결")
            print("3. 전처리 파이프라인 연결")
            print("4. API 엔드포인트 테스트")
        else:
            print("\n⚠️  모델 로딩은 성공했지만 예측에 문제가 있습니다.")
            print("   모델 구조를 다시 확인해주세요.")
    else:
        print("\n❌ 모델 로딩에 실패했습니다.")
        print("   체크포인트 파일을 다시 확인해주세요.")