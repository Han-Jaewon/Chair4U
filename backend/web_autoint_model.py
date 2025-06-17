# web_autoint_model.py
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import os
import logging
from typing import List, Dict, Tuple, Optional
from tensorflow.keras.layers import BatchNormalization

def normalize(inputs, epsilon=1e-8):
    """Layer Normalization 함수"""
    inputs_shape = inputs.get_shape()
    params_shape = inputs_shape[-1:]
    
    mean, variance = tf.nn.moments(inputs, [-1], keepdims=True)
    beta = tf.Variable(tf.zeros(params_shape))
    gamma = tf.Variable(tf.ones(params_shape))
    normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
    outputs = gamma * normalized + beta
    
    return outputs

def multihead_attention(queries, keys, values, num_units=None, num_heads=1,
                       dropout_keep_prob=1, is_training=True, has_residual=True):
    """Multi-head Self-Attention 메커니즘"""
    if num_units is None:
        num_units = queries.get_shape().as_list()[-1]

    # Linear projections
    Q = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)(queries)
    K = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)(keys)
    V = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)(values)
    if has_residual:
        V_res = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)(values)

    # Split and concat for multi-head
    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
    K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
    V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

    # Attention weights
    weights = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
    weights = weights / (K_.get_shape().as_list()[-1] ** 0.5)
    weights = tf.nn.softmax(weights)

    # Dropout
    weights = tf.cond(is_training,
                     lambda: tf.nn.dropout(weights, keep_prob=dropout_keep_prob),
                     lambda: weights)

    # Weighted sum
    outputs = tf.matmul(weights, V_)
    outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)

    # Residual connection
    if has_residual:
        outputs += V_res

    outputs = tf.nn.relu(outputs)
    outputs = normalize(outputs)
    
    return outputs

class WebAutoIntModel:
    """웹 서비스용 AutoInt 모델"""
    
    def __init__(self, model_config: Dict):
        """
        Args:
            model_config: 모델 설정 딕셔너리
        """
        # 모델 구조 파라미터
        self.feature_size = model_config.get('feature_size', 30)
        self.field_size = model_config.get('field_size', 24)
        self.embedding_size = model_config.get('embedding_size', 16)
        self.blocks = model_config.get('blocks', 2)
        self.heads = model_config.get('heads', 2)
        self.block_shape = model_config.get('block_shape', [16, 16])
        self.has_residual = model_config.get('has_residual', True)
        self.deep_layers = model_config.get('deep_layers', None)
        
        # 추론 관련 파라미터
        self.batch_size = model_config.get('batch_size', 1024)
        self.dropout_keep_prob = [1.0, 1.0, 1.0]  # 추론 시에는 드롭아웃 비활성화
        
        # TensorFlow 세션 관련
        self.graph = None
        self.sess = None
        self.model_loaded = False
        
        # 입력/출력 텐서 참조
        self.feat_index = None
        self.feat_value = None
        self.output = None
        self.train_phase = None
        
    def _build_inference_graph(self):
        """추론용 그래프 구축"""
        self.graph = tf.Graph()
        with self.graph.as_default():
            # 입력 플레이스홀더
            self.feat_index = tf.placeholder(tf.int32, shape=[None, None], name="feat_index")
            self.feat_value = tf.placeholder(tf.float32, shape=[None, None], name="feat_value")
            self.train_phase = tf.placeholder(tf.bool, name="train_phase")
            
            # 가중치 플레이스홀더 (모델 로딩 시 설정)
            self.weights = self._create_weight_placeholders()
            
            # 모델 구조 정의
            self._build_model_structure()
            
            # 세션 초기화
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config, graph=self.graph)
    
    def _create_weight_placeholders(self):
        """가중치 플레이스홀더 생성"""
        weights = {}
        
        # Feature embeddings
        weights["feature_embeddings"] = tf.Variable(
            tf.random_normal([self.feature_size, self.embedding_size], 0.0, 0.01),
            name="feature_embeddings")
        
        # DNN layers (if exists)
        if self.deep_layers is not None:
            layer0_size = self.field_size * self.embedding_size
            for i, layer_size in enumerate(self.deep_layers):
                if i == 0:
                    input_size = layer0_size
                else:
                    input_size = self.deep_layers[i-1]
                
                weights[f"layer_{i}"] = tf.Variable(
                    tf.random_normal([input_size, layer_size], 0.0, 0.01),
                    name=f"layer_{i}")
                weights[f"bias_{i}"] = tf.Variable(
                    tf.random_normal([layer_size], 0.0, 0.01),
                    name=f"bias_{i}")
            
            # DNN output layer
            weights["prediction_dense"] = tf.Variable(
                tf.random_normal([self.deep_layers[-1], 1], 0.0, 0.01),
                name="prediction_dense")
            weights["prediction_bias_dense"] = tf.Variable(
                tf.random_normal([1], 0.0, 0.01),
                name="prediction_bias_dense")
        
        # Prediction layer
        if self.blocks > 0:
            final_size = self.block_shape[-1] * self.field_size  # 16 * 24 = 384
        else:
            final_size = self.embedding_size * self.field_size
        
        weights["prediction"] = tf.Variable(
            tf.random_normal([final_size, 1], 0.0, 0.01),
            name="prediction")
        
        # prediction_bias는 스칼라!
        weights["prediction_bias"] = tf.Variable(
            tf.constant(0.0),  # 스칼라 값
            name="prediction_bias")
        
        return weights
    
    def _build_model_structure(self):
        """모델 구조 정의"""
        # 1. Embedding layer
        self.embeddings = tf.nn.embedding_lookup(
            self.weights["feature_embeddings"], self.feat_index)
        
        # Feature value 적용
        feat_value = tf.reshape(self.feat_value, 
                               shape=[-1, tf.shape(self.feat_index)[1], 1])
        self.embeddings = tf.multiply(self.embeddings, feat_value)
        
        # 2. DNN branch (optional)
        if self.deep_layers is not None:
            y_dense = tf.reshape(self.embeddings, 
                               shape=[-1, self.field_size * self.embedding_size])
            
            for i in range(len(self.deep_layers)):
                y_dense = tf.add(
                    tf.matmul(y_dense, self.weights[f"layer_{i}"]),
                    self.weights[f"bias_{i}"])
                y_dense = tf.nn.relu(y_dense)
            
            self.y_dense = tf.add(
                tf.matmul(y_dense, self.weights["prediction_dense"]),
                self.weights["prediction_bias_dense"])
        
        # 3. AutoInt branch (Multi-head Self-Attention)
        y_deep = self.embeddings
        for i in range(self.blocks):
            y_deep = multihead_attention(
                queries=y_deep, keys=y_deep, values=y_deep,
                num_units=self.block_shape[i], num_heads=self.heads,
                dropout_keep_prob=1.0, is_training=self.train_phase,
                has_residual=self.has_residual)
        
        # Flatten
        flat = tf.reshape(y_deep, [tf.shape(y_deep)[0], -1])
        
        # AutoInt output
        y_autoint = tf.add(
            tf.matmul(flat, self.weights['prediction']),
            self.weights['prediction_bias'])
        
        # 4. Final output
        if self.deep_layers is not None:
            self.output = tf.nn.sigmoid(y_autoint + self.y_dense)
        else:
            self.output = tf.nn.sigmoid(y_autoint)
    
    def load_model(self, model_path: str) -> bool:
        """저장된 모델 로드"""
        try:
            if not self.graph:
                self._build_inference_graph()
            
            with self.graph.as_default():
                # 체크포인트에서 모델 복원
                saver = tf.train.Saver()
                ckpt = tf.train.get_checkpoint_state(model_path)
                
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(self.sess, ckpt.model_checkpoint_path)
                    self.model_loaded = True
                    logging.info(f"Model loaded from {ckpt.model_checkpoint_path}")
                    return True
                else:
                    logging.error(f"No checkpoint found in {model_path}")
                    return False
                    
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return False
    
    def predict_batch(self, Xi: np.ndarray, Xv: np.ndarray) -> np.ndarray:
        """배치 예측 수행"""
        if not self.model_loaded:
            logging.warning("Model not loaded, returning random predictions")
            return np.random.random(Xi.shape[0])
        
        try:
            with self.graph.as_default():
                feed_dict = {
                    self.feat_index: Xi,
                    self.feat_value: Xv,
                    self.train_phase: False
                }
                predictions = self.sess.run(self.output, feed_dict=feed_dict)
                return predictions.reshape(-1)
                
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            return np.random.random(Xi.shape[0])
    
    def predict(self, Xi: np.ndarray, Xv: np.ndarray) -> np.ndarray:
        """전체 데이터에 대한 예측 (배치 단위로 처리)"""
        if len(Xi) == 0:
            return np.array([])
        
        # 작은 배치는 바로 처리
        if len(Xi) <= self.batch_size:
            return self.predict_batch(Xi, Xv)
        
        # 큰 데이터는 배치로 나누어 처리
        predictions = []
        for i in range(0, len(Xi), self.batch_size):
            end_idx = min(i + self.batch_size, len(Xi))
            batch_pred = self.predict_batch(Xi[i:end_idx], Xv[i:end_idx])
            predictions.extend(batch_pred)
        
        return np.array(predictions)
    
    def get_model_info(self) -> Dict:
        """모델 정보 반환"""
        return {
            'feature_size': self.feature_size,
            'field_size': self.field_size,
            'embedding_size': self.embedding_size,
            'blocks': self.blocks,
            'heads': self.heads,
            'has_deep_layers': self.deep_layers is not None,
            'model_loaded': self.model_loaded
        }
    
    def close(self):
        """리소스 정리"""
        if self.sess:
            self.sess.close()

class WebAutoIntService:
    """AutoInt 모델 서비스 래퍼"""
    
    def __init__(self, model_path: str, model_config: Dict = None):
        """
        Args:
            model_path: 모델 저장 경로
            model_config: 모델 설정 (None이면 기본값 사용)
        """
        if model_config is None:
            model_config = self._get_default_config()
        
        self.model = WebAutoIntModel(model_config)
        self.model_path = model_path
        self.is_loaded = False
    
    def _get_default_config(self) -> Dict:
        """기본 모델 설정"""
        return {
            'feature_size': 30,
            'field_size': 24,
            'embedding_size': 16,
            'blocks': 2,
            'heads': 2,
            'block_shape': [16, 16],
            'has_residual': True,
            'deep_layers': None,
            'batch_size': 1024
        }
    
    def initialize(self) -> bool:
        """모델 초기화 및 로딩"""
        success = self.model.load_model(self.model_path)
        self.is_loaded = success
        return success
    
    def predict_for_recommendation(self, 
                                  train_x: np.ndarray, 
                                  train_i: np.ndarray) -> np.ndarray:
        """추천을 위한 예측"""
        if not self.is_loaded:
            logging.error("Model not loaded")
            return np.array([])
        
        # train_i를 int32로, train_x를 float32로 변환
        Xi = train_i.astype(np.int32)
        Xv = train_x.astype(np.float32)
        
        return self.model.predict(Xi, Xv)
    
    def get_service_info(self) -> Dict:
        """서비스 정보 반환"""
        model_info = self.model.get_model_info()
        return {
            'model_path': self.model_path,
            'is_loaded': self.is_loaded,
            'model_info': model_info
        }
    
    def cleanup(self):
        """리소스 정리"""
        self.model.close()

# 통합 추천 서비스에 AutoInt 통합
class EnhancedChairRecommendationService:
    """AutoInt가 통합된 의자 추천 서비스"""
    
    def __init__(self, 
                 db_session,
                 model_path: str,
                 model_config: Dict = None,
                 tau: float = 10.0,
                 k: int = 50):  # 웹용으로 기본값을 50개로 증가
        
        from web_preprocessor import WebDataPreprocessor
        from web_scaler import prepare_scaled_prediction_data
        
        self.db_session = db_session
        self.preprocessor = WebDataPreprocessor(db_session)
        self.autoint_service = WebAutoIntService(model_path, model_config)
        self.k = k
        
        # AutoInt 모델 초기화
        self.model_ready = self.autoint_service.initialize()
        
        if not self.model_ready:
            logging.warning("AutoInt model failed to load, using fallback predictions")
    
    def recommend_chairs(self, 
                        person_id: int, 
                        chair_ids: Optional[List[int]] = None,
                        top_k: Optional[int] = None) -> List[Dict]:
        """완전한 의자 추천 파이프라인 (AutoInt 사용)"""
        if top_k is None:
            top_k = self.k
        
        try:
            # 1. 데이터 전처리 및 스케일링
            from web_scaler import prepare_scaled_prediction_data
            scaled_train_x, train_i, actual_chair_ids = prepare_scaled_prediction_data(
                self.preprocessor, person_id, chair_ids
            )
            
            if len(actual_chair_ids) == 0:
                return []
            
            # 2. AutoInt 모델 예측
            if self.model_ready:
                model_scores = self.autoint_service.predict_for_recommendation(
                    scaled_train_x, train_i
                )
            else:
                # 폴백: 랜덤 스코어
                model_scores = np.random.random(len(actual_chair_ids))
            
            # 3. 호환성 스코어 (전처리 단계에서 계산된 것)
            preprocessed_data = self.preprocessor.process_for_prediction(person_id, chair_ids)
            compatibility_scores = np.array([data['compatibility_score'] for data in preprocessed_data])
            
            # 4. 최종 스코어 계산 (가중평균)
            final_scores = 0.7 * model_scores + 0.3 * compatibility_scores
            
            # 5. 결과 생성 및 정렬
            results = []
            for i, chair_id in enumerate(actual_chair_ids):
                results.append({
                    'chair_id': chair_id,
                    'autoint_score': float(model_scores[i]),
                    'compatibility_score': float(compatibility_scores[i]),
                    'final_score': float(final_scores[i]),
                    'rank': None
                })
            
            # 최종 스코어로 정렬
            results.sort(key=lambda x: x['final_score'], reverse=True)
            
            # 랭크 설정
            for rank, result in enumerate(results, 1):
                result['rank'] = rank
            
            return results[:top_k]
            
        except Exception as e:
            logging.error(f"Error in recommend_chairs: {e}")
            return []
    
    def cleanup(self):
        """리소스 정리"""
        self.autoint_service.cleanup()

# 편의 함수
def create_autoint_recommendation_service(
    db_session, 
    model_path: str, 
    model_config: Dict = None
) -> EnhancedChairRecommendationService:
    """AutoInt 추천 서비스 생성"""
    return EnhancedChairRecommendationService(db_session, model_path, model_config)