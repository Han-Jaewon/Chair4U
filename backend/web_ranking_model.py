# web_ranking_model.py
import tensorflow as tf
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

class WebDifferentiableRanking:
    """
    웹 서비스용 Differentiable Ranking Metrics
    추론 시 랭킹 스코어 계산에 사용
    """
    
    def __init__(self, tau: float = 1.0, k: int = 5):
        """
        Args:
            tau: temperature parameter for neural sort
            k: top-k items to consider
        """
        self.tau = tau
        self.k = k
    
    @staticmethod
    def detNeuralSort(s, tau=1.0, k=2):
        """
        Deterministic neural sort for inference
        Args:
            s: scores tensor [batch_size, n_items]
            tau: temperature parameter
            k: top-k items to consider
        Returns:
            P_hat: permutation matrix [batch_size, k, n_items]
        """
        # Get dimensions
        batch_size = tf.shape(s)[0]
        n = tf.shape(s)[1]
        
        # Expand dimensions for broadcasting
        su = tf.expand_dims(s, axis=-1)  # [batch_size, n_items, 1]
        
        # Create matrices
        one = tf.ones([n, 1], dtype=tf.float32)
        one_k = tf.ones([1, k], dtype=tf.float32)
        
        # Compute A_s = |s_i - s_j|
        A_s = tf.abs(su - tf.transpose(su, [0, 2, 1]))  # [batch_size, n, n]
        
        # Compute B
        B = tf.matmul(A_s, tf.matmul(one, one_k))  # [batch_size, n, k]
        
        # Compute scaling
        scaling = tf.cast(n + 1 - 2 * (tf.range(n) + 1), tf.float32)
        scaling = tf.expand_dims(scaling, 0)  # [1, n]
        
        # Compute C
        C = tf.expand_dims(s * scaling, -1)[:, :, :k]  # [batch_size, n, k]
        
        # Compute P_max
        P_max = tf.transpose(C - B, [0, 2, 1])  # [batch_size, k, n]
        
        # Apply softmax
        P_hat = tf.nn.softmax(P_max / tau, axis=-1)
        
        return P_hat
    
    def calculate_ranking_scores(self, 
                                scores: np.ndarray, 
                                compatibility_scores: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        랭킹 스코어들을 계산 (추론용)
        
        Args:
            scores: 모델 예측 스코어 [batch_size, n_items]
            compatibility_scores: 호환성 스코어 (옵션) [batch_size, n_items]
        
        Returns:
            Dictionary with ranking scores
        """
        if len(scores.shape) == 1:
            scores = scores.reshape(1, -1)
        
        # TensorFlow 세션에서 계산
        with tf.Session() as sess:
            scores_tf = tf.constant(scores, dtype=tf.float32)
            
            # Neural sort 적용
            P_hat = self.detNeuralSort(scores_tf, tau=self.tau, k=self.k)
            
            # Top-k 인덱스 계산
            top_k_values, top_k_indices = tf.nn.top_k(scores_tf, k=min(self.k, scores.shape[1]))
            
            # 세션 실행
            P_hat_val, top_k_indices_val, top_k_values_val = sess.run([
                P_hat, top_k_indices, top_k_values
            ])
        
        return {
            'permutation_matrix': P_hat_val,
            'top_k_indices': top_k_indices_val,
            'top_k_scores': top_k_values_val,
            'raw_scores': scores
        }

class WebRankingPredictor:
    """웹 서비스용 랭킹 예측기"""
    
    def __init__(self, model_path: str = None, tau: float = 10.0, k: int = 5):
        """
        Args:
            model_path: 저장된 모델 경로
            tau: temperature parameter
            k: top-k 개수
        """
        self.model_path = model_path
        self.tau = tau
        self.k = k
        self.ranking_calculator = WebDifferentiableRanking(tau=tau, k=k)
        self.model = None
        self.session = None
        
    def load_model(self):
        """저장된 TensorFlow 모델 로드"""
        try:
            # SavedModel 형태로 저장된 경우
            if self.model_path:
                self.model = tf.saved_model.load(self.model_path)
                print(f"Model loaded from {self.model_path}")
            else:
                print("Warning: No model path provided")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def predict_batch(self, 
                     train_x: np.ndarray, 
                     train_i: np.ndarray) -> np.ndarray:
        """
        배치 예측 수행
        
        Args:
            train_x: 수치형 특성 [batch_size, 18]
            train_i: 범주형 특성 인덱스 [batch_size, categorical_features]
        
        Returns:
            예측 스코어 [batch_size]
        """
        if self.model is None:
            # 모델이 없는 경우 더미 예측 (개발/테스트용)
            print("Warning: No model loaded, returning dummy predictions")
            return np.random.random(train_x.shape[0])
        
        try:
            # 실제 모델 예측 (모델 구조에 따라 수정 필요)
            predictions = self.model(train_x, train_i)
            return predictions.numpy() if hasattr(predictions, 'numpy') else predictions
        except Exception as e:
            print(f"Error during prediction: {e}")
            return np.random.random(train_x.shape[0])
    
    def rank_chairs(self, 
                   train_x: np.ndarray, 
                   train_i: np.ndarray, 
                   chair_ids: List[int],
                   compatibility_scores: Optional[np.ndarray] = None) -> List[Dict]:
        """
        의자들을 랭킹하여 추천 순위 생성
        
        Args:
            train_x: 수치형 특성
            train_i: 범주형 특성 인덱스
            chair_ids: 의자 ID 리스트
            compatibility_scores: 사전 계산된 호환성 스코어
        
        Returns:
            랭킹된 추천 결과 리스트
        """
        # 모델 예측
        model_scores = self.predict_batch(train_x, train_i)
        
        # 호환성 스코어와 결합 (가중평균)
        if compatibility_scores is not None:
            # 모델 스코어 70%, 호환성 스코어 30%로 결합
            final_scores = 0.7 * model_scores + 0.3 * compatibility_scores
        else:
            final_scores = model_scores
        
        # 랭킹 계산
        ranking_info = self.ranking_calculator.calculate_ranking_scores(
            final_scores.reshape(1, -1),
            compatibility_scores.reshape(1, -1) if compatibility_scores is not None else None
        )
        
        # 결과 생성
        results = []
        for i, chair_id in enumerate(chair_ids):
            results.append({
                'chair_id': chair_id,
                'model_score': float(model_scores[i]),
                'compatibility_score': float(compatibility_scores[i]) if compatibility_scores is not None else None,
                'final_score': float(final_scores[i]),
                'rank': None  # 아래에서 설정
            })
        
        # 최종 스코어로 정렬
        results.sort(key=lambda x: x['final_score'], reverse=True)
        
        # 랭크 설정
        for rank, result in enumerate(results, 1):
            result['rank'] = rank
        
        return results

class ChairRecommendationService:
    """의자 추천 서비스 통합 클래스"""
    
    def __init__(self, 
                 db_session,
                 model_path: str = None,
                 tau: float = 10.0,
                 k: int = 10):
        """
        Args:
            db_session: 데이터베이스 세션
            model_path: 모델 경로
            tau: ranking temperature
            k: top-k 추천 개수
        """
        # 이전에 만든 모듈들 import
        from web_preprocessor import WebDataPreprocessor
        from web_scaler import prepare_scaled_prediction_data
        
        self.db_session = db_session
        self.preprocessor = WebDataPreprocessor(db_session)
        self.predictor = WebRankingPredictor(model_path, tau, k)
        self.k = k
        
        # 모델 로드
        self.predictor.load_model()
    
    def recommend_chairs(self, 
                        person_id: int, 
                        chair_ids: Optional[List[int]] = None,
                        top_k: Optional[int] = None) -> List[Dict]:
        """
        완전한 의자 추천 파이프라인
        
        Args:
            person_id: 사용자 ID
            chair_ids: 특정 의자들만 고려 (None이면 전체)
            top_k: 상위 k개만 반환 (None이면 기본값 사용)
        
        Returns:
            추천 결과 리스트
        """
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
            
            # 2. 호환성 스코어 추출 (전처리 단계에서 계산된 것)
            preprocessed_data = self.preprocessor.process_for_prediction(person_id, chair_ids)
            compatibility_scores = np.array([data['compatibility_score'] for data in preprocessed_data])
            
            # 3. 랭킹 예측
            ranking_results = self.predictor.rank_chairs(
                scaled_train_x, train_i, actual_chair_ids, compatibility_scores
            )
            
            # 4. 상위 k개만 반환
            return ranking_results[:top_k]
            
        except Exception as e:
            logging.error(f"Error in recommend_chairs: {e}")
            return []
    
    def get_recommendation_explanation(self, person_id: int, chair_id: int) -> Dict:
        """
        특정 추천에 대한 설명 제공
        
        Args:
            person_id: 사용자 ID
            chair_id: 의자 ID
        
        Returns:
            추천 설명 정보
        """
        try:
            # 단일 의자에 대한 분석
            preprocessed_data = self.preprocessor.process_for_prediction(person_id, [chair_id])
            
            if not preprocessed_data:
                return {}
            
            data = preprocessed_data[0]
            
            return {
                'chair_id': chair_id,
                'compatibility_score': data['compatibility_score'],
                'conditions_met': data['conditions_met'],
                'explanation': self._generate_explanation(data['conditions_met'])
            }
            
        except Exception as e:
            logging.error(f"Error in get_recommendation_explanation: {e}")
            return {}
    
    def _generate_explanation(self, conditions_met: Dict[str, bool]) -> str:
        """추천 이유 설명 생성"""
        explanations = []
        
        condition_messages = {
            't4 < A': "좌석 깊이가 적절합니다",
            'h8 ≈ B': "좌석 높이가 적절합니다", 
            'b3 > C': "좌석 너비가 충분합니다",
            'h7 < F': "등받이 높이가 적절합니다",
            'b4 ≥ G': "등받이 너비가 충분합니다"
        }
        
        for condition, is_met in conditions_met.items():
            if is_met and condition in condition_messages:
                explanations.append(condition_messages[condition])
        
        if explanations:
            return " / ".join(explanations)
        else:
            return "일부 조건이 맞지 않을 수 있습니다"

# 편의 함수
def create_recommendation_service(db_session, model_path: str = None) -> ChairRecommendationService:
    """추천 서비스 인스턴스 생성"""
    return ChairRecommendationService(db_session, model_path)