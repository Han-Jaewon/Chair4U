# web_scaler.py
import math
import numpy as np
from typing import Union, List

class WebDataScaler:
    """웹 서비스용 데이터 스케일러 - 실시간 스케일링"""
    
    def __init__(self):
        """스케일러 초기화"""
        pass
    
    def scale_single_value(self, x: Union[int, float]) -> int:
        """
        개별 수치형 피처를 스케일링하는 함수
        
        Args:
            x: 스케일링할 수치값
        
        Returns:
            스케일링된 값 (int)
        """
        if x > 2:
            # 로그 스케일링 기법 (skew가 너무 클 때 사용)
            # x가 2보다 크면 로그 변환 후 제곱을 취한 값의 정수 부분만 반환
            # 큰 값들을 압축하여 특성값의 범위를 줄이는 효과가 있음
            x = int(math.log(float(x))**2)
        return x
    
    def scale_array(self, data: np.ndarray) -> np.ndarray:
        """
        수치형 피처 배열 전체를 스케일링
        
        Args:
            data: 스케일링할 수치형 데이터 배열 (shape: [batch_size, 18])
        
        Returns:
            스케일링된 데이터 배열
        """
        if len(data.shape) == 1:
            # 1차원 배열인 경우 (단일 샘플)
            return np.array([self.scale_single_value(x) for x in data], dtype=np.float64)
        
        elif len(data.shape) == 2:
            # 2차원 배열인 경우 (배치)
            scaled_data = np.zeros_like(data, dtype=np.float64)
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    scaled_data[i, j] = self.scale_single_value(data[i, j])
            return scaled_data
        
        else:
            raise ValueError(f"Unsupported data shape: {data.shape}")
    
    def scale_feature_vector(self, values: List[float], numerical_only: bool = True) -> List[float]:
        """
        피처 벡터의 수치형 부분만 스케일링
        
        Args:
            values: 전체 피처 값 리스트
            numerical_only: True면 처음 18개만 스케일링, False면 전체
        
        Returns:
            스케일링된 피처 값 리스트
        """
        scaled_values = values.copy()
        
        if numerical_only:
            # 처음 18개 수치형 피처만 스케일링
            for i in range(min(18, len(values))):
                scaled_values[i] = self.scale_single_value(values[i])
        else:
            # 전체 스케일링
            for i in range(len(values)):
                scaled_values[i] = self.scale_single_value(values[i])
        
        return scaled_values

class ScaledDataFormatter:
    """스케일링된 데이터를 모델 입력 형태로 포맷팅"""
    
    def __init__(self):
        self.scaler = WebDataScaler()
        self.Column = 18  # 수치형 특성 개수
    
    def format_with_scaling(self, preprocessed_data: List[dict]) -> tuple:
        """
        전처리된 데이터를 스케일링하고 모델 입력 형태로 포맷팅
        
        Args:
            preprocessed_data: WebDataPreprocessor.process_for_prediction() 결과
            
        Returns:
            scaled_train_x: 스케일링된 수치형 특성 배열
            train_i: 범주형 특성 인덱스 배열  
            chair_ids: chair ID 리스트
        """
        if not preprocessed_data:
            return np.array([]), np.array([]), []
        
        batch_size = len(preprocessed_data)
        chair_ids = []
        
        # 수치형 특성 배열 초기화
        train_x = np.zeros((batch_size, self.Column), dtype=np.float64)
        train_i_list = []
        
        for idx, data in enumerate(preprocessed_data):
            chair_ids.append(data['chair_id'])
            
            # 값과 인덱스 추출
            values = [float(v) for v in data['values']]
            indices = [int(i) for i in data['indices']]
            
            # 수치형 피처 스케일링 적용
            scaled_values = self.scaler.scale_feature_vector(values[:self.Column])
            train_x[idx] = np.array(scaled_values)
            
            # 범주형 특성 인덱스 (스케일링 안함)
            categorical_indices = indices[self.Column:]
            train_i_list.append(categorical_indices)
        
        # 범주형 인덱스를 numpy 배열로 변환
        train_i = np.array(train_i_list, dtype=np.float64)
        
        return train_x, train_i, chair_ids
    
    def format_single_with_scaling(self, person_chair_data: dict) -> tuple:
        """
        단일 Person-Chair 조합을 스케일링하고 모델 입력 형태로 포맷팅
        
        Args:
            person_chair_data: 단일 전처리 결과
            
        Returns:
            scaled_train_x: 스케일링된 수치형 특성 배열 (1, 18)
            train_i: 범주형 특성 인덱스 배열 (1, categorical_features)
        """
        values = [float(v) for v in person_chair_data['values']]
        indices = [int(i) for i in person_chair_data['indices']]
        
        # 수치형 특성 스케일링
        scaled_numerical = self.scaler.scale_feature_vector(values[:self.Column])
        train_x = np.array(scaled_numerical, dtype=np.float64).reshape(1, -1)
        
        # 범주형 특성 인덱스
        categorical_indices = indices[self.Column:]
        train_i = np.array([categorical_indices], dtype=np.float64)
        
        return train_x, train_i

# 통합 편의 함수
def prepare_scaled_prediction_data(
    preprocessor, 
    person_id: int, 
    chair_ids: List[int] = None
) -> tuple:
    """
    스케일링을 포함한 전체 예측 데이터 준비 파이프라인
    
    Args:
        preprocessor: WebDataPreprocessor 인스턴스
        person_id: 사용자 ID
        chair_ids: 의자 ID 리스트
    
    Returns:
        scaled_train_x, train_i, chair_ids
    """
    # 1. 전처리 실행
    preprocessed_data = preprocessor.process_for_prediction(person_id, chair_ids)
    
    # 2. 스케일링 및 포맷팅
    formatter = ScaledDataFormatter()
    scaled_train_x, train_i, chair_ids = formatter.format_with_scaling(preprocessed_data)
    
    return scaled_train_x, train_i, chair_ids

def validate_scaled_data(train_x: np.ndarray, train_i: np.ndarray) -> bool:
    """스케일링된 데이터 검증"""
    try:
        # 기본 형태 검증
        assert train_x.ndim == 2, "train_x should be 2D array"
        assert train_i.ndim == 2, "train_i should be 2D array"
        assert train_x.shape[0] == train_i.shape[0], "batch size should match"
        assert train_x.shape[1] == 18, "train_x should have 18 features"
        
        # 데이터 타입 검증
        assert train_x.dtype == np.float64, "train_x should be float64"
        assert train_i.dtype == np.float64, "train_i should be float64"
        
        # 스케일링 결과 검증 (모든 값이 정수여야 함)
        assert np.allclose(train_x, train_x.astype(int)), "Scaled values should be integers"
        
        return True
    except AssertionError as e:
        print(f"Scaled data validation failed: {e}")
        return False

# 테스트 함수
def test_scaling():
    """스케일링 함수 테스트"""
    scaler = WebDataScaler()
    
    test_values = [0.5, 1.0, 2.0, 10.0, 100.0, 1000.0]
    print("Original -> Scaled")
    for val in test_values:
        scaled = scaler.scale_single_value(val)
        print(f"{val} -> {scaled}")
    
    # 배열 테스트
    test_array = np.array([[1.0, 5.0, 100.0], [0.5, 10.0, 50.0]])
    scaled_array = scaler.scale_array(test_array)
    print(f"\nArray scaling test:")
    print(f"Original:\n{test_array}")
    print(f"Scaled:\n{scaled_array}")

if __name__ == "__main__":
    test_scaling()