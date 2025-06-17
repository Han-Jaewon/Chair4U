# web_preprocessor.py
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

def safe_divide(numerator, denominator, default=0.0):
    """안전한 나눗셈 헬퍼 함수"""
    if denominator > 0:
        return numerator / denominator
    return default

class WebDataPreprocessor:
    def __init__(self, db_session: Session):
        self.db_session = db_session
        
        # 사용할 Chair 피처 정의
        self.categorical_features = [
            '헤드레스트 유무', '팔걸이 유무', '요추지지대 유무', 
            '높이 조절 레버 유무', '틸팅 여부', '등받이 곧/꺾'
        ]
        
        self.numerical_features = [
            'h8_지면-좌석 높이_MIN', 'h8_지면-좌석 높이_MAX',
            'b3_좌석 가로 길이', 't4_좌석 세로 길이 일반',
            'b4_등받이 가로 길이', 'h7_등받이 세로 길이'
        ]
        
        # Person 데이터의 피처 (cm -> mm 변환 필요)
        self.person_features = [
            'human-height', 'A_Buttock-popliteal length',
            'B_Popliteal-height', 'C_Hip-breadth',
            'F_Sitting-height', 'G_Shoulder-breadth'
        ]
        
        # 피처 매핑 초기화
        self.feature_idx_map = {}
        self.binary_offset = 0
        self._create_feature_mappings()

    def _create_feature_mappings(self):
        """피처 인덱스 매핑 생성"""
        idx = 1
        
        # Person 피처 (6개)
        for feat in self.person_features:
            self.feature_idx_map[f'person_{feat}'] = idx
            idx += 1
        
        # Chair 수치형 피처 (6개)
        for feat in self.numerical_features:
            self.feature_idx_map[f'chair_{feat}'] = idx
            idx += 1
        
        # 상호작용 피처 (6개 추가)
        interaction_features = [
            'height_match_score', 'width_margin_ratio', 'depth_margin_ratio',
            'backrest_height_ratio', 'shoulder_width_ratio', 'adjustable_range'
        ]
        for feat in interaction_features:
            self.feature_idx_map[feat] = idx
            idx += 1
        
        # 이진 범주형 피처 시작 오프셋
        self.binary_offset = idx
        
        # 이진 피처 인덱스 (각 피처당 2개씩: 0과 1)
        for i, feat in enumerate(self.categorical_features):
            self.feature_idx_map[f'{feat}_0'] = self.binary_offset + i * 2
            self.feature_idx_map[f'{feat}_1'] = self.binary_offset + i * 2 + 1

    # web_preprocessor.py의 load_person_data_from_db 메서드 수정
    def load_person_data_from_db(self, person_id: Optional[int] = None) -> pd.DataFrame:
        """DB에서 Person 데이터 로드 및 전처리"""
        try:
            if person_id:
                query = """
                SELECT 
                    person_id,
                    human_height,
                    buttock_popliteal_length as "A_Buttock-popliteal length",
                    popliteal_height as "B_Popliteal-height", 
                    hip_breadth as "C_Hip-breadth",
                    sitting_height as "F_Sitting-height",
                    shoulder_breadth as "G_Shoulder-breadth"
                FROM persons 
                WHERE person_id = %(person_id)s
                """
                # pandas read_sql with dict params
                person_df = pd.read_sql(query, self.db_session.bind, params={'person_id': person_id})
            else:
                query = """
                SELECT 
                    person_id,
                    human_height,
                    buttock_popliteal_length as "A_Buttock-popliteal length",
                    popliteal_height as "B_Popliteal-height", 
                    hip_breadth as "C_Hip-breadth",
                    sitting_height as "F_Sitting-height",
                    shoulder_breadth as "G_Shoulder-breadth"
                FROM persons
                """
                person_df = pd.read_sql(query, self.db_session.bind)
            
            # human_height는 이미 mm 단위, 나머지는 cm이므로 mm로 변환
            cm_columns = ["A_Buttock-popliteal length", "B_Popliteal-height", "C_Hip-breadth", 
                        "F_Sitting-height", "G_Shoulder-breadth"]
            for col in cm_columns:
                if col in person_df.columns:
                    person_df[col] = person_df[col] * 10  # cm to mm
            
            # human-height 컬럼명 맞추기
            if 'human_height' in person_df.columns:
                person_df.rename(columns={'human_height': 'human-height'}, inplace=True)
            
            return person_df
            
        except Exception as e:
            print(f"Error loading person data: {e}")
            raise

    def load_chair_data_from_db(self, chair_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """DB에서 Chair 데이터 로드 및 전처리"""
        try:
            if chair_ids:
                # PostgreSQL array 사용
                query = """
                SELECT 
                    c.chair_id,
                    c.has_headrest as "헤드레스트 유무",
                    c.has_armrest as "팔걸이 유무", 
                    c.has_lumbar_support as "요추지지대 유무",
                    c.has_height_adjustment as "높이 조절 레버 유무",
                    c.has_tilting as "틸팅 여부",
                    c.backrest_type as "등받이 곧/꺾",
                    cs.seat_height_min as "h8_지면-좌석 높이_MIN",
                    cs.seat_height_max as "h8_지면-좌석 높이_MAX", 
                    cs.seat_width as "b3_좌석 가로 길이",
                    cs.seat_depth as "t4_좌석 세로 길이 일반",
                    cs.backrest_width as "b4_등받이 가로 길이",
                    cs.backrest_height as "h7_등받이 세로 길이"
                FROM o_chairs c
                LEFT JOIN o_chair_specifications cs ON c.chair_id = cs.chair_id
                WHERE c.chair_id = ANY(%(chair_ids)s)
                """
                chair_df = pd.read_sql(query, self.db_session.bind, params={'chair_ids': chair_ids})
            else:
                query = """
                SELECT 
                    c.chair_id,
                    c.has_headrest as "헤드레스트 유무",
                    c.has_armrest as "팔걸이 유무", 
                    c.has_lumbar_support as "요추지지대 유무",
                    c.has_height_adjustment as "높이 조절 레버 유무",
                    c.has_tilting as "틸팅 여부",
                    c.backrest_type as "등받이 곧/꺾",
                    cs.seat_height_min as "h8_지면-좌석 높이_MIN",
                    cs.seat_height_max as "h8_지면-좌석 높이_MAX", 
                    cs.seat_width as "b3_좌석 가로 길이",
                    cs.seat_depth as "t4_좌석 세로 길이 일반",
                    cs.backrest_width as "b4_등받이 가로 길이",
                    cs.backrest_height as "h7_등받이 세로 길이"
                FROM o_chairs c
                LEFT JOIN o_chair_specifications cs ON c.chair_id = cs.chair_id
                """
                chair_df = pd.read_sql(query, self.db_session.bind)
            
            # Boolean을 1/0으로 변환
            boolean_columns = ["헤드레스트 유무", "팔걸이 유무", "요추지지대 유무", 
                            "높이 조절 레버 유무", "틸팅 여부"]
            for col in boolean_columns:
                if col in chair_df.columns:
                    chair_df[col] = chair_df[col].astype(int)
            
            # 등받이 타입 변환
            if '등받이 곧/꺾' in chair_df.columns:
                chair_df['등받이 곧/꺾'] = chair_df['등받이 곧/꺾'].map({
                    '곧': 0, 
                    '꺾': 1,
                    'straight': 0, 
                    'curved': 1,
                    None: 0
                }).fillna(0).astype(int)
            
            # 결측값 처리
            chair_df['h8_지면-좌석 높이_MAX'] = np.where(
                pd.isna(chair_df['h8_지면-좌석 높이_MAX']),
                chair_df['h8_지면-좌석 높이_MIN'],
                chair_df['h8_지면-좌석 높이_MAX']
            )
            
            # 범주형 피처 결측값을 0으로
            chair_df[self.categorical_features] = chair_df[self.categorical_features].fillna(0).astype(int)
            
            # 수치형 피처 결측값을 평균으로
            for col in self.numerical_features:
                if col in chair_df.columns:
                    chair_df[col] = chair_df[col].fillna(chair_df[col].mean())
            
            return chair_df
            
        except Exception as e:
            print(f"Error loading chair data: {e}")
            raise

    def calculate_interaction_features(self, person_row: pd.Series, chair_row: pd.Series) -> Dict[str, float]:
        """Person과 Chair 간의 상호작용 피처 계산"""
        features = {}
        
        # 높이 적합도 (조절 범위 내에 있는지)
        h8_mid = (chair_row['h8_지면-좌석 높이_MIN'] + chair_row['h8_지면-좌석 높이_MAX']) / 2
        h8_range = chair_row['h8_지면-좌석 높이_MAX'] - chair_row['h8_지면-좌석 높이_MIN']
        popliteal_height = person_row['B_Popliteal-height']
        
        if h8_range > 0:
            if chair_row['h8_지면-좌석 높이_MIN'] <= popliteal_height <= chair_row['h8_지면-좌석 높이_MAX']:
                features['height_match_score'] = 1.0
            else:
                if popliteal_height < chair_row['h8_지면-좌석 높이_MIN']:
                    dist = chair_row['h8_지면-좌석 높이_MIN'] - popliteal_height
                else:
                    dist = popliteal_height - chair_row['h8_지면-좌석 높이_MAX']
                features['height_match_score'] = max(0, 1 - dist / 100)
        else:
            features['height_match_score'] = max(0, 1 - abs(h8_mid - popliteal_height) / 50)
        
        # 안전한 나눗셈 사용
        features['width_margin_ratio'] = safe_divide(
            chair_row['b3_좌석 가로 길이'] - person_row['C_Hip-breadth'], 
            person_row['C_Hip-breadth']
        )
        features['depth_margin_ratio'] = safe_divide(
            person_row['A_Buttock-popliteal length'] - chair_row['t4_좌석 세로 길이 일반'], 
            person_row['A_Buttock-popliteal length']
        )
        features['backrest_height_ratio'] = safe_divide(
            chair_row['h7_등받이 세로 길이'], 
            person_row['F_Sitting-height']
        )
        features['shoulder_width_ratio'] = safe_divide(
            chair_row['b4_등받이 가로 길이'], 
            person_row['G_Shoulder-breadth']
        )
        features['adjustable_range'] = h8_range
        
        return features

    def check_matching_conditions(self, person_row: pd.Series, chair_row: pd.Series) -> Tuple[int, float, Dict[str, bool]]:
        """필수 매칭 조건 확인 및 레이블 생성"""
        conditions = {
            't4 < A': chair_row['t4_좌석 세로 길이 일반'] < person_row['A_Buttock-popliteal length'],
            'h8 ≈ B': (chair_row['h8_지면-좌석 높이_MIN'] <= person_row['B_Popliteal-height'] <= chair_row['h8_지면-좌석 높이_MAX']) 
                      if chair_row['h8_지면-좌석 높이_MAX'] > chair_row['h8_지면-좌석 높이_MIN']
                      else abs((chair_row['h8_지면-좌석 높이_MIN'] + chair_row['h8_지면-좌석 높이_MAX'])/2 - person_row['B_Popliteal-height']) < 50,
            'b3 > C': chair_row['b3_좌석 가로 길이'] > person_row['C_Hip-breadth'],
            'h7 < F': chair_row['h7_등받이 세로 길이'] < person_row['F_Sitting-height'],
            'b4 ≥ G': chair_row['b4_등받이 가로 길이'] >= person_row['G_Shoulder-breadth']
        }
        
        all_satisfied = all(conditions.values())
        soft_label = sum(conditions.values()) / len(conditions)
        
        return int(all_satisfied), soft_label, conditions

    def create_feature_vector(self, person_row: pd.Series, chair_row: pd.Series) -> Tuple[List[str], List[str]]:
        """단일 Person-Chair 조합에 대한 피처 벡터 생성"""
        values = []
        indices = []
        
        # 1. Person 수치형 피처
        for feat in self.person_features:
            if feat in person_row.index:
                values.append(str(person_row[feat]))
                indices.append(str(self.feature_idx_map[f'person_{feat}']))
        
        # 2. Chair 수치형 피처
        for feat in self.numerical_features:
            values.append(str(chair_row[feat]))
            indices.append(str(self.feature_idx_map[f'chair_{feat}']))
        
        # 3. 상호작용 피처 계산
        interaction_feats = self.calculate_interaction_features(person_row, chair_row)
        for feat_name, feat_value in interaction_feats.items():
            values.append(str(feat_value))
            indices.append(str(self.feature_idx_map[feat_name]))
        
        # 4. 이진 범주형 피처
        for feat in self.categorical_features:
            values.append('1')
            feat_value = int(chair_row[feat]) if not pd.isna(chair_row[feat]) else 0
            idx_key = f'{feat}_{feat_value}'
            indices.append(str(self.feature_idx_map[idx_key]))
        
        return values, indices

    def process_for_prediction(self, person_id: int, chair_ids: Optional[List[int]] = None) -> List[Dict]:
        """예측을 위한 데이터 처리 (단일 Person과 여러 Chair 조합)"""
        person_df = self.load_person_data_from_db(person_id)
        chair_df = self.load_chair_data_from_db(chair_ids)
        
        if person_df.empty:
            raise ValueError(f"Person with ID {person_id} not found")
        
        person_row = person_df.iloc[0]
        results = []
        
        for _, chair_row in chair_df.iterrows():
            values, indices = self.create_feature_vector(person_row, chair_row)
            hard_label, soft_label, conditions = self.check_matching_conditions(person_row, chair_row)
            
            results.append({
                'chair_id': int(chair_row['chair_id']),
                'values': values,
                'indices': indices,
                'compatibility_score': soft_label,
                'conditions_met': conditions
            })
        
        return results

    def get_metadata(self) -> Dict:
        """메타데이터 반환"""
        return {
            'feature_mappings': self.feature_idx_map,
            'total_features': len(self.feature_idx_map),
            'numerical_features': self.binary_offset - 1,
            'categorical_features': self.categorical_features,
            'person_features': self.person_features,
            'chair_numerical_features': self.numerical_features
        }


# FastAPI에서 사용할 의존성 함수
def get_preprocessor(db_session: Session) -> WebDataPreprocessor:
    """의존성 주입용 전처리기 생성 함수"""
    return WebDataPreprocessor(db_session)