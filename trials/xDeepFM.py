import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import ndcg_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict
import warnings
import copy
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """데이터 전처리 클래스"""
    
    def __init__(self):
        self.person_scaler = StandardScaler()
        self.chair_scaler = StandardScaler()
        
    def load_and_preprocess_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """데이터 로드 및 전처리"""
        
        # Person 데이터 로드
        person_df = pd.read_csv('person.csv')
        print(f"Person 데이터 형태: {person_df.shape}")
        print(f"Person 컬럼: {person_df.columns.tolist()}")
        
        # Chair 데이터 로드
        chair_df = pd.read_excel('chair.xlsx')
        print(f"Chair 데이터 형태: {chair_df.shape}")
        print(f"Chair 컬럼: {chair_df.columns.tolist()}")
        
        # Person 데이터 전처리
        person_df = self._preprocess_person_data(person_df)
        
        # Chair 데이터 전처리
        chair_df = self._preprocess_chair_data(chair_df)
        
        return person_df, chair_df
    
    def _preprocess_person_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Person 데이터 전처리"""
        
        # 결측치 확인
        print("Person 데이터 결측치:")
        print(df.isnull().sum())
        
        # 결측치 보간 - 평균값으로 대체
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        
        # 이미지명 컬럼이 있다면 제거 (모델링에 불필요)
        if 'image-name' in df.columns:
            df = df.drop('image-name', axis=1)
        
        # 컬럼명 정리
        df.columns = [col.replace(' ', '').replace('-', '_') for col in df.columns]
        
        return df
    
    def _preprocess_chair_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Chair 데이터 전처리"""
        
        # 결측치 확인
        print("Chair 데이터 결측치:")
        print(df.isnull().sum())
        
        # 범위형 데이터 처리 (MIN, MAX 컬럼들)
        processed_df = df.copy()
        
        # MIN, MAX 쌍을 찾아서 중간값과 범위로 변환
        min_cols = [col for col in df.columns if col.endswith('_MIN')]
        max_cols = [col for col in df.columns if col.endswith('_MAX')]
        
        for min_col in min_cols:
            max_col = min_col.replace('_MIN', '_MAX')
            if max_col in df.columns:
                base_name = min_col.replace('_MIN', '')
                
                # 중간값 계산
                processed_df[f'{base_name}_mean'] = (df[min_col] + df[max_col]) / 2
                # 범위 계산
                processed_df[f'{base_name}_range'] = df[max_col] - df[min_col]
                
                # 원본 MIN, MAX 컬럼 제거
                processed_df = processed_df.drop([min_col, max_col], axis=1)
        
        # 단위 변환 (mm → cm)
        numeric_columns = processed_df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            processed_df[col] = processed_df[col] / 10  # mm를 cm로 변환
        
        # 결측치 보간
        for col in numeric_columns:
            if processed_df[col].isnull().sum() > 0:
                processed_df[col] = processed_df[col].fillna(processed_df[col].median())
        
        # 컬럼명 정리
        processed_df.columns = [col.replace(' ', '').replace('-', '_') for col in processed_df.columns]
        
        return processed_df
    
    def create_compatibility_labels(self, person_df: pd.DataFrame, chair_df: pd.DataFrame) -> pd.DataFrame:
        """매칭 조건 기반 호환성 레이블 생성"""
        
        # 매칭 조건 정의
        matching_conditions = {
            'A_Buttockpopliteal_length': ('t4_mean', 'less_than'),  # t4 < A
            'B_Popliteal_height': ('h8_지면좌석_높이_mean', 'approximately'),  # h8 ≈ B
            'C_Hip_breadth': ('b3_좌석_가로_길이_mean', 'greater_than'),  # b3 > C
            'E_Elbow_rest_height': ('h9_좌석팔걸이_높이_mean', 'approximately'),  # h9 ≈ E
            'F_Sitting_height': ('h7_등받이_세로_길이_mean', 'less_than'),  # h7 < F
            'G_Shoulder_breadth': ('b4_등받이_가로_길이_mean', 'greater_equal')  # b4 >= G
        }
        
        compatibility_data = []
        
        for person_idx, person_row in person_df.iterrows():
            for chair_idx, chair_row in chair_df.iterrows():
                
                compatibility_score = 0
                condition_scores = {}
                
                for person_feature, (chair_feature, condition_type) in matching_conditions.items():
                    if person_feature in person_row and chair_feature in chair_row:
                        person_val = person_row[person_feature]
                        chair_val = chair_row[chair_feature]
                        
                        # 조건별 점수 계산
                        if condition_type == 'less_than':
                            score = 1.0 if chair_val < person_val else max(0, 1 - abs(chair_val - person_val) / person_val)
                        elif condition_type == 'greater_than':
                            score = 1.0 if chair_val > person_val else max(0, 1 - abs(chair_val - person_val) / person_val)
                        elif condition_type == 'greater_equal':
                            score = 1.0 if chair_val >= person_val else max(0, 1 - abs(chair_val - person_val) / person_val)
                        elif condition_type == 'approximately':
                            # 10% 오차 허용
                            tolerance = 0.1 * person_val
                            score = 1.0 if abs(chair_val - person_val) <= tolerance else max(0, 1 - abs(chair_val - person_val) / person_val)
                        
                        condition_scores[f'{person_feature}_{chair_feature}'] = score
                        compatibility_score += score
                
                # 평균 호환성 점수
                avg_compatibility = compatibility_score / len(matching_conditions)
                
                # 데이터 행 생성
                row_data = {
                    'person_id': person_idx,
                    'chair_id': chair_idx,
                    'compatibility_score': avg_compatibility,
                    'label': 1 if avg_compatibility >= 0.7 else 0  # 70% 이상이면 호환
                }
                
                # Person 특성 추가
                for col in person_df.columns:
                    if col != 'image-name':
                        row_data[f'person_{col}'] = person_row[col]
                
                # Chair 특성 추가
                for col in chair_df.columns:
                    row_data[f'chair_{col}'] = chair_row[col]
                
                # 조건별 점수 추가
                row_data.update(condition_scores)
                
                compatibility_data.append(row_data)
        
        compatibility_df = pd.DataFrame(compatibility_data)
        print(f"호환성 데이터 형태: {compatibility_df.shape}")
        print(f"긍정 레이블 비율: {compatibility_df['label'].mean():.3f}")
        
        return compatibility_df

class CIN(nn.Module):
    """Compressed Interaction Network"""
    
    def __init__(self, field_dim: int, embed_dim: int, hidden_layers: List[int]):
        super(CIN, self).__init__()
        self.field_dim = field_dim
        self.embed_dim = embed_dim
        self.hidden_layers = hidden_layers
        
        # CIN 레이어들
        self.cin_layers = nn.ModuleList()
        prev_dim = field_dim
        
        for hidden_dim in hidden_layers:
            self.cin_layers.append(
                nn.Conv1d(prev_dim * field_dim, hidden_dim, kernel_size=1)
            )
            prev_dim = hidden_dim
        
        # 출력 차원
        self.output_dim = sum(hidden_layers)
    
    def forward(self, x):
        """
        x: (batch_size, field_dim, embed_dim)
        """
        batch_size = x.size(0)
        x0 = x  # 원본 임베딩
        cin_outputs = []
        
        xk = x0
        for layer in self.cin_layers:
            # Outer product 계산
            # xk와 x0의 외적 계산
            outer_product = torch.einsum('bfe,bge->bfge', xk, x0)  # (batch, field, field, embed)
            outer_product = outer_product.view(batch_size, -1, self.embed_dim)  # (batch, field*field, embed)
            
            # Convolution 적용
            # (batch, field*field, embed) -> (batch, embed, field*field) -> conv -> (batch, hidden_dim, embed)
            outer_product = outer_product.transpose(1, 2)
            xk = layer(outer_product.view(batch_size, -1, 1)).squeeze(-1)  # (batch, hidden_dim)
            xk = xk.unsqueeze(-1).expand(-1, -1, self.embed_dim)  # (batch, hidden_dim, embed)
            xk = xk.transpose(1, 2)  # (batch, embed, hidden_dim) -> (batch, hidden_dim, embed)
            
            # Sum pooling
            pooled = torch.sum(xk, dim=-1)  # (batch, hidden_dim)
            cin_outputs.append(pooled)
        
        # 모든 레이어 출력 연결
        cin_output = torch.cat(cin_outputs, dim=1)  # (batch, sum(hidden_layers))
        return cin_output

class xDeepFM(nn.Module):
    """xDeepFM 모델 (CIN + DNN + Linear)"""
    
    def __init__(self, feature_dims: Dict[str, int], embed_dim: int = 16, 
                 cin_hidden_layers: List[int] = [128, 64], 
                 dnn_hidden_layers: List[int] = [256, 128, 64]):
        super(xDeepFM, self).__init__()
        
        self.feature_dims = feature_dims
        self.embed_dim = embed_dim
        self.total_features = sum(feature_dims.values())
        
        # 임베딩 레이어들
        self.embeddings = nn.ModuleDict()
        for field_name, field_dim in feature_dims.items():
            self.embeddings[field_name] = nn.Embedding(field_dim, embed_dim)
        
        # Linear part
        self.linear = nn.Linear(self.total_features, 1)
        
        # CIN part
        self.cin = CIN(len(feature_dims), embed_dim, cin_hidden_layers)
        
        # DNN part
        dnn_input_dim = len(feature_dims) * embed_dim
        dnn_layers = []
        prev_dim = dnn_input_dim
        
        for hidden_dim in dnn_hidden_layers:
            dnn_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        self.dnn = nn.Sequential(*dnn_layers)
        
        # 최종 출력 레이어
        final_input_dim = 1 + sum(cin_hidden_layers) + prev_dim  # linear + cin + dnn
        self.final_layer = nn.Linear(final_input_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x_dict: Dict[str, torch.Tensor], x_linear: torch.Tensor):
        """
        x_dict: 각 필드별 인덱스 텐서
        x_linear: 선형 부분용 원본 특성
        """
        
        # Linear part
        linear_out = self.linear(x_linear)
        
        # 임베딩 생성
        embeddings = []
        for field_name, indices in x_dict.items():
            embed = self.embeddings[field_name](indices)
            embeddings.append(embed)
        
        # 임베딩 스택: (batch_size, num_fields, embed_dim)
        embed_stack = torch.stack(embeddings, dim=1)
        
        # CIN part
        cin_out = self.cin(embed_stack)
        
        # DNN part
        dnn_input = embed_stack.view(embed_stack.size(0), -1)  # flatten
        dnn_out = self.dnn(dnn_input)
        
        # 모든 부분 결합
        final_input = torch.cat([linear_out, cin_out, dnn_out], dim=1)
        output = self.final_layer(final_input)
        
        return self.sigmoid(output)

class DRMRanker:
    """DRM 기반 Top-K 랭킹 클래스"""
    
    def __init__(self, temperature: float = 0.5):
        self.temperature = temperature
    
    def differentiable_ranking(self, scores: torch.Tensor) -> torch.Tensor:
        """DRM 기법을 이용한 차별화 가능한 랭킹"""
        
        # NeuralSort의 relaxed sorting 구현
        n = scores.size(-1)
        
        # Pairwise differences
        diff_matrix = scores.unsqueeze(-1) - scores.unsqueeze(-2)  # (batch, n, n)
        
        # Soft ranking matrix using softmax
        ranking_matrix = torch.softmax(diff_matrix / self.temperature, dim=-1)
        
        # Top-k selection using soft attention
        return ranking_matrix
    
    def get_top_k_recommendations(self, model_scores: torch.Tensor, k: int = 5) -> List[List[int]]:
        """Top-K 의자 추천"""
        
        batch_size = model_scores.size(0)
        recommendations = []
        
        for i in range(batch_size):
            scores = model_scores[i]
            
            # DRM ranking 적용
            ranking_matrix = self.differentiable_ranking(scores.unsqueeze(0))
            
            # 점수 기반 정렬
            sorted_indices = torch.argsort(scores, descending=True)
            top_k_indices = sorted_indices[:k].cpu().numpy().tolist()
            
            recommendations.append(top_k_indices)
        
        return recommendations

class ChairRecommendationSystem:
    """전체 의자 추천 시스템"""
    
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.model = None
        self.ranker = DRMRanker()
        self.person_df = None
        self.chair_df = None
        self.compatibility_df = None
        
    def prepare_data(self):
        """데이터 준비"""
        print("=== 데이터 로드 및 전처리 ===")
        self.person_df, self.chair_df = self.preprocessor.load_and_preprocess_data()
        
        print("\n=== 호환성 레이블 생성 ===")
        self.compatibility_df = self.preprocessor.create_compatibility_labels(
            self.person_df, self.chair_df
        )
        
        return self.compatibility_df
    
    def prepare_model_data(self, df: pd.DataFrame):
        """모델 학습용 데이터 준비"""
        
        # 특성 컬럼 분리
        person_cols = [col for col in df.columns if col.startswith('person_')]
        chair_cols = [col for col in df.columns if col.startswith('chair_')]
        
        # 수치형 특성들
        numeric_features = []
        categorical_features = {}
        
        # Person 특성 처리
        for col in person_cols:
            if df[col].dtype in ['int64', 'float64']:
                numeric_features.append(col)
        
        # Chair 특성 처리  
        for col in chair_cols:
            if df[col].dtype in ['int64', 'float64']:
                numeric_features.append(col)
        
        # 범주형 특성을 위한 인덱싱 (간단한 구현을 위해 생략하고 수치형만 사용)
        X_numeric = df[numeric_features].copy()
        
        # 정규화
        scaler = StandardScaler()
        X_numeric_scaled = scaler.fit_transform(X_numeric)
        X_numeric_scaled = pd.DataFrame(X_numeric_scaled, columns=numeric_features)
        
        # 간단한 특성 인덱싱 (실제로는 더 정교한 방법 필요)
        feature_dims = {}
        x_dict = {}
        
        # 수치형 특성을 구간별로 나누어 범주화
        for i, col in enumerate(numeric_features):
            values = X_numeric_scaled[col].values
            # 10개 구간으로 나누어 인덱싱
            bins = np.percentile(values, np.linspace(0, 100, 11))
            indices = np.digitize(values, bins) - 1
            indices = np.clip(indices, 0, 9)  # 0-9 범위로 클리핑
            
            feature_dims[col] = 10
            x_dict[col] = torch.tensor(indices, dtype=torch.long)
        
        # 선형 부분용 원본 특성
        x_linear = torch.tensor(X_numeric_scaled.values, dtype=torch.float32)
        
        # 레이블
        y = torch.tensor(df['label'].values, dtype=torch.float32).unsqueeze(1)
        
        return x_dict, x_linear, y, feature_dims, scaler
    
    def train_model(self, df: pd.DataFrame):
        """모델 학습 (소규모 데이터셋용 - 학습/테스트 2-way 분할)"""
        print("=== 모델 데이터 준비 ===")
        x_dict, x_linear, y, feature_dims, scaler = self.prepare_model_data(df)
        
        # 학습/테스트 분할 (80%/20%)
        indices = list(range(len(y)))
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, 
                                             stratify=y.numpy() if len(np.unique(y.numpy())) > 1 else None)
        
        print(f"데이터 분할: 학습 {len(train_idx)}, 테스트 {len(test_idx)}")
        
        # 모델 초기화
        self.model = xDeepFM(
            feature_dims=feature_dims,
            embed_dim=16,
            cin_hidden_layers=[64, 32],
            dnn_hidden_layers=[128, 64, 32]
        )
        
        # 손실 함수 및 옵티마이저
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        
        print("=== 모델 학습 시작 ===")
        
        # 학습 설정 (소규모 데이터에 맞게 조정)
        num_epochs = 100  # 에포크 수를 늘려서 충분히 학습
        batch_size = min(64, len(train_idx) // 4)  # 작은 배치 사이즈
        
        # 학습 히스토리
        train_history = {'loss': [], 'accuracy': []}
        
        # 최적 모델 저장용
        best_train_loss = float('inf')
        best_model_state = None
        
        for epoch in range(num_epochs):
            # ===== 학습 단계 =====
            self.model.train()
            epoch_loss = 0
            epoch_correct = 0
            epoch_total = 0
            num_batches = 0
            
            # 학습 데이터 셔플
            train_indices_shuffled = torch.randperm(len(train_idx))
            
            for i in range(0, len(train_idx), batch_size):
                batch_indices = train_indices_shuffled[i:i+batch_size]
                batch_idx = [train_idx[j] for j in batch_indices]
                
                # 배치 데이터 준비
                batch_x_dict = {}
                for key, tensor in x_dict.items():
                    batch_x_dict[key] = tensor[batch_idx]
                
                batch_x_linear = x_linear[batch_idx]
                batch_y = y[batch_idx]
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(batch_x_dict, batch_x_linear)
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                
                # 그래디언트 클리핑 (소규모 데이터에서 안정성 향상)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # 통계 수집
                epoch_loss += loss.item()
                predictions = (outputs > 0.5).float()
                epoch_correct += (predictions == batch_y).sum().item()
                epoch_total += batch_y.size(0)
                num_batches += 1
            
            avg_train_loss = epoch_loss / num_batches
            train_accuracy = epoch_correct / epoch_total
            
            # 히스토리 저장
            train_history['loss'].append(avg_train_loss)
            train_history['accuracy'].append(train_accuracy)
            
            # 최적 모델 저장 (학습 손실 기준)
            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
                best_model_state = copy.deepcopy(self.model.state_dict())
            
            # 로그 출력 (더 빈번하게)
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}] - 손실: {avg_train_loss:.4f}, 정확도: {train_accuracy:.4f}')
        
        # 최적 모델 복원
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f'최적 모델 복원 완료 (최적 학습 손실: {best_train_loss:.4f})')
        
        # ===== 최종 테스트 평가 =====
        print("\n=== 최종 테스트 평가 ===")
        self.model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        all_test_outputs = []
        all_test_targets = []
        
        with torch.no_grad():
            # 테스트 데이터도 배치로 처리
            for i in range(0, len(test_idx), batch_size):
                end_idx = min(i + batch_size, len(test_idx))
                batch_idx = test_idx[i:end_idx]
                
                # 배치 데이터 준비
                batch_x_dict = {}
                for key, tensor in x_dict.items():
                    batch_x_dict[key] = tensor[batch_idx]
                
                batch_x_linear = x_linear[batch_idx]
                batch_y = y[batch_idx]
                
                # Forward pass
                outputs = self.model(batch_x_dict, batch_x_linear)
                loss = criterion(outputs, batch_y)
                
                # 통계 수집
                test_loss += loss.item() * len(batch_idx)  # 가중평균을 위해
                predictions = (outputs > 0.5).float()
                test_correct += (predictions == batch_y).sum().item()
                test_total += batch_y.size(0)
                
                # 상세 분석용 데이터 수집
                all_test_outputs.extend(outputs.cpu().numpy())
                all_test_targets.extend(batch_y.cpu().numpy())
        
        avg_test_loss = test_loss / len(test_idx)
        test_accuracy = test_correct / test_total
        
        print(f'최종 테스트 결과:')
        print(f'  손실: {avg_test_loss:.4f}')
        print(f'  정확도: {test_accuracy:.4f}')
        
        # 추가 평가 지표 (테스트 세트가 충분히 클 때만)
        if len(test_idx) > 10:  # 최소 10개 이상의 테스트 샘플이 있을 때
            try:
                from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
                
                test_predictions = np.array(all_test_outputs) > 0.5
                test_targets = np.array(all_test_targets)
                
                # 클래스가 모두 포함되어 있는지 확인
                if len(np.unique(test_targets)) > 1:
                    precision = precision_score(test_targets, test_predictions, zero_division=0)
                    recall = recall_score(test_targets, test_predictions, zero_division=0)
                    f1 = f1_score(test_targets, test_predictions, zero_division=0)
                    auc = roc_auc_score(test_targets, all_test_outputs)
                    
                    print(f'  정밀도: {precision:.4f}')
                    print(f'  재현율: {recall:.4f}')
                    print(f'  F1 점수: {f1:.4f}')
                    print(f'  AUC: {auc:.4f}')
                else:
                    print('  테스트 세트에 한 클래스만 포함되어 상세 평가 생략')
                    
            except Exception as e:
                print(f'  상세 평가 중 오류: {str(e)}')
        
        # 학습 곡선 시각화 (단순화)
        self._plot_simple_training_curve(train_history)
        
        # 결과 저장
        self.train_history = train_history
        self.test_metrics = {
            'loss': avg_test_loss,
            'accuracy': test_accuracy
        }
        
        return self.model, scaler
    
    def _plot_simple_training_curve(self, train_history: Dict):
        """간단한 학습 곡선 시각화"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        epochs = range(1, len(train_history['loss']) + 1)
        
        # 손실 곡선
        ax1.plot(epochs, train_history['loss'], 'b-', linewidth=2, label='학습 손실')
        ax1.set_title('학습 손실 변화')
        ax1.set_xlabel('에포크')
        ax1.set_ylabel('손실')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 정확도 곡선
        ax2.plot(epochs, train_history['accuracy'], 'g-', linewidth=2, label='학습 정확도')
        ax2.set_title('학습 정확도 변화')
        ax2.set_xlabel('에포크')
        ax2.set_ylabel('정확도')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("학습 곡선이 'training_curve.png'로 저장되었습니다.")
    
    def recommend_chairs_for_all_users(self, k: int = 5) -> Dict[int, List[Dict]]:
        """모든 사용자에 대한 Top-K 의자 추천"""
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        all_recommendations = {}
        
        print(f"\n=== 모든 사용자에 대한 Top-{k} 의자 추천 ===")
        
        # 각 사용자별로 추천 수행
        for person_idx, person_row in self.person_df.iterrows():
            print(f"\n사용자 {person_idx+1} (키: {person_row['human_height']:.1f}cm) 추천 중...")
            
            # 사용자 특성 추출
            person_features = {}
            for col in self.person_df.columns:
                if col != 'image-name':  # 이미지명 제외
                    clean_col = col.replace(' ', '').replace('-', '_')
                    person_features[clean_col] = person_row[col]
            
            # 모든 의자에 대해 호환성 점수 계산
            chair_scores = []
            
            for chair_idx, chair_row in self.chair_df.iterrows():
                # 호환성 점수 계산
                compatibility_score = self._calculate_compatibility(person_features, chair_row.to_dict())
                
                # 의자 정보와 점수 저장
                chair_info = chair_row.to_dict()
                chair_info['chair_id'] = chair_idx
                chair_info['compatibility_score'] = compatibility_score
                
                chair_scores.append((chair_idx, compatibility_score, chair_info))
            
            # 점수 기준 정렬 (높은 점수 순)
            chair_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Top-K 추천 생성
            top_k_recommendations = []
            for i in range(min(k, len(chair_scores))):
                chair_idx, score, chair_info = chair_scores[i]
                
                recommendation = {
                    'rank': i + 1,
                    'chair_id': chair_idx,
                    'compatibility_score': score,
                    'chair_specs': chair_info
                }
                
                top_k_recommendations.append(recommendation)
            
            # 사용자별 추천 결과 저장
            all_recommendations[person_idx] = {
                'user_info': person_features,
                'recommendations': top_k_recommendations
            }
            
            # 각 사용자의 Top-3 추천 출력
            print(f"  Top-3 추천 의자:")
            for rec in top_k_recommendations[:3]:
                print(f"    {rec['rank']}순위: 의자 {rec['chair_id']}, 호환성: {rec['compatibility_score']:.3f}")
        
        return all_recommendations
    
    def save_recommendations_to_file(self, recommendations: Dict[int, List[Dict]], filename: str = "chair_recommendations.txt"):
        """추천 결과를 파일로 저장"""
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=== 개인 맞춤형 의자 추천 결과 ===\n\n")
            
            for person_idx, user_data in recommendations.items():
                user_info = user_data['user_info']
                user_recommendations = user_data['recommendations']
                
                f.write(f"👤 사용자 {person_idx + 1}\n")
                f.write(f"   키: {user_info.get('human_height', 'N/A'):.1f}cm\n")
                f.write(f"   허벅지 길이: {user_info.get('A_Buttockpopliteal_length', 'N/A'):.1f}cm\n")
                f.write(f"   다리 높이: {user_info.get('B_Popliteal_height', 'N/A'):.1f}cm\n")
                f.write(f"   엉덩이 너비: {user_info.get('C_Hip_breadth', 'N/A'):.1f}cm\n\n")
                
                f.write(f"🪑 추천 의자 목록:\n")
                for rec in user_recommendations:
                    f.write(f"   {rec['rank']}순위: 의자 ID {rec['chair_id']}\n")
                    f.write(f"      호환성 점수: {rec['compatibility_score']:.3f}\n")
                    
                    # 주요 의자 스펙 출력
                    specs = rec['chair_specs']
                    if 'h8_지면좌석_높이_mean' in specs:
                        f.write(f"      좌석 높이: {specs['h8_지면좌석_높이_mean']:.1f}cm\n")
                    if 't4_mean' in specs:
                        f.write(f"      좌석 깊이: {specs['t4_mean']:.1f}cm\n")
                    if 'b3_좌석_가로_길이_mean' in specs:
                        f.write(f"      좌석 너비: {specs['b3_좌석_가로_길이_mean']:.1f}cm\n")
                    f.write("\n")
                
                f.write("-" * 50 + "\n\n")
        
        print(f"추천 결과가 '{filename}' 파일로 저장되었습니다.")
    
    def create_recommendation_summary(self, recommendations: Dict[int, List[Dict]]) -> pd.DataFrame:
        """추천 결과 요약 테이블 생성"""
        
        summary_data = []
        
        for person_idx, user_data in recommendations.items():
            user_info = user_data['user_info']
            user_recommendations = user_data['recommendations']
            
            # 각 사용자의 Top-5 추천을 한 행으로 정리
            row_data = {
                'user_id': person_idx,
                'user_height': user_info.get('human_height', 0),
                'user_thigh_length': user_info.get('A_Buttockpopliteal_length', 0),
                'user_leg_height': user_info.get('B_Popliteal_height', 0),
            }
            
            # Top-5 의자 ID와 점수 추가
            for i in range(5):
                if i < len(user_recommendations):
                    rec = user_recommendations[i]
                    row_data[f'top_{i+1}_chair_id'] = rec['chair_id']
                    row_data[f'top_{i+1}_score'] = round(rec['compatibility_score'], 3)
                else:
                    row_data[f'top_{i+1}_chair_id'] = None
                    row_data[f'top_{i+1}_score'] = None
            
            summary_data.append(row_data)
        
        summary_df = pd.DataFrame(summary_data)
        return summary_df
    
    def visualize_recommendations(self, recommendations: Dict[int, List[Dict]]):
        """추천 결과 시각화"""
        
        # 데이터 준비
        user_ids = []
        chair_ids = []
        scores = []
        ranks = []
        
        for person_idx, user_data in recommendations.items():
            for rec in user_data['recommendations']:
                user_ids.append(f'사용자{person_idx+1}')
                chair_ids.append(f'의자{rec["chair_id"]}')
                scores.append(rec['compatibility_score'])
                ranks.append(rec['rank'])
        
        # 히트맵 생성
        plt.figure(figsize=(12, 8))
        
        # 사용자별 Top-5 추천 의자의 호환성 점수 히트맵
        pivot_data = []
        users = []
        
        for person_idx, user_data in recommendations.items():
            user_name = f'사용자{person_idx+1}'
            users.append(user_name)
            
            user_scores = []
            for i in range(5):  # Top-5만 표시
                if i < len(user_data['recommendations']):
                    score = user_data['recommendations'][i]['compatibility_score']
                    user_scores.append(score)
                else:
                    user_scores.append(0)
            
            pivot_data.append(user_scores)
        
        pivot_df = pd.DataFrame(pivot_data, 
                               index=users, 
                               columns=[f'1순위', f'2순위', f'3순위', f'4순위', f'5순위'])
        
        sns.heatmap(pivot_df, annot=True, cmap='YlOrRd', fmt='.3f', cbar_kws={'label': '호환성 점수'})
        plt.title('사용자별 Top-5 의자 추천 호환성 점수')
        plt.xlabel('추천 순위')
        plt.ylabel('사용자')
        plt.tight_layout()
        plt.savefig('recommendation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 호환성 점수 분포
        plt.figure(figsize=(10, 6))
        plt.hist(scores, bins=20, alpha=0.7, edgecolor='black')
        plt.title('전체 추천 의자의 호환성 점수 분포')
        plt.xlabel('호환성 점수')
        plt.ylabel('빈도')
        plt.grid(True, alpha=0.3)
        plt.savefig('compatibility_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("시각화 결과가 저장되었습니다:")
        print("- recommendation_heatmap.png: 사용자별 추천 히트맵")
        print("- compatibility_distribution.png: 호환성 점수 분포")
    
    def _calculate_compatibility(self, person_features: Dict, chair_features: Dict) -> float:
        """호환성 점수 계산"""
        
        matching_conditions = {
            'A_Buttockpopliteal_length': ('t4_mean', 'less_than'),
            'B_Popliteal_height': ('h8_지면좌석_높이_mean', 'approximately'),
            'C_Hip_breadth': ('b3_좌석_가로_길이_mean', 'greater_than'),
            'E_Elbow_rest_height': ('h9_좌석팔걸이_높이_mean', 'approximately'),
            'F_Sitting_height': ('h7_등받이_세로_길이_mean', 'less_than'),
            'G_Shoulder_breadth': ('b4_등받이_가로_길이_mean', 'greater_equal')
        }
        
        total_score = 0
        valid_conditions = 0
        
        for person_feature, (chair_feature, condition_type) in matching_conditions.items():
            if person_feature in person_features and chair_feature in chair_features:
                person_val = person_features[person_feature]
                chair_val = chair_features[chair_feature]
                
                if condition_type == 'less_than':
                    score = 1.0 if chair_val < person_val else max(0, 1 - abs(chair_val - person_val) / person_val)
                elif condition_type == 'greater_than':
                    score = 1.0 if chair_val > person_val else max(0, 1 - abs(chair_val - person_val) / person_val)
                elif condition_type == 'greater_equal':
                    score = 1.0 if chair_val >= person_val else max(0, 1 - abs(chair_val - person_val) / person_val)
                elif condition_type == 'approximately':
                    tolerance = 0.1 * person_val
                    score = 1.0 if abs(chair_val - person_val) <= tolerance else max(0, 1 - abs(chair_val - person_val) / person_val)
                
                total_score += score
                valid_conditions += 1
        
        return total_score / valid_conditions if valid_conditions > 0 else 0.0

def main():
    """메인 실행 함수"""
    
    # 시스템 초기화
    system = ChairRecommendationSystem()
    
    try:
        # 데이터 준비
        compatibility_df = system.prepare_data()
        
        # 모델 학습
        model, scaler = system.train_model(compatibility_df)
        
        # 예시 사용자 특성
        example_person = {
            'human_height': 170.0,
            'A_Buttockpopliteal_length': 48.0,
            'B_Popliteal_height': 42.0,
            'C_Hip_breadth': 36.0,
            'E_Elbow_rest_height': 24.0,
            'F_Sitting_height': 85.0,
            'G_Shoulder_breadth': 45.0
        }
        
        print("\n=== 의자 추천 결과 ===")
        recommendations = system.recommend_chairs(example_person, k=5)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"\n추천 {i}순위:")
            print(f"  추천 점수: {rec['recommendation_score']:.3f}")
            print(f"  주요 특성:")
                            for key, value in rec.items():
                    if key not in ['recommendation_score', 'rank'] and isinstance(value, (int, float)):
                        print(f"    {key}: {value:.1f}")
        
        print("\n=== 호환성 분석 ===")
        # 호환성 분포 시각화
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.hist(compatibility_df['compatibility_score'], bins=30, alpha=0.7)
        plt.title('호환성 점수 분포')
        plt.xlabel('호환성 점수')
        plt.ylabel('빈도')
        
        plt.subplot(2, 2, 2)
        label_counts = compatibility_df['label'].value_counts()
        plt.pie(label_counts.values, labels=['비호환', '호환'], autopct='%1.1f%%')
        plt.title('호환성 레이블 분포')
        
        plt.subplot(2, 2, 3)
        person_heights = compatibility_df['person_human_height']
        plt.hist(person_heights, bins=20, alpha=0.7)
        plt.title('사용자 키 분포')
        plt.xlabel('키 (cm)')
        plt.ylabel('빈도')
        
        plt.subplot(2, 2, 4)
        # 키와 호환성 점수의 관계
        plt.scatter(person_heights, compatibility_df['compatibility_score'], alpha=0.5)
        plt.title('키 vs 호환성 점수')
        plt.xlabel('키 (cm)')
        plt.ylabel('호환성 점수')
        
        plt.tight_layout()
        plt.savefig('compatibility_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 매칭 조건별 분석
        condition_columns = [col for col in compatibility_df.columns if '_mean' in col and col.endswith('_mean')]
        if condition_columns:
            plt.figure(figsize=(15, 10))
            for i, col in enumerate(condition_columns[:6], 1):
                plt.subplot(2, 3, i)
                plt.hist(compatibility_df[col], bins=20, alpha=0.7)
                plt.title(f'{col} 점수 분포')
                plt.xlabel('점수')
                plt.ylabel('빈도')
            
            plt.tight_layout()
            plt.savefig('condition_scores_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()

# 추가: 모델 성능 평가 클래스
class ModelEvaluator:
    """모델 성능 평가 클래스"""
    
    def __init__(self):
        pass
    
    def evaluate_recommendations(self, true_labels: List[List[int]], 
                               predicted_rankings: List[List[int]], 
                               k: int = 5) -> Dict[str, float]:
        """추천 시스템 성능 평가"""
        
        metrics = {}
        
        # Precision@K
        precisions = []
        for true_list, pred_list in zip(true_labels, predicted_rankings):
            if len(pred_list) > 0:
                relevant_items = set(true_list[:k])
                recommended_items = set(pred_list[:k])
                precision = len(relevant_items.intersection(recommended_items)) / len(recommended_items)
                precisions.append(precision)
        
        metrics['precision_at_k'] = np.mean(precisions) if precisions else 0.0
        
        # Recall@K  
        recalls = []
        for true_list, pred_list in zip(true_labels, predicted_rankings):
            if len(true_list) > 0 and len(pred_list) > 0:
                relevant_items = set(true_list[:k])
                recommended_items = set(pred_list[:k])
                recall = len(relevant_items.intersection(recommended_items)) / len(relevant_items)
                recalls.append(recall)
        
        metrics['recall_at_k'] = np.mean(recalls) if recalls else 0.0
        
        # NDCG@K (간단한 구현)
        ndcg_scores = []
        for true_list, pred_list in zip(true_labels, predicted_rankings):
            if len(true_list) > 0 and len(pred_list) > 0:
                # DCG 계산
                dcg = 0
                for i, item in enumerate(pred_list[:k]):
                    if item in true_list[:k]:
                        dcg += 1 / np.log2(i + 2)  # i+2 because log2(1) = 0
                
                # IDCG 계산 (이상적인 DCG)
                idcg = sum(1 / np.log2(i + 2) for i in range(min(k, len(true_list))))
                
                ndcg = dcg / idcg if idcg > 0 else 0
                ndcg_scores.append(ndcg)
        
        metrics['ndcg_at_k'] = np.mean(ndcg_scores) if ndcg_scores else 0.0
        
        return metrics
    
    def print_evaluation_results(self, metrics: Dict[str, float], k: int):
        """평가 결과 출력"""
        print(f"\n=== Top-{k} 추천 성능 평가 ===")
        print(f"Precision@{k}: {metrics['precision_at_k']:.4f}")
        print(f"Recall@{k}: {metrics['recall_at_k']:.4f}")  
        print(f"NDCG@{k}: {metrics['ndcg_at_k']:.4f}")

# DRM 구현 개선
class AdvancedDRMRanker:
    """개선된 DRM 랭킹 클래스"""
    
    def __init__(self, temperature: float = 0.5):
        self.temperature = temperature
    
    def neural_sort(self, scores: torch.Tensor) -> torch.Tensor:
        """NeuralSort를 이용한 차별화 가능한 정렬"""
        
        batch_size, n = scores.shape
        device = scores.device
        
        # Permutation matrix 생성을 위한 준비
        scores_expanded = scores.unsqueeze(-1)  # (batch, n, 1)
        scores_transposed = scores.unsqueeze(1)  # (batch, 1, n)
        
        # 모든 쌍에 대한 차이 계산
        pairwise_diff = scores_expanded - scores_transposed  # (batch, n, n)
        
        # Relaxed permutation matrix 계산
        P_hat = torch.softmax(pairwise_diff / self.temperature, dim=-1)
        
        return P_hat
    
    def compute_drm_loss(self, scores: torch.Tensor, 
                        true_rankings: torch.Tensor) -> torch.Tensor:
        """DRM 손실 함수 계산"""
        
        # NeuralSort를 이용한 soft permutation matrix
        P_hat = self.neural_sort(scores)
        
        # True rankings를 one-hot으로 변환
        batch_size, n = scores.shape
        true_rankings_one_hot = torch.zeros(batch_size, n, n, device=scores.device)
        
        for b in range(batch_size):
            for i, rank in enumerate(true_rankings[b]):
                if rank < n:  # 유효한 랭킹인 경우
                    true_rankings_one_hot[b, i, rank] = 1.0
        
        # MSE 손실 계산
        loss = torch.mean((P_hat - true_rankings_one_hot) ** 2)
        
        return loss
    
    def get_top_k_with_drm(self, scores: torch.Tensor, k: int = 5) -> torch.Tensor:
        """DRM을 이용한 Top-K 추출"""
        
        # Soft permutation matrix 계산
        P_hat = self.neural_sort(scores)
        
        # Top-K 위치의 가중합 계산
        batch_size, n = scores.shape
        
        # Top-K 마스크 생성
        top_k_mask = torch.zeros(n, device=scores.device)
        top_k_mask[:k] = 1.0
        
        # 각 항목이 Top-K에 포함될 확률 계산
        top_k_probs = torch.sum(P_hat * top_k_mask.unsqueeze(0).unsqueeze(0), dim=-1)
        
        return top_k_probs

# 고급 특성 엔지니어링 클래스
class AdvancedFeatureEngineer:
    """고급 특성 엔지니어링"""
    
    def __init__(self):
        self.feature_importance = {}
        
    def create_interaction_features(self, person_df: pd.DataFrame, 
                                  chair_df: pd.DataFrame) -> pd.DataFrame:
        """상호작용 특성 생성"""
        
        interaction_features = []
        
        for person_idx, person_row in person_df.iterrows():
            for chair_idx, chair_row in chair_df.iterrows():
                
                features = {
                    'person_id': person_idx,
                    'chair_id': chair_idx
                }
                
                # 비율 기반 특성
                if person_row['B_Popliteal_height'] > 0:
                    features['seat_height_ratio'] = chair_row.get('h8_지면좌석_높이_mean', 0) / person_row['B_Popliteal_height']
                
                if person_row['A_Buttockpopliteal_length'] > 0:
                    features['seat_depth_ratio'] = chair_row.get('t4_mean', 0) / person_row['A_Buttockpopliteal_length']
                
                if person_row['C_Hip_breadth'] > 0:
                    features['seat_width_ratio'] = chair_row.get('b3_좌석_가로_길이_mean', 0) / person_row['C_Hip_breadth']
                
                # 차이 기반 특성
                features['height_diff'] = abs(chair_row.get('h8_지면좌석_높이_mean', 0) - person_row['B_Popliteal_height'])
                features['depth_diff'] = abs(chair_row.get('t4_mean', 0) - person_row['A_Buttockpopliteal_length'])
                features['width_diff'] = abs(chair_row.get('b3_좌석_가로_길이_mean', 0) - person_row['C_Hip_breadth'])
                
                # BMI 스타일 복합 지표
                if person_row['human_height'] > 0:
                    features['body_proportion'] = person_row['F_Sitting_height'] / person_row['human_height']
                
                # 조절성 특성 (범위가 있는 경우)
                for col in chair_row.index:
                    if col.endswith('_range'):
                        base_name = col.replace('_range', '')
                        features[f'{base_name}_adjustability'] = chair_row[col]
                
                interaction_features.append(features)
        
        return pd.DataFrame(interaction_features)
    
    def calculate_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """특성 중요도 계산 (간단한 버전)"""
        
        importance_dict = {}
        
        # 모델의 가중치를 기반으로 중요도 추정
        if hasattr(model, 'linear'):
            linear_weights = model.linear.weight.data.abs().mean().item()
            importance_dict['linear_features'] = linear_weights
        
        if hasattr(model, 'final_layer'):
            final_weights = model.final_layer.weight.data.abs().mean().item()
            importance_dict['combined_features'] = final_weights
        
        return importance_dict

# 실시간 추천 시스템 클래스
class RealTimeRecommender:
    """실시간 추천 시스템"""
    
    def __init__(self, model, chair_df: pd.DataFrame):
        self.model = model
        self.chair_df = chair_df
        self.model.eval()
    
    def quick_recommend(self, user_measurements: Dict[str, float], 
                       k: int = 5, 
                       constraints: Dict = None) -> List[Dict]:
        """빠른 추천 (제약 조건 포함)"""
        
        recommendations = []
        
        # 제약 조건 필터링
        filtered_chairs = self.chair_df.copy()
        
        if constraints:
            if 'max_price' in constraints:
                filtered_chairs = filtered_chairs[filtered_chairs.get('price', 0) <= constraints['max_price']]
            
            if 'color_preference' in constraints:
                filtered_chairs = filtered_chairs[filtered_chairs.get('color', '') == constraints['color_preference']]
            
            if 'brand_preference' in constraints:
                filtered_chairs = filtered_chairs[filtered_chairs.get('brand', '') == constraints['brand_preference']]
        
        scores = []
        
        # 빠른 호환성 점수 계산
        for chair_idx, chair_row in filtered_chairs.iterrows():
            compatibility_score = self._quick_compatibility_score(user_measurements, chair_row.to_dict())
            scores.append((chair_idx, compatibility_score, chair_row.to_dict()))
        
        # 정렬 및 Top-K 선택
        scores.sort(key=lambda x: x[1], reverse=True)
        
        for i in range(min(k, len(scores))):
            chair_idx, score, chair_info = scores[i]
            chair_info['recommendation_score'] = score
            chair_info['rank'] = i + 1
            chair_info['chair_id'] = chair_idx
            recommendations.append(chair_info)
        
        return recommendations
    
    def _quick_compatibility_score(self, user_measurements: Dict[str, float], 
                                 chair_specs: Dict[str, float]) -> float:
        """빠른 호환성 점수 계산"""
        
        # 핵심 매칭 조건만 빠르게 계산
        quick_conditions = [
            ('B_Popliteal_height', 'h8_지면좌석_높이_mean', 'approximately'),
            ('A_Buttockpopliteal_length', 't4_mean', 'less_than'),
            ('C_Hip_breadth', 'b3_좌석_가로_길이_mean', 'greater_than')
        ]
        
        total_score = 0
        valid_conditions = 0
        
        for user_key, chair_key, condition_type in quick_conditions:
            if user_key in user_measurements and chair_key in chair_specs:
                user_val = user_measurements[user_key]
                chair_val = chair_specs[chair_key]
                
                if condition_type == 'approximately':
                    tolerance = 0.15 * user_val  # 15% 허용
                    score = 1.0 if abs(chair_val - user_val) <= tolerance else max(0, 1 - abs(chair_val - user_val) / user_val)
                elif condition_type == 'less_than':
                    score = 1.0 if chair_val < user_val else max(0, 1 - (chair_val - user_val) / user_val)
                elif condition_type == 'greater_than':
                    score = 1.0 if chair_val > user_val else max(0, 1 - (user_val - chair_val) / user_val)
                
                total_score += score
                valid_conditions += 1
        
        return total_score / valid_conditions if valid_conditions > 0 else 0.0

# 실행 부분에 평가 추가
def run_comprehensive_evaluation():
    """종합 평가 실행"""
    
    print("\n" + "="*60)
    print("종합 평가 및 추가 분석")
    print("="*60)
    
    # 모의 데이터로 평가 테스트
    evaluator = ModelEvaluator()
    
    # 모의 true labels와 predictions
    true_labels = [
        [0, 1, 2, 3, 4],  # 사용자 1의 실제 선호 의자들
        [1, 3, 4, 7, 9],  # 사용자 2의 실제 선호 의자들
        [0, 2, 5, 8, 6]   # 사용자 3의 실제 선호 의자들
    ]
    
    predicted_rankings = [
        [0, 2, 1, 5, 3],  # 모델이 예측한 사용자 1의 추천 순서
        [1, 4, 3, 2, 7],  # 모델이 예측한 사용자 2의 추천 순서
        [2, 0, 5, 1, 8]   # 모델이 예측한 사용자 3의 추천 순서
    ]
    
    # 다양한 K값에 대해 평가
    for k in [3, 5, 10]:
        metrics = evaluator.evaluate_recommendations(true_labels, predicted_rankings, k=k)
        evaluator.print_evaluation_results(metrics, k=k)

def demonstrate_advanced_features():
    """고급 기능 시연"""
    
    print("\n" + "="*60)
    print("고급 기능 시연")
    print("="*60)
    
    # 실시간 추천 시연
    print("\n=== 실시간 추천 시연 ===")
    
    # 모의 의자 데이터
    mock_chair_df = pd.DataFrame([
        {'h8_지면좌석_높이_mean': 42, 't4_mean': 45, 'b3_좌석_가로_길이_mean': 50, 'price': 300000, 'brand': 'A'},
        {'h8_지면좌석_높이_mean': 45, 't4_mean': 48, 'b3_좌석_가로_길이_mean': 52, 'price': 500000, 'brand': 'B'},
        {'h8_지면좌석_높이_mean': 40, 't4_mean': 44, 'b3_좌석_가로_길이_mean': 48, 'price': 200000, 'brand': 'C'},
    ])
    
    # 모의 모델 (실제로는 학습된 모델 사용)
    class MockModel:
        def eval(self): pass
    
    mock_model = MockModel()
    realtime_recommender = RealTimeRecommender(mock_model, mock_chair_df)
    
    # 사용자 측정값
    user_measurements = {
        'B_Popliteal_height': 43.0,
        'A_Buttockpopliteal_length': 47.0,
        'C_Hip_breadth': 49.0
    }
    
    # 제약 조건
    constraints = {
        'max_price': 400000,
        'brand_preference': 'A'
    }
    
    recommendations = realtime_recommender.quick_recommend(
        user_measurements, 
        k=3, 
        constraints=constraints
    )
    
    print("추천 결과:")
    for rec in recommendations:
        print(f"  순위 {rec['rank']}: 점수 {rec['recommendation_score']:.3f}, 브랜드 {rec.get('brand', 'N/A')}")

if __name__ == "__main__":
    # 메인 실행
    main()
    
    # 종합 평가 실행
    run_comprehensive_evaluation()
    
    # 고급 기능 시연
    demonstrate_advanced_features()
    
    print("\n" + "="*60)
    print("의자 추천 시스템 실행 완료!")
    print("생성된 파일:")
    print("- compatibility_analysis.png")
    print("- condition_scores_analysis.png")
    print("="*60)
                