==============
"""
메타데이터 기반 정밀 인체 측정 시스템
- DepthPro JSON 메타데이터 직접 활용
- 유클리디안 거리 계산 추가
- 체형별 비율 조정 시스템
- 발 높이 정밀 측정 (발바닥-발목)
- 의자 제작 기준 출력 (A~G 표준)
"""

import numpy as np
import json
import os
import csv
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path
from datetime import datetime

class EnhancedMeasurementSystem:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        
        # 체형별 신체 비율 (실제 데이터 기반)
        self.body_type_proportions = {
            'inverted_triangle': {
                'shoulder_to_height': 0.234,
                'torso_to_height': 0.405,
                'hip_to_height': 0.202,
                'shoulder_to_hip': 1.158
            },
            'triangle': {
                'shoulder_to_height': 0.230,
                'torso_to_height': 0.406,
                'hip_to_height': 0.204,
                'shoulder_to_hip': 1.132
            },
            'rectangle': {
                'shoulder_to_height': 0.228,
                'torso_to_height': 0.402,
                'hip_to_height': 0.201,
                'shoulder_to_hip': 1.132
            }
        }
        
        # 기본 비율 (체형 판별 전 사용)
        self.default_proportions = {
            'shoulder_to_height': 0.231,
            'torso_to_height': 0.404,
            'hip_to_height': 0.202,
            'thigh_to_height': 0.24,
            'calf_to_height': 0.23,
            'foot_to_height': 0.06
        }
        
        # 키포인트 인덱스
        self.keypoint_ids = {
            'nose': 0,
            'left_shoulder': 5, 'right_shoulder': 6,
            'left_hip': 9, 'right_hip': 10,
            'left_knee': 11, 'right_knee': 12,
            'left_ankle': 13, 'right_ankle': 14,
            'left_big_toe': 15, 'right_big_toe': 18,
            'left_heel': 17, 'right_heel': 20,
            'left_acromion': 67, 'right_acromion': 68,
            'neck': 69
        }
        
        # 의자 제작 표준 매핑
        self.chair_measurements = {
            'A_Buttock_popliteal_length': 'thigh_length',      # 좌석 깊이
            'B_Popliteal_height': 'calf_foot_combined',        # 앉은 다리 높이
            'C_Hip_breadth': 'hip_width',                      # 엉덩이 폭
            'F_Sitting_height': 'torso_length',                # 앉은 키
            'G_Shoulder_breadth': 'shoulder_width'             # 어깨 너비
        }
        
    def load_data(self, filename: str, category: str = "validation") -> Dict[str, Any]:
        """데이터 로드 (메타데이터 포함)"""
        try:
            pose_json = self.base_dir / "pose" / category / f"{filename}.json"
            depthpro_json = self.base_dir / "depth" / category / f"{filename}.json"
            depthpro_npy = self.base_dir / "depth" / category / f"{filename}.npy"
            sapiens_depth_npy = self.base_dir / "sapiensdepth" / category / f"{filename}.npy"
            
            # Sapiens 키포인트
            with open(pose_json, 'r', encoding='utf-8') as f:
                pose_data = json.load(f)
            keypoints_raw = np.array(pose_data['instance_info'][0]['keypoints'])
            keypoints = keypoints_raw.reshape(-1, 2) if keypoints_raw.shape != (308, 2) else keypoints_raw
            
            # DepthPro + 메타데이터
            with open(depthpro_json, 'r', encoding='utf-8') as f:
                depthpro_metadata = json.load(f)
            depthpro = np.load(depthpro_npy)
            
            # Sapiens Depth
            sapiens_depth = np.load(sapiens_depth_npy)
            
            focal_length = depthpro_metadata.get('focal_length', 0)
            depth_range = depthpro_metadata.get('depth_range', {'min': 0.5, 'max': 10.0})
            
            if focal_length <= 0:
                return {}
            
            return {
                'filename': filename,
                'keypoints': keypoints,
                'depthpro': depthpro,
                'sapiens_depth': sapiens_depth,
                'focal_length': focal_length,
                'depth_range': depth_range,
                'metadata': depthpro_metadata
            }
            
        except Exception as e:
            return {}

    def create_fusion_depth(self, depthpro: np.ndarray, sapiens_depth: np.ndarray) -> np.ndarray:
        """DepthPro 중심 융합"""
        try:
            sapiens_q25, sapiens_q75 = np.percentile(sapiens_depth, [25, 75])
            depthpro_q25, depthpro_q75 = np.percentile(depthpro, [25, 75])
            
            if sapiens_q75 != sapiens_q25:
                scale = (depthpro_q75 - depthpro_q25) / (sapiens_q75 - sapiens_q25)
                offset = depthpro_q25 - scale * sapiens_q25
                sapiens_aligned = sapiens_depth * scale + offset
            else:
                sapiens_aligned = np.full_like(sapiens_depth, np.mean(depthpro))
            
            return 0.75 * depthpro + 0.25 * sapiens_aligned
            
        except:
            return depthpro.copy()

    def find_measurement_points(self, sapiens_depth: np.ndarray, keypoints: np.ndarray) -> Dict[str, np.ndarray]:
        """측정점 탐지"""
        try:
            points = {}
            
            # 기본 키포인트들
            for name, idx in self.keypoint_ids.items():
                if idx < len(keypoints):
                    points[name] = keypoints[idx].copy()
            
            # 정밀한 머리정수리
            if 'nose' in points:
                nose = points['nose']
                search_y_min = max(0, int(nose[1] - 50))
                search_y_max = int(nose[1])
                search_x_min = max(0, int(nose[0] - 25))
                search_x_max = min(sapiens_depth.shape[1], int(nose[0] + 25))
                
                search_region = sapiens_depth[search_y_min:search_y_max, search_x_min:search_x_max]
                if search_region.size > 0:
                    min_idx = np.unravel_index(np.argmin(search_region), search_region.shape)
                    points['head_top'] = np.array([search_x_min + min_idx[1], search_y_min + min_idx[0]])
                else:
                    points['head_top'] = nose
            
            # 정밀한 발바닥 (최저점)
            foot_points = []
            for name in ['left_big_toe', 'right_big_toe', 'left_ankle', 'right_ankle']:
                if name in points:
                    foot_points.append(points[name])
            
            if foot_points:
                foot_points = np.array(foot_points)
                lowest_foot = foot_points[np.argmax(foot_points[:, 1])]
                
                search_y_min = int(lowest_foot[1])
                search_y_max = min(sapiens_depth.shape[0], int(lowest_foot[1] + 40))
                search_x_min = max(0, int(lowest_foot[0] - 20))
                search_x_max = min(sapiens_depth.shape[1], int(lowest_foot[0] + 20))
                
                search_region = sapiens_depth[search_y_min:search_y_max, search_x_min:search_x_max]
                if search_region.size > 0:
                    max_idx = np.unravel_index(np.argmax(search_region), search_region.shape)
                    points['foot_bottom'] = np.array([search_x_min + max_idx[1], search_y_min + max_idx[0]])
                else:
                    points['foot_bottom'] = lowest_foot
            
            return points
            
        except Exception as e:
            return {}

    def calculate_euclidean_distance_3d(self, p1: np.ndarray, p2: np.ndarray, 
                                       depth1: float, depth2: float, focal_length: float) -> float:
        """3D 유클리디안 거리 계산"""
        try:
            # 2D 픽셀 거리
            pixel_dist = np.linalg.norm(p2 - p1)
            
            # 평균 depth 사용하여 실제 거리 변환
            avg_depth = (depth1 + depth2) / 2
            real_2d_dist = (pixel_dist * avg_depth) / focal_length * 100  # cm 단위
            
            # depth 차이로 z축 거리
            depth_diff = abs(depth2 - depth1) * 100  # cm 단위
            
            # 3D 유클리디안 거리
            euclidean_3d = np.sqrt(real_2d_dist**2 + depth_diff**2)
            
            return euclidean_3d
            
        except Exception as e:
            return 0

    def calculate_foot_height(self, ankle_point: np.ndarray, foot_bottom_point: np.ndarray,
                             fusion_depth: np.ndarray, focal_length: float) -> float:
        """발 높이 측정 (발바닥-발목 수직 거리)"""
        try:
            def get_depth(point):
                x, y = int(np.clip(point[0], 0, fusion_depth.shape[1]-1)), \
                       int(np.clip(point[1], 0, fusion_depth.shape[0]-1))
                return fusion_depth[y, x]
            
            ankle_depth = get_depth(ankle_point)
            foot_depth = get_depth(foot_bottom_point)
            
            # 유클리디안 거리로 발 높이 계산
            foot_height = self.calculate_euclidean_distance_3d(
                ankle_point, foot_bottom_point, ankle_depth, foot_depth, focal_length
            )
            
            # 합리적 범위 제한 (3~15cm)
            return np.clip(foot_height, 3.0, 15.0)
            
        except Exception as e:
            return 6.0  # 기본값

    def determine_body_type(self, shoulder_width: float, hip_width: float) -> str:
        """체형 판별 (어깨/골반 비율 기준)"""
        try:
            if shoulder_width <= 0 or hip_width <= 0:
                return 'rectangle'  # 기본값
            
            shoulder_hip_ratio = shoulder_width / hip_width
            
            if shoulder_hip_ratio > 1.15:
                return 'inverted_triangle'
            elif shoulder_hip_ratio < 1.13:
                return 'triangle'
            else:
                return 'rectangle'
                
        except:
            return 'rectangle'

    def calculate_metadata_based_height(self, points: Dict[str, np.ndarray], 
                                       fusion_depth: np.ndarray, metadata: Dict) -> Tuple[float, Dict]:
        """메타데이터 기반 정밀 키 측정"""
        try:
            focal_length = metadata.get('focal_length', 0)
            depth_range = metadata.get('depth_range', {})
            
            def get_depth(point):
                x, y = int(np.clip(point[0], 0, fusion_depth.shape[1]-1)), \
                       int(np.clip(point[1], 0, fusion_depth.shape[0]-1))
                return fusion_depth[y, x]
            
            if not ('head_top' in points and 'foot_bottom' in points):
                return 0, {}
            
            # 유클리디안 거리로 키 측정
            head_depth = get_depth(points['head_top'])
            foot_depth = get_depth(points['foot_bottom'])
            direct_height = self.calculate_euclidean_distance_3d(
                points['head_top'], points['foot_bottom'], head_depth, foot_depth, focal_length
            )
            
            # 메타데이터 기반 보정
            min_depth = depth_range.get('min', 0.5)
            max_depth = depth_range.get('max', 10.0)
            depth_span = max_depth - min_depth
            
            if focal_length > 1600:  # 망원
                correction_factor = 0.95 if depth_span < 2.0 else 1.02
            elif focal_length > 1400:  # 표준
                correction_factor = 0.98 if depth_span < 2.5 else 1.05
            else:  # 광각
                correction_factor = 1.08 if depth_span < 2.0 else 1.12
            
            corrected_height = direct_height * correction_factor
            
            height_info = {
                'direct_height': direct_height,
                'corrected_height': corrected_height,
                'correction_factor': correction_factor,
                'focal_length': focal_length,
                'depth_span': depth_span,
                'final_height': corrected_height
            }
            
            return corrected_height, height_info
            
        except Exception as e:
            return 0, {}

    def calculate_measurements(self, points: Dict[str, np.ndarray], 
                             fusion_depth: np.ndarray, focal_length: float, 
                             depth_range: Dict, metadata: Dict) -> Dict[str, Any]:
        """메타데이터 기반 전체 측정"""
        try:
            def get_depth(point):
                x, y = int(np.clip(point[0], 0, fusion_depth.shape[1]-1)), \
                       int(np.clip(point[1], 0, fusion_depth.shape[0]-1))
                return fusion_depth[y, x]
            
            measurements = {}
            
            # 1. 키 측정
            height_cm, height_info = self.calculate_metadata_based_height(points, fusion_depth, metadata)
            if height_cm > 0:
                measurements['height'] = {
                    'cm': height_cm,
                    'method': 'euclidean_3d_metadata',
                    **height_info
                }
            
            # 2. 어깨폭 (견봉 포함)
            if 'left_shoulder' in points and 'right_shoulder' in points:
                left_point = points['left_shoulder'].copy()
                right_point = points['right_shoulder'].copy()
                
                if 'left_acromion' in points:
                    left_point = (left_point + points['left_acromion']) / 2
                if 'right_acromion' in points:
                    right_point = (right_point + points['right_acromion']) / 2
                
                left_depth = get_depth(left_point)
                right_depth = get_depth(right_point)
                shoulder_cm = self.calculate_euclidean_distance_3d(
                    left_point, right_point, left_depth, right_depth, focal_length
                )
                
                measurements['shoulder_width'] = {
                    'cm': shoulder_cm,
                    'method': 'euclidean_3d_acromion'
                }
            
            # 3. 골반폭
            if 'left_hip' in points and 'right_hip' in points:
                left_depth = get_depth(points['left_hip'])
                right_depth = get_depth(points['right_hip'])
                hip_cm = self.calculate_euclidean_distance_3d(
                    points['left_hip'], points['right_hip'], left_depth, right_depth, focal_length
                )
                
                # focal_length 기반 보정
                if focal_length > 1600:
                    hip_correction = 1.8
                elif focal_length > 1400:
                    hip_correction = 1.6
                else:
                    hip_correction = 1.4
                
                hip_cm *= hip_correction
                
                measurements['hip_width'] = {
                    'cm': hip_cm,
                    'method': 'euclidean_3d_corrected'
                }
            
            # 체형 판별 및 비율 조정
            body_type = 'rectangle'  # 기본값
            if 'shoulder_width' in measurements and 'hip_width' in measurements:
                body_type = self.determine_body_type(
                    measurements['shoulder_width']['cm'],
                    measurements['hip_width']['cm']
                )
                current_proportions = self.body_type_proportions[body_type]
                
                # 체형별 비율로 보정
                if height_cm > 0:
                    expected_shoulder = height_cm * current_proportions['shoulder_to_height']
                    expected_hip = height_cm * current_proportions['hip_to_height']
                    
                    measurements['shoulder_width']['cm'] = (
                        0.7 * measurements['shoulder_width']['cm'] + 0.3 * expected_shoulder
                    )
                    measurements['hip_width']['cm'] = (
                        0.7 * measurements['hip_width']['cm'] + 0.3 * expected_hip
                    )
            
            measurements['body_type'] = body_type
            
            # 4. 토르소 (체형별 비율 적용)
            if all(point in points for point in ['neck', 'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']):
                neck = points['neck']
                left_shoulder = points['left_shoulder']
                right_shoulder = points['right_shoulder']
                left_hip = points['left_hip']
                right_hip = points['right_hip']
                
                shoulder_center = (left_shoulder + right_shoulder) / 2
                shoulder_line = right_shoulder - left_shoulder
                neck_to_left = neck - left_shoulder
                
                if np.linalg.norm(shoulder_line) > 0:
                    projection = np.dot(neck_to_left, shoulder_line) / np.dot(shoulder_line, shoulder_line)
                    projection = np.clip(projection, 0, 1)
                    top_center = left_shoulder + projection * shoulder_line
                else:
                    top_center = shoulder_center
                
                bottom_center = (left_hip + right_hip) / 2
                top_depth = get_depth(top_center)
                bottom_depth = get_depth(bottom_center)
                torso_length = self.calculate_euclidean_distance_3d(
                    top_center, bottom_center, top_depth, bottom_depth, focal_length
                )
                
                # 체형별 토르소 비율 적용
                if height_cm > 0 and body_type in self.body_type_proportions:
                    expected_torso = height_cm * self.body_type_proportions[body_type]['torso_to_height']
                    torso_length = 0.6 * torso_length + 0.4 * expected_torso
                
                measurements['torso_length'] = {
                    'cm': torso_length,
                    'method': f'euclidean_3d_{body_type}_adjusted'
                }
            
            # 5. 허벅지 (좌우 평균)
            thigh_lengths = []
            if 'left_hip' in points and 'left_knee' in points:
                left_thigh = self.calculate_euclidean_distance_3d(
                    points['left_hip'], points['left_knee'],
                    get_depth(points['left_hip']), get_depth(points['left_knee']), focal_length
                )
                thigh_lengths.append(left_thigh)
            
            if 'right_hip' in points and 'right_knee' in points:
                right_thigh = self.calculate_euclidean_distance_3d(
                    points['right_hip'], points['right_knee'],
                    get_depth(points['right_hip']), get_depth(points['right_knee']), focal_length
                )
                thigh_lengths.append(right_thigh)
            
            if thigh_lengths:
                avg_thigh = np.mean(thigh_lengths)
                measurements['thigh_length'] = {
                    'cm': avg_thigh,
                    'method': 'euclidean_3d_bilateral'
                }
            
            # 6. 종아리 (좌우 평균)
            calf_lengths = []
            if 'left_knee' in points and 'left_ankle' in points:
                left_calf = self.calculate_euclidean_distance_3d(
                    points['left_knee'], points['left_ankle'],
                    get_depth(points['left_knee']), get_depth(points['left_ankle']), focal_length
                )
                calf_lengths.append(left_calf)
            
            if 'right_knee' in points and 'right_ankle' in points:
                right_calf = self.calculate_euclidean_distance_3d(
                    points['right_knee'], points['right_ankle'],
                    get_depth(points['right_knee']), get_depth(points['right_ankle']), focal_length
                )
                calf_lengths.append(right_calf)
            
            if calf_lengths:
                avg_calf = np.mean(calf_lengths)
                measurements['calf_length'] = {
                    'cm': avg_calf,
                    'method': 'euclidean_3d_bilateral'
                }
            
            # 7. 발 높이 (발바닥-발목)
            foot_heights = []
            
            # 왼발
            if 'left_ankle' in points and 'foot_bottom' in points:
                left_foot_height = self.calculate_foot_height(
                    points['left_ankle'], points['foot_bottom'], fusion_depth, focal_length
                )
                if left_foot_height > 0:
                    foot_heights.append(left_foot_height)
            
            # 오른발
            if 'right_ankle' in points and 'foot_bottom' in points:
                right_foot_height = self.calculate_foot_height(
                    points['right_ankle'], points['foot_bottom'], fusion_depth, focal_length
                )
                if right_foot_height > 0:
                    foot_heights.append(right_foot_height)
            
            if foot_heights:
                avg_foot_height = np.mean(foot_heights)
                measurements['foot_height'] = {
                    'cm': avg_foot_height,
                    'method': 'euclidean_3d_ankle_to_bottom'
                }
            
            # 8. 의자 제작용 복합 측정값
            # B_Popliteal_height = 종아리 + 발높이
            if 'calf_length' in measurements and 'foot_height' in measurements:
                calf_foot_combined = measurements['calf_length']['cm'] + measurements['foot_height']['cm']
                measurements['calf_foot_combined'] = {
                    'cm': calf_foot_combined,
                    'method': 'calf_plus_foot_height'
                }
            
            return measurements
            
        except Exception as e:
            return {}

    def convert_to_chair_format(self, measurements: Dict[str, Any], filename: str) -> Dict[str, Any]:
        """의자 제작 표준 형식으로 변환"""
        try:
            chair_data = {
                'image_name': filename,
                'human_height': measurements.get('height', {}).get('cm', 0),
                'A_Buttock_popliteal_length': measurements.get('thigh_length', {}).get('cm', 0),
                'B_Popliteal_height': measurements.get('calf_foot_combined', {}).get('cm', 0),
                'C_Hip_breadth': measurements.get('hip_width', {}).get('cm', 0),
                'F_Sitting_height': measurements.get('torso_length', {}).get('cm', 0),
                'G_Shoulder_breadth': measurements.get('shoulder_width', {}).get('cm', 0),
                'body_type': measurements.get('body_type', 'rectangle'),
                'metadata': {
                    'focal_length': measurements.get('height', {}).get('focal_length', 0),
                    'depth_span': measurements.get('height', {}).get('depth_span', 0),
                    'correction_factor': measurements.get('height', {}).get('correction_factor', 1.0)
                }
            }
            
            return chair_data
            
        except Exception as e:
            return {}

    def measure_single_image(self, filename: str, category: str = "validation") -> Dict[str, Any]:
        """단일 이미지 측정"""
        try:
            data = self.load_data(filename, category)
            if not data:
                return {}
            
            fusion_depth = self.create_fusion_depth(data['depthpro'], data['sapiens_depth'])
            points = self.find_measurement_points(data['sapiens_depth'], data['keypoints'])
            if not points:
                return {}
            
            measurements = self.calculate_measurements(points, fusion_depth, 
                                                     data['focal_length'], data['depth_range'], data['metadata'])
            if not measurements:
                return {}
            
            chair_data = self.convert_to_chair_format(measurements, filename)
            
            return {
                'filename': filename,
                'category': category,
                'measurements': measurements,
                'chair_format': chair_data
            }
            
        except Exception as e:
            return {}

    def save_results(self, results: List[Dict], output_dir: str):
        """결과 저장 (JSON + CSV)"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # JSON 저장
            json_file = output_path / f"measurement_results_{timestamp}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            # CSV 저장 (의자 제작 표준 형식)
            csv_file = output_path / f"chair_measurements_{timestamp}.csv"
            
            if results:
                chair_results = [r['chair_format'] for r in results if 'chair_format' in r]
                
                if chair_results:
                    fieldnames = [
                        'image_name', 'human_height', 'A_Buttock_popliteal_length',
                        'B_Popliteal_height', 'C_Hip_breadth', 'F_Sitting_height', 
                        'G_Shoulder_breadth', 'body_type'
                    ]
                    
                    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        
                        for result in chair_results:
                            row = {key: result.get(key, 0) for key in fieldnames}
                            writer.writerow(row)
            
            return json_file, csv_file
            
        except Exception as e:
            return None, None

    def process_all_categories(self) -> List[Dict[str, Any]]:
        """모든 카테고리 처리 (validation, women, men)"""
        categories = ['validation', 'women', 'men']
        all_results = []
        
        print("🔧 Enhanced Measurement System v2.0")
        print("=" * 50)
        
        for category in categories:
            pose_dir = self.base_dir / "pose" / category
            if not pose_dir.exists():
                continue
            
            json_files = list(pose_dir.glob("*.json"))
            image_files = []
            
            for json_file in json_files:
                filename = json_file.stem
                depth_json = self.base_dir / "depth" / category / f"{filename}.json"
                depth_npy = self.base_dir / "depth" / category / f"{filename}.npy"
                sapiens_npy = self.base_dir / "sapiensdepth" / category / f"{filename}.npy"
                
                if all([depth_json.exists(), depth_npy.exists(), sapiens_npy.exists()]):
                    image_files.append(filename)
            
            if not image_files:
                continue
            
            image_files.sort()
            print(f"📂 {category}: {len(image_files)}개 처리중...", end=" ")
            
            category_results = []
            for filename in image_files:
                try:
                    result = self.measure_single_image(filename, category)
                    if result:
                        category_results.append(result)
                        all_results.append(result)
                except:
                    pass
            
            print(f"{len(category_results)}개 완료")
        
        return all_results

def display_results_summary(results: List[Dict[str, Any]]):
    """결과 요약 표시"""
    if not results:
        print("측정 결과 없음")
        return
    
    print(f"측정 결과 ({len(results)}개)")
    print("=" * 90)
    print(f"{'Name':^15} {'키':^6} {'어깨':^6} {'토르소':^6} {'골반':^6} {'허벅지':^6} {'종아리':^6} {'발높이':^6} {'체형':^10}")
    print("-" * 90)
    
    # 카테고리별 정렬
    category_order = {'validation': 1, 'women': 2, 'men': 3}
    results_sorted = sorted(results, key=lambda x: (category_order.get(x['category'], 4), x['filename']))
    
    for result in results_sorted:
        filename = result['filename']
        measurements = result['measurements']
        
        # 측정값 추출
        height = measurements.get('height', {}).get('cm', 0)
        shoulder = measurements.get('shoulder_width', {}).get('cm', 0)
        torso = measurements.get('torso_length', {}).get('cm', 0)
        hip = measurements.get('hip_width', {}).get('cm', 0)
        thigh = measurements.get('thigh_length', {}).get('cm', 0)
        calf = measurements.get('calf_length', {}).get('cm', 0)
        foot_height = measurements.get('foot_height', {}).get('cm', 0)
        body_type = measurements.get('body_type', 'rectangle')
        
        print(f"{filename:^15} {height:^6.0f} {shoulder:^6.0f} {torso:^6.0f} {hip:^6.0f} {thigh:^6.0f} {calf:^6.0f} {foot_height:^6.0f} {body_type:^10}")

def display_chair_format_summary(results: List[Dict[str, Any]]):
    """의자 제작 형식 요약 표시"""
    if not results:
        return
    
    print(f"\n🪑 의자 제작 표준 ({len(results)}개)")
    print("=" * 85)
    print(f"{'Name':^15} {'Height':^7} {'A_좌석':^7} {'B_다리':^7} {'C_골반':^7} {'F_등받이':^8} {'G_어깨':^7} {'체형':^8}")
    print("-" * 85)
    
    for result in results:
        chair_data = result.get('chair_format', {})
        
        image_name = chair_data.get('image_name', '')[:13]
        height = chair_data.get('human_height', 0)
        a_buttock = chair_data.get('A_Buttock_popliteal_length', 0)
        b_popliteal = chair_data.get('B_Popliteal_height', 0)
        c_hip = chair_data.get('C_Hip_breadth', 0)
        f_sitting = chair_data.get('F_Sitting_height', 0)
        g_shoulder = chair_data.get('G_Shoulder_breadth', 0)
        body_type = chair_data.get('body_type', 'rect')[:4]
        
        print(f"{image_name:^15} {height:^7.0f} {a_buttock:^7.0f} {b_popliteal:^7.0f} {c_hip:^7.0f} {f_sitting:^8.0f} {g_shoulder:^7.0f} {body_type:^8}")

def analyze_body_type_distribution(results: List[Dict[str, Any]]):
    """체형 분포 분석"""
    if not results:
        return
    
    print(f"체형 분포")
    print("-" * 25)
    
    body_type_counts = {}
    for result in results:
        body_type = result['measurements'].get('body_type', 'rectangle')
        body_type_counts[body_type] = body_type_counts.get(body_type, 0) + 1
    
    total = len(results)
    for body_type, count in sorted(body_type_counts.items()):
        percentage = (count / total) * 100
        print(f"{body_type}: {count}개 ({percentage:.1f}%)")

def run_enhanced_system():
    """향상된 측정 시스템 실행"""
    processor = EnhancedMeasurementSystem("C:/Users/grace/OneDrive/Desktop/dataset")
    
    # 모든 카테고리 처리
    results = processor.process_all_categories()
    
    if not results:
        print("측정 결과 없음")
        return
    
    # 결과 저장
    output_dir = "C:/Users/grace/OneDrive/Desktop/dataset/finish"
    json_file, csv_file = processor.save_results(results, output_dir)
    
    print(f"저장 완료: {len(results)}개 결과")
    if json_file and csv_file:
        print(f"위치: {output_dir}")
    
    # 결과 표시
    display_results_summary(results)
    display_chair_format_summary(results)
    analyze_body_type_distribution(results)
    
    print(f"완료: {len(results)}개 이미지 측정")
    
    return results

if __name__ == "__main__":
    run_enhanced_system()

