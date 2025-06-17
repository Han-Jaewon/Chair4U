import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import shutil
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
import json
from datetime import datetime
import os
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class YOLOAnalyzer:
    # YOLO 검출
    
    def __init__(self):
        self.detection_stats = {
            'box_sizes': [],
            'box_positions': [],
            'confidence_scores': [],
            'overlap_cases': [],
            'image_sizes': [],
            'box_aspect_ratios': []
        }
        
    def analyze_detections(self, img_path, persons, save_viz=False):
        img = Image.open(img_path)
        img_width, img_height = img.size
        
        analysis = {
            'image_path': str(img_path),
            'image_size': (img_width, img_height),
            'num_persons': len(persons),
            'boxes': []
        }
        
        for i, person in enumerate(persons):
            bbox = person['bbox']
            x1, y1, x2, y2 = bbox
            
            box_width = x2 - x1
            box_height = y2 - y1
            box_area = box_width * box_height
            img_area = img_width * img_height
            area_ratio = (box_area / img_area) * 100
            
            box_center_x = (x1 + x2) / 2
            box_center_y = (y1 + y2) / 2
            center_offset_x = (box_center_x - img_width/2) / img_width
            center_offset_y = (box_center_y - img_height/2) / img_height
            
            aspect_ratio = box_height / box_width if box_width > 0 else 0
            
            box_info = {
                'index': i,
                'bbox': bbox,
                'confidence': person['conf'],
                'box_size': (box_width, box_height),
                'area_ratio': area_ratio,
                'center_offset': (center_offset_x, center_offset_y),
                'aspect_ratio': aspect_ratio,
                'position_desc': self.describe_position(x1, y1, x2, y2, img_width, img_height)
            }
            
            analysis['boxes'].append(box_info)
            
            self.detection_stats['box_sizes'].append(area_ratio)
            self.detection_stats['confidence_scores'].append(person['conf'])
            self.detection_stats['box_aspect_ratios'].append(aspect_ratio)
        
        if len(persons) > 1:
            overlap_info = self.analyze_overlaps(persons)
            analysis['overlaps'] = overlap_info
            self.detection_stats['overlap_cases'].append(overlap_info)
        
        if save_viz and len(persons) > 0:
            self.visualize_detection(img, persons, analysis, img_path)
        
        return analysis
    
    def describe_position(self, x1, y1, x2, y2, img_w, img_h):
        # 박스 위치
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        x_pos = "왼쪽" if cx < img_w/3 else ("중앙" if cx < 2*img_w/3 else "오른쪽")
        y_pos = "상단" if cy < img_h/3 else ("중앙" if cy < 2*img_h/3 else "하단")
        
        edge_touches = []
        if x1 <= 5: edge_touches.append("왼쪽 가장자리")
        if x2 >= img_w - 5: edge_touches.append("오른쪽 가장자리")
        if y1 <= 5: edge_touches.append("상단 가장자리")
        if y2 >= img_h - 5: edge_touches.append("하단 가장자리")
        
        position_desc = f"{y_pos} {x_pos}"
        if edge_touches:
            position_desc += f" ({', '.join(edge_touches)} 접촉)"
        
        return position_desc
    
    def analyze_overlaps(self, persons):
        # 다중 검출 시 중첩 분석
        overlaps = []
        
        for i in range(len(persons)):
            for j in range(i+1, len(persons)):
                iou = self.calc_iou(persons[i]['bbox'], persons[j]['bbox'])
                if iou > 0:
                    overlap_type = self.classify_overlap(persons[i]['bbox'], persons[j]['bbox'], iou)
                    overlaps.append({
                        'box_indices': (i, j),
                        'iou': iou,
                        'overlap_type': overlap_type,
                        'conf_diff': abs(persons[i]['conf'] - persons[j]['conf'])
                    })
        
        return overlaps
    
    def classify_overlap(self, box1, box2, iou):
        # 중첩 타입 분류
        if iou > 0.8:
            return "거의 동일"
        elif iou > 0.5:
            return "상당 부분 중첩"
        elif iou > 0.3:
            return "일부 중첩"
        else:
            return "약간 겹침"
    
    def calc_iou(self, box1, box2):
        # IoU 계산
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        
        xi1, yi1 = max(x1, x3), max(y1, y3)
        xi2, yi2 = min(x2, x4), min(y2, y4)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0
        
        inter = (xi2 - xi1) * (yi2 - yi1)
        union = (x2 - x1) * (y2 - y1) + (x4 - x3) * (y4 - y3) - inter
        
        return inter / union if union > 0 else 0
    
    def visualize_detection(self, img, persons, analysis, img_path):
        # 검출 결과 시각화
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(img)
        
        colors = plt.cm.rainbow(np.linspace(0, 1, len(persons)))
        
        for i, (person, color) in enumerate(zip(persons, colors)):
            x1, y1, x2, y2 = person['bbox']
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
            box_info = analysis['boxes'][i]
            label = f"Person {i+1}\nConf: {person['conf']:.2f}\nArea: {box_info['area_ratio']:.1f}%\n{box_info['position_desc']}"
            ax.text(x1, y1-5, label, color='white', fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7))
        
        ax.set_title(f"YOLO Detection: {len(persons)} person(s) detected")
        ax.axis('off')
        
        save_path = img_path.parent / f"{img_path.stem}_yolo_analysis.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def generate_summary_report(self):
        # 전체 YOLO 검출 통계 리포트
        if not self.detection_stats['box_sizes']:
            return "데이터 없음"
        
        box_sizes = self.detection_stats['box_sizes']
        
        tiny = sum(1 for s in box_sizes if s < 5)
        small = sum(1 for s in box_sizes if 5 <= s < 20)
        medium = sum(1 for s in box_sizes if 20 <= s < 50)
        large = sum(1 for s in box_sizes if s >= 50)
        
        print(f"검출 박스 크기별 분포:")
        print(f"  극소형 (<5%): {tiny}개")
        print(f"  소형 (5-20%): {small}개")
        print(f"  중형 (20-50%): {medium}개")
        print(f"  대형 (≥50%): {large}개")
        
        confidences = self.detection_stats['confidence_scores']
        print("검출 신뢰도 분석")
        print(f"평균 신뢰도: {np.mean(confidences):.3f}")
        print(f"최소 신뢰도: {np.min(confidences):.3f}")
        print(f"최대 신뢰도: {np.max(confidences):.3f}")
        
        aspect_ratios = self.detection_stats['box_aspect_ratios']
        print("박스 종횡비 분석 (높이/너비)")
        print(f"평균 종횡비: {np.mean(aspect_ratios):.2f}")
        print(f"표준 인체 비율(2.0-3.0) 범위 내: {sum(1 for r in aspect_ratios if 2.0 <= r <= 3.0)}개")
        
        if self.detection_stats['overlap_cases']:
            print("다중 검출 중첩 패턴")
            total_overlaps = len(self.detection_stats['overlap_cases'])
            print(f"다중 검출 케이스: {total_overlaps}개")
            
            overlap_types = {}
            for case in self.detection_stats['overlap_cases']:
                for overlap in case:
                    otype = overlap['overlap_type']
                    overlap_types[otype] = overlap_types.get(otype, 0) + 1
            
            for otype, count in overlap_types.items():
                print(f"  {otype}: {count}개")


class OptimizedImageFilter:
    def __init__(self):
        # GPU 최적화. 이미지 필터링 시스템 초기화
        os.environ['YOLO_VERBOSE'] = 'False'
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.yolo = YOLO('yolov8n.pt')
        self.yolo.overrides['verbose'] = False
        
        sapiens_path = r"sapiens_0.3b_goliath_best_goliath_AP_573_torchscript.pt2"
        self.sapiens = torch.jit.load(sapiens_path, map_location=self.device)
        self.sapiens.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((1024, 768)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.log_data = defaultdict(list)
        self.realtime_stats = defaultdict(lambda: {
            'total': 0, 'pass': 0, 'zero': 0, 'multi': 0, 'not_fullbody': 0
        })
        
        # YOLO 분석
        self.yolo_analyzer = YOLOAnalyzer()
        self.enable_yolo_analysis = True
        self.save_yolo_viz = False
        
        self.keypoint_names = {
            0: 'nose', 1: 'left_eye', 2: 'right_eye', 3: 'left_ear', 4: 'right_ear',
            5: 'left_shoulder', 6: 'right_shoulder', 7: 'left_elbow', 8: 'right_elbow',
            9: 'left_wrist', 10: 'right_wrist', 11: 'left_hip', 12: 'right_hip',
            13: 'left_knee', 14: 'right_knee', 15: 'left_ankle', 16: 'right_ankle',
            17: 'left_big_toe', 18: 'left_small_toe', 19: 'left_heel',
            20: 'right_big_toe', 21: 'right_small_toe', 22: 'right_heel',
            69: 'neck'
        }

        self.essential_parts = {
            'nose': {  # 코
                'indices': [0],
                'min_required': 1, 
                'threshold': 0.05, 
                'is_essential': True
            },
            'neck_or_ankles': {  # 목 또는 발목 중 하나만 있으면 됨 (OR 조건)
                'neck_indices': [69],
                'ankles_indices': [15, 16],
                'min_required': 1,
                'threshold': 0.15,
                'is_essential': True
            },
            'feet': {  # 발가락 검출
                'indices': [17, 18, 19, 20, 21, 22],
                'min_required': 1,  # 1개만 있으면 됨
                'threshold': 0.15,  
                'is_essential': True
            },
            # 참고용 부위
            'shoulders': {
                'indices': [5, 6], 
                'min_required': 1, 
                'threshold': 0.3,
                'is_essential': False
            },
            'hips': {
                'indices': [11, 12], 
                'min_required': 1, 
                'threshold': 0.3,
                'is_essential': False
            },
            'knees': {
                'indices': [13, 14], 
                'min_required': 1, 
                'threshold': 0.3,
                'is_essential': False
            }
        }

    def yolo_detect_gpu(self, img_path):
        """GPU 최적화된 YOLO 사람 검출 - 더 적극적인 NMS 및 신뢰도 조정"""
        try:
            # 신뢰도 더 많은 사람 검출 여부
            # NMS 임계값 중복 검출 여부
            results = self.yolo(str(img_path), verbose=False, conf=0.6, iou=0.25, device=self.device)
            
            persons = []
            for r in results:
                if r.boxes is not None:
                    boxes = r.boxes.cpu()
                    for box in boxes:
                        if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.6:
                            persons.append({
                                'bbox': box.xyxy[0].tolist(),
                                'conf': float(box.conf[0])
                            })
            
            # YOLO 분석 수행
            if self.enable_yolo_analysis and persons:
                analysis = self.yolo_analyzer.analyze_detections(
                    img_path, persons, 
                    save_viz=(self.save_yolo_viz and len(persons) > 1)
                )
                
                # 다중 검출 로깅
                if len(persons) > 1:
                    print(f"다중 검출: {img_path.name}")
                    for box_info in analysis['boxes']:
                        print(f"   Person {box_info['index']+1}: {box_info['position_desc']}, "
                              f"크기 {box_info['area_ratio']:.1f}%, 신뢰도 {box_info['confidence']:.2f}")
                    
                    if 'overlaps' in analysis:
                        for overlap in analysis['overlaps']:
                            print(f"   → 중첩: Box {overlap['box_indices'][0]+1} & {overlap['box_indices'][1]+1}, "
                                  f"IoU {overlap['iou']:.2f} ({overlap['overlap_type']})")
            
            if len(persons) <= 1:
                return persons
            
            return self.remove_duplicates(persons)
            
        except Exception as e:
            print(f"YOLO 검출 오류: {e}")
            return []

    def remove_duplicates(self, persons):
        """IoU 기반 중복 제거 - 더 적극적인 임계값"""
        if len(persons) <= 1:
            return persons
        
        filtered = []
        for i, p1 in enumerate(persons):
            keep = True
            for j, p2 in enumerate(persons):
                if i != j:
                    iou = self.calc_iou(p1['bbox'], p2['bbox'])
                    # IoU 적극적으로 중복 제거
                    if iou > 0.25:
                        if p1['conf'] <= p2['conf']:
                            keep = False
                            break
            if keep:
                filtered.append(p1)
        
        return filtered

    def calc_iou(self, box1, box2):
        """IoU 계산 최적화"""
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        
        xi1, yi1 = max(x1, x3), max(y1, y3)
        xi2, yi2 = min(x2, x4), min(y2, y4)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0
        
        inter = (xi2 - xi1) * (yi2 - yi1)
        union = (x2 - x1) * (y2 - y1) + (x4 - x3) * (y4 - y3) - inter
        
        return inter / union if union > 0 else 0

    def sapiens_check_gpu(self, img_path, bbox):
        """GPU 최적화된 Sapiens 전신 검사"""
        try:
            img = Image.open(str(img_path))
            x1, y1, x2, y2 = [int(x) for x in bbox]
            cropped = img.crop((x1, y1, x2, y2))
            
            tensor = self.transform(cropped).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                keypoints = self.sapiens(tensor)
            
            if isinstance(keypoints, torch.Tensor):
                keypoints = keypoints.cpu().numpy()
            
            analysis = self.analyze_keypoints_improved(keypoints)
            
            return analysis
            
        except Exception as e:
            return {
                'is_fullbody': False,
                'error': str(e),
                'detected_parts': {},
                'missing_parts': ['nose', 'neck_or_ankles', 'feet'],
                'confidence_scores': {},
                'essential_parts_passed': 0,
                'total_essential_parts': 3
            }

    def analyze_keypoints_improved(self, keypoints):
        """개선된 키포인트 분석 - 실용적 기준 적용"""
        detected_parts = {}
        missing_parts = []
        confidence_scores = {}
        essential_parts_passed = 0
        
        # 1. Nose (코) 검사
        nose_indices = [0]
        nose_threshold = 0.05
        nose_confidences = []
        
        for point_idx in nose_indices:
            if point_idx >= keypoints.shape[-1]:
                continue
            
            if keypoints.ndim == 4:
                confidence = float(keypoints[0, :, :, point_idx].max())
            elif keypoints.ndim == 2:
                confidence = float(keypoints[0, point_idx])
            else:
                confidence = float(keypoints.flat[point_idx] if point_idx < len(keypoints.flat) else 0)
            
            nose_confidences.append(confidence)
        
        nose_detected = any(conf > nose_threshold for conf in nose_confidences)
        detected_parts['nose'] = nose_detected
        confidence_scores['nose'] = {
            'max': max(nose_confidences) if nose_confidences else 0,
            'avg': sum(nose_confidences) / len(nose_confidences) if nose_confidences else 0,
            'detected_count': sum(1 for c in nose_confidences if c > nose_threshold),
            'required_count': 1,
            'is_essential': True
        }
        
        if nose_detected:
            essential_parts_passed += 1
        else:
            missing_parts.append('nose')
        
        # 2. Neck OR Ankles 검사 (둘 중 하나만 있으면 됨)
        neck_indices = [69]
        ankles_indices = [15, 16]
        neck_ankle_threshold = 0.15
        
        # 목 검사
        neck_confidences = []
        for point_idx in neck_indices:
            if point_idx >= keypoints.shape[-1]:
                continue
            
            if keypoints.ndim == 4:
                confidence = float(keypoints[0, :, :, point_idx].max())
            elif keypoints.ndim == 2:
                confidence = float(keypoints[0, point_idx])
            else:
                confidence = float(keypoints.flat[point_idx] if point_idx < len(keypoints.flat) else 0)
            
            neck_confidences.append(confidence)
        
        neck_detected = any(conf > neck_ankle_threshold for conf in neck_confidences)
        
        # 발목 검사
        ankle_confidences = []
        for point_idx in ankles_indices:
            if point_idx >= keypoints.shape[-1]:
                continue
            
            if keypoints.ndim == 4:
                confidence = float(keypoints[0, :, :, point_idx].max())
            elif keypoints.ndim == 2:
                confidence = float(keypoints[0, point_idx])
            else:
                confidence = float(keypoints.flat[point_idx] if point_idx < len(keypoints.flat) else 0)
            
            ankle_confidences.append(confidence)
        
        ankles_detected = sum(1 for conf in ankle_confidences if conf > neck_ankle_threshold) >= 1
        
        # 목 또는 발목 중 하나라도 검출되면 통과
        neck_or_ankles_detected = neck_detected or ankles_detected
        detected_parts['neck_or_ankles'] = neck_or_ankles_detected
        confidence_scores['neck_or_ankles'] = {
            'neck_max': max(neck_confidences) if neck_confidences else 0,
            'ankles_max': max(ankle_confidences) if ankle_confidences else 0,
            'neck_detected': neck_detected,
            'ankles_detected': ankles_detected,
            'is_essential': True
        }
        
        if neck_or_ankles_detected:
            essential_parts_passed += 1
        else:
            missing_parts.append('neck_or_ankles')
        
        # 3. Feet (발가락) 검사 - 1개 이상만 있으면 됨
        feet_indices = [17, 18, 19, 20, 21, 22]
        feet_threshold = 0.15
        feet_confidences = []
        
        for point_idx in feet_indices:
            if point_idx >= keypoints.shape[-1]:
                continue
            
            if keypoints.ndim == 4:
                confidence = float(keypoints[0, :, :, point_idx].max())
            elif keypoints.ndim == 2:
                confidence = float(keypoints[0, point_idx])
            else:
                confidence = float(keypoints.flat[point_idx] if point_idx < len(keypoints.flat) else 0)
            
            feet_confidences.append(confidence)
        
        feet_detected_count = sum(1 for conf in feet_confidences if conf > feet_threshold)
        feet_detected = feet_detected_count >= 1 
        
        detected_parts['feet'] = feet_detected
        confidence_scores['feet'] = {
            'max': max(feet_confidences) if feet_confidences else 0,
            'avg': sum(feet_confidences) / len(feet_confidences) if feet_confidences else 0,
            'detected_count': feet_detected_count,
            'required_count': 1,
            'is_essential': True
        }
        
        if feet_detected:
            essential_parts_passed += 1
        else:
            missing_parts.append('feet')
        
        # 4. 참고 부위들 (성공률 분석용)
        reference_parts = ['shoulders', 'hips', 'knees']
        for part_name in reference_parts:
            if part_name in self.essential_parts:
                part_info = self.essential_parts[part_name]
                indices = part_info['indices']
                min_required = part_info['min_required']
                threshold = part_info['threshold']
                
                detected_count = 0
                part_confidences = []
                
                for point_idx in indices:
                    if point_idx >= keypoints.shape[-1]:
                        continue
                    
                    if keypoints.ndim == 4:
                        confidence = float(keypoints[0, :, :, point_idx].max())
                    elif keypoints.ndim == 2:
                        confidence = float(keypoints[0, point_idx])
                    else:
                        confidence = float(keypoints.flat[point_idx] if point_idx < len(keypoints.flat) else 0)
                    
                    part_confidences.append(confidence)
                    
                    if confidence > threshold:
                        detected_count += 1
                
                is_part_detected = detected_count >= min_required
                detected_parts[part_name] = is_part_detected
                confidence_scores[part_name] = {
                    'max': max(part_confidences) if part_confidences else 0,
                    'avg': sum(part_confidences) / len(part_confidences) if part_confidences else 0,
                    'detected_count': detected_count,
                    'required_count': min_required,
                    'is_essential': False
                }
                
                if not is_part_detected:
                    missing_parts.append(part_name)
        
        # 총 3개의 필수 부위: nose, neck_or_ankles, feet
        total_essential_parts = 3
        is_fullbody = essential_parts_passed >= total_essential_parts
        
        return {
            'is_fullbody': is_fullbody,
            'detected_parts': detected_parts,
            'missing_parts': missing_parts,
            'confidence_scores': confidence_scores,
            'essential_parts_passed': essential_parts_passed,
            'total_essential_parts': total_essential_parts,
            'parts_ratio': f"{essential_parts_passed}/{total_essential_parts}"
        }

    def update_progress_simple(self, category, current, total):
        """간단한 진행률 표시"""
        if current % max(1, total // 20) == 0 or current == total:
            progress = (current / total) * 100
            print(f"   📊 {category}: {progress:.1f}% ({current}/{total})", end='\r')
            if current == total:
                print()

    def process_folder_optimized(self, input_dir, output_dir, category="unknown"):
        """GPU 최적화된 폴더 처리 - 파일 카운팅 수정"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 이미지 파일 수집 - 중복 제거
        img_extensions = ['jpg', 'jpeg', 'png', 'webp', 'bmp']
        img_files = set()  # set 사용으로 중복 제거
        
        for ext in img_extensions:
            # 대소문자 구분 없이 검색
            for file in input_path.glob(f'*.{ext}'):
                img_files.add(file)
            for file in input_path.glob(f'*.{ext.upper()}'):
                img_files.add(file)
            # 혼합 케이스도 처리 (예: .Jpg, .JPG)
            for file in input_path.glob(f'*.{ext.capitalize()}'):
                img_files.add(file)
        
        # set을 list로 변환하고 정렬
        img_files = sorted(list(img_files))
        total_files = len(img_files)
        
        # 파일 확장자별 통계 출력
        ext_counts = {}
        for file in img_files:
            ext = file.suffix.lower()
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
        
        print(f"📁 {category} 폴더: 총 {total_files}개 파일")
        print(f"   확장자별: {ext_counts}")
        
        if total_files == 0:
            print(f"이미지 파일이 없습니다: {input_path}")
            return self.realtime_stats[category]
        
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        
        start_time = time.time()
        
        for i, img_file in enumerate(img_files):
            current_stats = self.realtime_stats[category]
            current_stats['total'] += 1
            
            log_entry = {
                'filename': img_file.name,
                'category': category,
                'timestamp': datetime.now().isoformat(),
                'file_size': img_file.stat().st_size,
                'processing_time': 0,
                'status': '',
                'persons_detected': 0,
                'yolo_boxes': [],
                'sapiens_analysis': None
            }
            
            process_start = time.time()
            
            try:
                persons = self.yolo_detect_gpu(img_file)
                log_entry['persons_detected'] = len(persons)
                log_entry['yolo_boxes'] = [{'bbox': p['bbox'], 'conf': p['conf']} for p in persons]
                
                if len(persons) == 0:
                    log_entry['status'] = 'zero_person'
                    current_stats['zero'] += 1
                    
                elif len(persons) > 1:
                    log_entry['status'] = 'multiple_person'
                    current_stats['multi'] += 1
                    
                else:
                    bbox = persons[0]['bbox']
                    analysis = self.sapiens_check_gpu(img_file, bbox)
                    log_entry['sapiens_analysis'] = analysis
                    
                    if analysis['is_fullbody']:
                        shutil.copy2(img_file, output_path / img_file.name)
                        log_entry['status'] = 'success'
                        log_entry['output_path'] = str(output_path / img_file.name)
                        current_stats['pass'] += 1
                    else:
                        log_entry['status'] = 'not_fullbody'
                        current_stats['not_fullbody'] += 1
                
            except Exception as e:
                log_entry['status'] = 'error'
                log_entry['error'] = str(e)
                print(f"❌ 처리 오류: {img_file.name} - {e}")
            
            log_entry['processing_time'] = time.time() - process_start
            self.log_data[category].append(log_entry)
            
            self.update_progress_simple(category, i + 1, total_files)
            
            if i % 100 == 0 and self.device == 'cuda':
                torch.cuda.empty_cache()
        
        elapsed = time.time() - start_time
        avg_speed = total_files / (elapsed / 60) if elapsed > 0 else 0
        
        print(f"{category} 폴더 완료!")
        print(f"   처리 시간: {elapsed:.1f}초")
        print(f"   평균 속도: {avg_speed:.1f}장/분")
        print(f"   성공률: {(current_stats['pass']/total_files)*100:.1f}%")
        
        self.save_folder_log(category, output_path.parent)
        
        return current_stats

    def save_folder_log(self, category, output_base):
        """폴더별 로그 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path(output_base) / f"filter_log_{category}_{timestamp}.json"
        
        folder_data = self.log_data[category]
        stats = self.realtime_stats[category]
        
        processing_times = [entry['processing_time'] for entry in folder_data if 'processing_time' in entry]
        
        sapiens_results = [entry['sapiens_analysis'] for entry in folder_data 
                          if entry.get('sapiens_analysis')]
        
        missing_parts_count = defaultdict(int)
        confidence_stats = defaultdict(list)
        
        for result in sapiens_results:
            if result and not result['is_fullbody']:
                for part in result['missing_parts']:
                    missing_parts_count[part] += 1
                
                for part, scores in result['confidence_scores'].items():
                    if isinstance(scores, dict) and 'max' in scores:
                        confidence_stats[part].append(scores['max'])
        
        log_summary = {
            'category': category,
            'timestamp': timestamp,
            'summary_stats': dict(stats),
            'performance_metrics': {
                'total_processing_time': sum(processing_times),
                'avg_processing_time': sum(processing_times) / len(processing_times) if processing_times else 0,
                'files_per_minute': len(folder_data) / (sum(processing_times) / 60) if processing_times else 0
            },
            'sapiens_analysis': {
                'missing_parts_frequency': dict(missing_parts_count),
                'confidence_stats': {
                    part: {
                        'avg': sum(scores) / len(scores) if scores else 0,
                        'max': max(scores) if scores else 0,
                        'min': min(scores) if scores else 0
                    } for part, scores in confidence_stats.items()
                }
            },
            'improved_filtering_criteria': {
                'essential_parts': ['nose', 'neck_or_ankles', 'feet'],
                'nose_threshold': 0.05, 
                'neck_ankles_threshold': 0.15,  
                'feet_threshold': 0.15,  
                'feet_min_points': 1,
                'neck_or_ankles_logic': 'OR condition - either neck OR ankles',
                'yolo_confidence': 0.6,
                'yolo_nms_iou': 0.25
            },
            'output_location': str(Path(output_base) / category),
            'detailed_results': folder_data
        }
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_summary, f, ensure_ascii=False, indent=2)
        
        print(f"로그 저장: {log_file}")
        return log_file

    def generate_final_analysis(self):
        """최종 종합 분석 리포트 생성 - 개선된 기준 반영"""
        print("\n" + "="*100)
        print("최종 종합 분석 리포트 (개선된 기준)")
        print("="*100)
        
        all_data = []
        total_stats = {'total': 0, 'pass': 0, 'zero': 0, 'multi': 0, 'not_fullbody': 0}
        
        for category in ['men', 'women', 'validation']:
            if category in self.log_data and len(self.log_data[category]) > 0:
                folder_data = self.log_data[category]
                all_data.extend(folder_data)
            
            if category in self.realtime_stats and self.realtime_stats[category]:
                folder_stats = self.realtime_stats[category]
                for key in total_stats:
                    if key in folder_stats:
                        total_stats[key] += folder_stats[key]
        
        print("전체 성공률 분석")
        print("-" * 50)
        if total_stats['total'] > 0:
            success_rate = (total_stats['pass'] / total_stats['total']) * 100
            zero_rate = (total_stats['zero'] / total_stats['total']) * 100
            multi_rate = (total_stats['multi'] / total_stats['total']) * 100
            fullbody_fail_rate = (total_stats['not_fullbody'] / total_stats['total']) * 100
            
            print(f"최종 통과율: {success_rate:.2f}% ({total_stats['pass']:,}/{total_stats['total']:,})")
            print(f"0명 검출율: {zero_rate:.2f}% ({total_stats['zero']:,}건)")
            print(f"다수 검출율: {multi_rate:.2f}% ({total_stats['multi']:,}건)")
            print(f"전신 실패율: {fullbody_fail_rate:.2f}% ({total_stats['not_fullbody']:,}건)")
        
        print("폴더별 성능 비교")
        print("-" * 50)
        for category in ['men', 'women', 'validation']:
            if category in self.realtime_stats:
                stats = self.realtime_stats[category]
                if stats['total'] > 0:
                    success_rate = (stats['pass'] / stats['total']) * 100
                    print(f"{category.upper():>10}: {success_rate:6.2f}% ({stats['pass']:,}/{stats['total']:,})")
        
        print("Sapiens 부위별 실패 분석 (개선된 기준)")
        print("-" * 50)
        
        missing_parts_count = defaultdict(int)
        confidence_by_part = defaultdict(list)
        total_sapiens_checks = 0
        essential_failures = defaultdict(int)
        
        for entry in all_data:
            if entry.get('sapiens_analysis') and not entry['sapiens_analysis'].get('is_fullbody', True):
                total_sapiens_checks += 1
                analysis = entry['sapiens_analysis']
                
                for part in analysis.get('missing_parts', []):
                    missing_parts_count[part] += 1
                    
                    if part in ['nose', 'neck_or_ankles', 'feet']:
                        essential_failures[part] += 1
                
                for part, scores in analysis.get('confidence_scores', {}).items():
                    if isinstance(scores, dict):
                        if 'max' in scores:
                            confidence_by_part[part].append(scores['max'])
                        elif 'neck_max' in scores:  # neck_or_ankles 특수 처리
                            confidence_by_part['neck'].append(scores['neck_max'])
                            confidence_by_part['ankles'].append(scores['ankles_max'])
        
        if essential_failures:
            print("필수 부위 실패 현황 (개선된 기준):")
            for part, count in essential_failures.items():
                failure_rate = (count / total_sapiens_checks) * 100 if total_sapiens_checks > 0 else 0
                print(f"   {part:>15}: {failure_rate:6.2f}% ({count:,}/{total_sapiens_checks:,}건)")
        
        if missing_parts_count:
            print("참고 부위 실패 현황:")
            reference_parts = ['shoulders', 'hips', 'knees']
            for part in reference_parts:
                if part in missing_parts_count:
                    count = missing_parts_count[part]
                    failure_rate = (count / total_sapiens_checks) * 100 if total_sapiens_checks > 0 else 0
                    print(f"   {part:>12}: {failure_rate:6.2f}% ({count:,}/{total_sapiens_checks:,}건)")
        
        print("부위별 평균 신뢰도 점수 (개선된 임계값)")
        print("-" * 50)
        
        essential_parts_info = {
            'nose': 0.05,      # 매우 완화 (원래 0.3의 1/6)
            'neck': 0.15,      # 더 완화
            'ankles': 0.15,    # 더 완화
            'feet': 0.15       # 더 완화
        }
        
        print("필수 부위 (개선된 기준):")
        for part, threshold in essential_parts_info.items():
            if part in confidence_by_part:
                scores = confidence_by_part[part]
                avg_conf = sum(scores) / len(scores)
                min_conf = min(scores)
                max_conf = max(scores)
                
                status = "🟢" if avg_conf >= threshold else "🔴"
                print(f"   {status} {part:>8}: 평균 {avg_conf:.3f} (범위: {min_conf:.3f}~{max_conf:.3f}) [임계값: {threshold:.3f}]")
        
        reference_parts = ['shoulders', 'hips', 'knees']
        print("\n📊 참고 부위:")
        for part in reference_parts:
            if part in confidence_by_part:
                scores = confidence_by_part[part]
                avg_conf = sum(scores) / len(scores)
                min_conf = min(scores)
                max_conf = max(scores)
                current_threshold = 0.3
                
                status = "🟢" if avg_conf >= current_threshold else "🔴"
                print(f"   {status} {part:>12}: 평균 {avg_conf:.3f} (범위: {min_conf:.3f}~{max_conf:.3f}) [임계값: {current_threshold:.3f}]")
        
        # YOLO 분석 리포트 추가
        if self.enable_yolo_analysis:
            self.yolo_analyzer.generate_summary_report()
        
        print("\n" + "="*100)


def check_folder_contents(folder_path):
    """폴더 내용 상세 검사"""
    path = Path(folder_path)
    
    print(f"\n📂 폴더 검사: {path}")
    print("=" * 60)
    
    # 모든 파일 확인
    all_files = list(path.glob('*'))
    print(f"전체 파일 수: {len(all_files)}")
    
    # 확장자별 분류
    ext_groups = {}
    for file in all_files:
        if file.is_file():
            ext = file.suffix.lower() if file.suffix else 'no_extension'
            if ext not in ext_groups:
                ext_groups[ext] = []
            ext_groups[ext].append(file.name)
    
    # 확장자별 출력
    for ext, files in sorted(ext_groups.items()):
        print(f"\n{ext}: {len(files)}개")
        if len(files) <= 5:
            for f in files:
                print(f"  - {f}")
        else:
            print(f"  - {files[0]}")
            print(f"  - {files[1]}")
            print(f"  - ...")
            print(f"  - {files[-1]}")
    
    # 이미지 파일만 카운트
    image_exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}
    image_files = [f for f in all_files if f.is_file() and f.suffix.lower() in image_exts]
    print(f"\n✅ 실제 이미지 파일: {len(image_files)}개")
    
    return len(image_files)


if __name__ == "__main__":
    start_time = time.time()
    
    # 폴더 내용 먼저 확인
    base = r"C:\Users\grace\OneDrive\Desktop\dataset"
    
    print("🔍 폴더 내용 사전 검사")
    men_count = check_folder_contents(f"{base}/image/men/false_name")
    women_count = check_folder_contents(f"{base}/image/women/false_name")
    val_count = check_folder_contents(f"{base}/image/validation")
    
    print(f"\n📊 총 이미지 파일 수:")
    print(f"   Men: {men_count}개")
    print(f"   Women: {women_count}개")
    print(f"   Validation: {val_count}개")
    print(f"   총합: {men_count + women_count + val_count}개")
    
    print("\n" + "="*80)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
    
    filter_system = OptimizedImageFilter()
    
    output_base = f"{base}/one_person_pass"
    
    folders = [
        ("men", f"{base}/image/men/false_name"),
        ("women", f"{base}/image/women/false_name"),
        ("validation", f"{base}/image/validation")
    ]
    
    total_stats = {'total': 0, 'pass': 0, 'zero': 0, 'multi': 0, 'not_fullbody': 0}
    
    for category, input_path in folders:
        print(f"\n🔄 {category.upper()} 폴더 처리 시작...")
        
        folder_stats = filter_system.process_folder_optimized(
            input_path,
            f"{output_base}/{category}",
            category
        )
        
        for key in total_stats:
            total_stats[key] += folder_stats[key]
    
    filter_system.generate_final_analysis()
    
    end_time = time.time()
    total_elapsed = end_time - start_time
    
    print("="*80)
    print(f"총 성공: {total_stats['pass']:,}건")
    print(f"0명 검출: {total_stats['zero']:,}건")
    print(f"다수 검출: {total_stats['multi']:,}건")
    print(f"전신 불가: {total_stats['not_fullbody']:,}건")
    print(f"총 처리: {total_stats['total']:,}건")
    print(f"전체 성공률: {(total_stats['pass']/total_stats['total'])*100:.2f}%")
    print(f"출력 위치: {output_base}")
    print("="*80)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()