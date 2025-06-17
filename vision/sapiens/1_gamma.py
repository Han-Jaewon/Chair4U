# 1_0_gamma_test.py 결과를 토대로 값 기입


import cv2
import numpy as np
import os
from pathlib import Path
import warnings
import csv

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

try:
    cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
except AttributeError:
    pass

# 유니코드 경로 처리 함수
def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """유니코드 경로 이미지 읽기"""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, flags)

def imwrite_unicode(path, img):
    """유니코드 경로 이미지 쓰기"""
    ext = os.path.splitext(path)[1].lower()
    result, encimg = cv2.imencode(ext, img)
    if not result:
        raise IOError(f"이미지 인코딩 실패: {path}")
    encimg.tofile(path)

# 감마 보정 핵심 함수들
def generate_gamma_lut(gamma: float) -> np.ndarray:
    """감마값으로 256 크기 LUT 생성"""
    inv_gamma = 1.0 / gamma
    arr = np.arange(256, dtype=np.float32) / 255.0
    lut = np.power(arr, inv_gamma) * 255.0
    return np.clip(lut, 0, 255).astype(np.uint8)

def apply_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    """이미지에 LUT 기반 감마 보정 적용"""
    lut = generate_gamma_lut(gamma)
    return cv2.LUT(img, lut)

def compute_average_brightness(img: np.ndarray) -> float:
    """그레이스케일 평균 밝기 계산"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))

# 최적 파라미터 기반 감마 계산 (고정값 사용)
def compute_optimal_gamma(avg_brightness: float, target: float = 0.5) -> float:
    """
    이미 찾은 최적 파라미터 사용: T=5, alpha=0.8, beta=1.0
    (전체 데이터셋 기준: mean_err=34.38)
    """
    # 최적 파라미터 (그리드 탐색 결과)
    T = 5.0          # 임계값
    alpha = 0.8      # 어두운 이미지 보정 강도
    beta = 1.0       # 밝은 이미지 보정 지수
    
    target_brightness = target * 255.0
    brightness_diff = abs(avg_brightness - target_brightness)
    
    # 목표 밝기 근처에서는 보정하지 않음
    if brightness_diff < T:
        return 1.0
    
    if avg_brightness < target_brightness:
        # 어두운 이미지: gamma > 1.0으로 밝게 만들기
        ratio = target_brightness / avg_brightness
        gamma = 1.0 + (ratio - 1.0) * alpha
    else:
        # 밝은 이미지: gamma < 1.0으로 어둡게 만들기
        ratio = avg_brightness / target_brightness
        gamma = 1.0 / (1.0 + (ratio - 1.0) ** beta)
    
    return float(np.clip(gamma, 0.3, 3.0))

# 카테고리별 감마 보정 이미지 생성 (직접 처리)
def process_images_with_fixed_params(img_dirs: list, output_base_dir: Path):
    """
    고정된 최적 파라미터로 바로 이미지 처리 -> 1_0_gamma_test.py 결과를 토대로 값 기입
    """
    # 백그라운드 제거된 이미지 접두사들 (감마 보정 제외)
    skip_prefixes = ['asmonaco', 'malloca', 'Villarreal', 'werder', 'montpellier', 'shinhan']
    
    total_processed = 0
    total_skipped = 0
    category_results = {}
    all_gamma_records = []
    
    # 지원하는 이미지 확장자 패턴
    image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp', '*.tiff',
                     '*.JPG', '*.JPEG', '*.PNG', '*.WEBP', '*.BMP', '*.TIFF']
    
    for img_dir in img_dirs:
        # 카테고리명을 폴더 경로로 직접 결정
        if "women" in str(img_dir):
            category_name = "women"
        elif "men" in str(img_dir):
            category_name = "men"
        elif "validation" in str(img_dir):
            category_name = "validation"
        else:
            category_name = img_dir.name  # 원본 이름 사용
        
        # 출력 디렉토리 생성
        output_dir = output_base_dir / category_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 이미지 수집 (중복 제거)
        images_set = set()
        for pattern in image_patterns:
            for img_path in img_dir.glob(pattern):
                images_set.add(img_path)
        
        images = list(images_set)
        processed_count = 0
        skipped_count = 0
        category_errors = []
        category_gamma_records = []
        
        print(f"{category_name} 처리 중... ({len(images)}장)")
        
        for i, img_path in enumerate(images):
            try:
                # 진행률 표시
                if (i + 1) % 50 == 0 or i == 0:
                    print(f"   진행: {i + 1}/{len(images)}장 처리 중...")
                
                # 백그라운드 제거된 이미지인지 확인
                filename = img_path.name
                is_background_removed = any(filename.startswith(prefix) for prefix in skip_prefixes)
                
                img = imread_unicode(str(img_path))
                if img is None:
                    continue
                
                if is_background_removed:
                    # 백그라운드 제거된 이미지는 감마 보정 없이 원본 그대로 복사
                    output_path = output_dir / filename
                    imwrite_unicode(str(output_path), img)
                    skipped_count += 1
                    
                    # 기록에는 감마=1.0으로 표시
                    avg_brightness = compute_average_brightness(img)
                    gamma_record = {
                        "카테고리": category_name,
                        "파일명": filename,
                        "원본_밝기": round(avg_brightness, 2),
                        "적용_감마": 1.0,  # 감마 보정 안함
                        "보정후_밝기": round(avg_brightness, 2),  # 원본과 동일
                        "목표대비_오차": round(abs(avg_brightness - 127.5), 2),
                        "비고": "백그라운드_제거_이미지"
                    }
                    
                    print(f"   ⏭️ 스킵: {filename} (백그라운드 제거된 이미지)")
                    
                else:
                    # 일반 이미지는 감마 보정 적용
                    avg_brightness = compute_average_brightness(img)
                    gamma = compute_optimal_gamma(avg_brightness, target=0.5)
                    adjusted_img = apply_gamma(img, gamma)
                    
                    # 보정 후 밝기 측정
                    adjusted_brightness = compute_average_brightness(adjusted_img)
                    error = abs(adjusted_brightness - 127.5)
                    category_errors.append(error)
                    
                    # 감마 보정된 이미지 저장
                    output_path = output_dir / filename
                    imwrite_unicode(str(output_path), adjusted_img)
                    processed_count += 1
                    
                    # 감마 기록 저장
                    gamma_record = {
                        "카테고리": category_name,
                        "파일명": filename,
                        "원본_밝기": round(avg_brightness, 2),
                        "적용_감마": round(gamma, 3),
                        "보정후_밝기": round(adjusted_brightness, 2),
                        "목표대비_오차": round(error, 2)
                    }
                
                category_gamma_records.append(gamma_record)
                all_gamma_records.append(gamma_record)
                
            except Exception as e:
                print(f"⚠️ 처리 실패: {img_path.name} - {e}")
                continue
        
        # 카테고리별 CSV 저장
        if category_gamma_records:
            csv_path = output_dir / f"{category_name}_감마보정_기록.csv"
            with open(csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
                fieldnames = ["파일명", "원본_밝기", "적용_감마", "보정후_밝기", "목표대비_오차", "비고"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for record in category_gamma_records:
                    writer.writerow({k: v for k, v in record.items() if k != "카테고리"})
        
        # 카테고리별 결과 저장
        avg_error = sum(category_errors) / len(category_errors) if category_errors else 0
        category_results[category_name] = {
            "processed": processed_count,
            "skipped": skipped_count,
            "total": len(images),
            "avg_error": avg_error
        }
        
        total_processed += processed_count
        total_skipped += skipped_count
        print(f"✅ {category_name}: 감마보정 {processed_count}장, 스킵 {skipped_count}장, 전체 {len(images)}장 (평균 오차: {avg_error:.1f})")
    
    # 전체 통합 CSV 저장
    if all_gamma_records:
        overall_csv_path = output_base_dir / "전체_감마보정_기록.csv"
        with open(overall_csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
            fieldnames = ["카테고리", "파일명", "원본_밝기", "적용_감마", "보정후_밝기", "목표대비_오차", "비고"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_gamma_records)
        print(f"\n📊 전체 감마 기록 저장: {overall_csv_path}")
    
    return category_results, total_processed, total_skipped

# 메인 실행부
if __name__ == "__main__":
    print("=== 고정 파라미터 감마 보정 시작 ===")
    print("📊 사용 파라미터: T=5, α=0.8, β=1.0 (전체 데이터셋 최적값)")
    
    # 입력 디렉토리 설정 (여성/남성/검증 폴더 매핑)
    img_dirs = [
        Path(r"C:\Users\grace\OneDrive\Desktop\dataset\one_person_pass\women"),
        Path(r"C:\Users\grace\OneDrive\Desktop\dataset\one_person_pass\men"),
        Path(r"C:\Users\grace\OneDrive\Desktop\dataset\one_person_pass\validation"),
    ]
    
    # 출력 디렉토리 설정
    output_base_dir = Path(r"C:\Users\grace\OneDrive\Desktop\dataset\gamma")
    
    # 바로 이미지 처리 (그리드 탐색 없이)
    category_results, total_processed, total_skipped = process_images_with_fixed_params(img_dirs, output_base_dir)
    
    # 최종 결과 요약
    print(f"\n=== 처리 완료 요약 ===")
    print(f"감마 보정된 이미지: {total_processed}장")
    print(f"스킵된 이미지: {total_skipped}장 (백그라운드 제거)")
    print(f"총 처리된 이미지: {total_processed + total_skipped}장")
    
    for category, result in category_results.items():
        print(f"   - {category}: 감마보정 {result['processed']}장, 스킵 {result['skipped']}장, 전체 {result['total']}장")
        if result['processed'] > 0:
            print(f"     └── 평균 오차: {result['avg_error']:.1f}")
    
    print(f"출력 위치: {output_base_dir}")