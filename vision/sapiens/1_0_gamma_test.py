import cv2
import numpy as np
import os
from pathlib import Path
from itertools import product
from statistics import mean, stdev
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

# 1. 유니코드 경로 처리 입출력 함수
def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """유니코드 경로를 포함한 이미지 읽기"""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, flags)

def imwrite_unicode(path, img):
    """유니코드 경로를 포함한 이미지 저장"""
    ext = os.path.splitext(path)[1].lower()
    result, encimg = cv2.imencode(ext, img)
    if not result:
        raise IOError(f"이미지 인코딩 실패: {path}")
    encimg.tofile(path)

# 2. 감마 보정용 LUT 생성 및 캐시 시스템
_lut_cache: dict[float, np.ndarray] = {}

def generate_gamma_lut(gamma: float) -> np.ndarray:
    """감마값에 따른 Look-Up Table 생성"""
    inv_gamma = 1.0 / gamma
    arr = np.arange(256, dtype=np.float32) / 255.0
    lut = np.power(arr, inv_gamma) * 255.0
    return np.clip(lut, 0, 255).astype(np.uint8)

def apply_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    """LUT를 이용한 효율적인 감마 보정 적용"""
    key = round(gamma, 2)
    if key not in _lut_cache:
        _lut_cache[key] = generate_gamma_lut(gamma)
    return cv2.LUT(img, _lut_cache[key])

# 3. 이미지 특성 분석 함수
def compute_average_brightness(img: np.ndarray) -> float:
    """이미지의 평균 밝기 계산"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))

def compute_contrast(img: np.ndarray) -> float:
    """이미지의 대비 (표준편차) 계산"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(np.std(gray))

# 4. 적응적 감마 파라미터 계산
def compute_gamma_param(avg: float, contrast: float,
                        T: float, alpha: float, beta: float,
                        target: float = 0.5) -> float:
    """이미지 특성에 따른 적응적 감마 파라미터 계산"""
    μ_t = target * 255.0
    
    # 임계값 내의 밝기인 경우 보정하지 않음
    if abs(avg - μ_t) < T:
        return 1.0
    
    # 어두운 이미지: 감마값 증가로 밝게 조정
    if avg < μ_t:
        return 1.0 + (μ_t / avg - 1.0) * alpha
    # 밝은 이미지: 감마값 감소로 어둡게 조정
    else:
        return 1.0 / (1.0 + (avg / μ_t - 1.0) ** beta)

# 5. 병렬 처리용 개별 이미지 처리 함수 (최적화용)
def process_one_for_optimization(args):
    """최적화를 위한 개별 이미지 처리 - skip_prefixes 제외하고 모든 이미지 사용"""
    img_path, T, alpha, beta, skip_prefixes = args
    
    # skip_prefixes 체크 - 해당하는 것은 제외
    filename = Path(img_path).name.lower()
    if skip_prefixes and any(filename.startswith(prefix.lower()) for prefix in skip_prefixes):
        return None  # 제외 대상이면 None 반환
    
    try:
        img = imread_unicode(str(img_path))
        if img is None:
            return None
            
        μ = compute_average_brightness(img)
        σ = compute_contrast(img)
        γ = compute_gamma_param(μ, σ, T, alpha, beta)
        adj = apply_gamma(img, γ)
        μ_adj = compute_average_brightness(adj)
        return abs(μ_adj - 0.5 * 255.0)
    except Exception as e:
        print(f"처리 중 오류 발생 - {img_path}: {e}")
        return None

# 6. 단색 테스트 이미지 생성 함수
def create_solid_color_image(brightness: int, size: tuple = (200, 200)) -> np.ndarray:
    """지정된 밝기의 단색 이미지 생성"""
    height, width = size
    # BGR 형식으로 단색 이미지 생성
    img = np.full((height, width, 3), brightness, dtype=np.uint8)
    return img

def generate_gamma_test_images(
    gamma_test_dir: Path,
    brightness_values: list[int],
    T: float, alpha: float, beta: float,
    image_size: tuple = (200, 200)
):
    """색상값 기반 감마 테스트 이미지 생성 - 각 밝기에 맞는 감마 개별 계산"""
    
    # 출력 디렉토리 생성
    gamma_test_dir.mkdir(parents=True, exist_ok=True)
    
    for brightness in brightness_values:
        # 원본 단색 이미지 생성
        original_img = create_solid_color_image(brightness, image_size)
        original_name = f"test_original_{brightness}.png"
        original_path = gamma_test_dir / original_name
        imwrite_unicode(str(original_path), original_img)
        
        # 해당 밝기에 맞는 감마값 계산 (대비는 기본값 50 사용)
        optimal_gamma = compute_gamma_param(
            avg=float(brightness), 
            contrast=50.0,  # 기본 대비값
            T=T, alpha=alpha, beta=beta,
            target=0.5  # 목표: 127.5 (0.5 * 255)
        )
        
        # 계산된 감마값으로 이미지 생성
        gamma_img = apply_gamma(original_img, optimal_gamma)
        gamma_name = f"test_gamma_{optimal_gamma:.2f}_{brightness}.png"
        gamma_path = gamma_test_dir / gamma_name
        imwrite_unicode(str(gamma_path), gamma_img)

# 7. 병렬 민감도 테스트 (최적 파라미터 탐색)
def sensitivity_test_parallel(
    img_dirs: list[Path],
    Ts: list, alphas: list, betas: list,
    skip_prefixes: list[str] = None,
    exts: list[str] = None
):
    """병렬 처리를 통한 최적 파라미터 탐색"""
    if exts is None:
        exts = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff']
    
    if skip_prefixes is None:
        skip_prefixes = ['asmonaco', 'mallorca', 'villarreal', 'werder', 'montpellier', 'shinhan']

    # 이미지 목록 수집
    images = []
    for d in img_dirs:
        if d.exists():
            for p in d.rglob("*"):
                if p.suffix.lower() in exts and p.is_file():
                    images.append(p)
    
    images = list(set(images))
    print(f"발견된 총 이미지 파일: {len(images)}개")
    
    # skip_prefixes에 해당하는 이미지 수 계산
    excluded_count = 0
    for img in images:
        filename = img.name.lower()
        if skip_prefixes and any(filename.startswith(prefix.lower()) for prefix in skip_prefixes):
            excluded_count += 1
    
    print(f"제외될 이미지: {excluded_count}개")
    print(f"최적화에 사용될 이미지: {len(images) - excluded_count}개")

    results = []
    for T, alpha, beta in product(Ts, alphas, betas):
        tasks = [(p, T, alpha, beta, skip_prefixes) for p in images]
        
        with ProcessPoolExecutor() as executor:
            errs = list(tqdm(
                executor.map(process_one_for_optimization, tasks),
                total=len(tasks),
                desc=f"최적화 진행 중 - T={T}, α={alpha}, β={beta}"
            ))
        
        # None 값 제거 (제외된 이미지들)
        valid_errs = [err for err in errs if err is not None]
        
        if valid_errs:
            results.append({
                "T": T, "alpha": alpha, "beta": beta,
                "mean_err": mean(valid_errs), 
                "std_err": stdev(valid_errs) if len(valid_errs) > 1 else 0.0,
                "valid_count": len(valid_errs)
            })

    if not results:
        raise ValueError("유효한 결과가 없습니다. 이미지 경로와 설정을 확인해주세요.")

    best = min(results, key=lambda x: x["mean_err"])
    return best, results

# 8. 메인 실행부
if __name__ == "__main__":
    # 입력 디렉토리 설정
    img_dirs = [
        Path(r"C:\Users\grace\OneDrive\Desktop\dataset\one_person_pass\men"),
        Path(r"C:\Users\grace\OneDrive\Desktop\dataset\one_person_pass\women"),
        Path(r"C:\Users\grace\OneDrive\Desktop\dataset\one_person_pass\validation")
    ]
    
    # 출력 디렉토리 설정  
    gamma_test_dir = Path(r"C:\Users\grace\OneDrive\Desktop\dataset\gamma_test")
    
    # 파라미터 탐색 범위
    Ts = [5, 10, 15, 20]
    alphas = [0.4, 0.6, 0.8]
    betas = [1.0, 1.5, 2.0]
    
    # 제외할 파일명 접두사 (이 접두사를 가진 이미지는 최적화에서 제외)
    skip_prefixes = ['asmonaco', 'mallorca', 'villarreal', 'werder', 'montpellier', 'shinhan']
    
    # 테스트용 밝기값들 (0-255 범위)
    brightness_values = [60, 100, 127, 150, 200]
    
    print("=== 감마 보정 최적 파라미터 탐색 시작 ===")
    best_params, all_results = sensitivity_test_parallel(
        img_dirs, Ts, alphas, betas, skip_prefixes
    )
    
    print("\n=== 최적 파라미터 (평균 오차 기준) ===")
    print(f"T: {best_params['T']}")
    print(f"α: {best_params['alpha']}")  
    print(f"β: {best_params['beta']}")
    print(f"평균 오차: {best_params['mean_err']:.4f}")
    print(f"표준편차: {best_params['std_err']:.4f}")
    print(f"유효 이미지 수: {best_params['valid_count']}")
    
    # 색상값 기반 테스트 이미지 생성
    generate_gamma_test_images(
        gamma_test_dir,
        brightness_values,
        best_params['T'],
        best_params['alpha'], 
        best_params['beta']
    )