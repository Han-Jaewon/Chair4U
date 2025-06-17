import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
import cv2
from PIL import Image
import depth_pro
from scipy.ndimage import gaussian_filter, median_filter
from scipy.interpolate import griddata
import json
import logging
import time
import gc
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Callable, cast
from tqdm import tqdm

# PyTorch 임포트
try:
    import torch
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError as e:
    raise ImportError(f"PyTorch 설치 필요: {e}")

# ===== 설정 =====
CONFIG = {
    "base_dir": Path(r"C:\Users\grace\OneDrive\Desktop\dataset"),
    # 기본 입력 폴더명
    "input_folder_base": "padding1536",             
    # 기본 출력 폴더명
    "output_folder_base": "depth",           
    "subdirs": ["women", "men", "validation"],              
    # 하위 폴더
    "depthpro_base_dir": Path(r"C:\Users\grace\OneDrive\Desktop\depth_pro"),
    "checkpoint_path": Path(r"C:\Users\grace\OneDrive\Desktop\depth_pro\checkpoints\depth_pro.pt"),
    "force_cuda": True,
    "enable_memory_optimization": True,
    "gc_frequency": 5,
    "save_visualization": True,
    "save_metadata": True,
}

# 지원되는 이미지 형식
SUPPORTED_FORMATS = {'.jpg', '.png'}

class DepthProOfficialVisualizer:
    def __init__(self, device="cuda"):
        # 디바이스 설정
        if CONFIG.get("force_cuda", False) and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # DepthPro 경로 설정
        self._setup_depthpro_paths()
        
        # 메모리 최적화
        if CONFIG["enable_memory_optimization"]:
            self._optimize_gpu_memory()
        
        # 모델 관련 변수 - 타입 힌트 추가
        self.model: Optional[torch.nn.Module] = None
        self.transform: Optional[Union[transforms.Compose, Callable[[Any], torch.Tensor]]] = None
        self.model_loaded: bool = False
        self.processed_count: int = 0
        
        self.official_colormaps = {
            'depthpro_default': 'magma',
            'depthpro_alternative': 'plasma',
            'scientific_standard': 'viridis',
            'boundary_enhanced': 'inferno',
            'publication_safe': 'cividis'
        }
        
        # Apple 공식 방법으로 모델 로드
        self._load_model_official()
    
    def _setup_depthpro_paths(self) -> None:
        """DepthPro 경로 설정 및 검증"""
        depthpro_base = CONFIG["depthpro_base_dir"]
        checkpoint_path = CONFIG["checkpoint_path"]
        
        # DepthPro 설치 경로 확인
        if not depthpro_base.exists():
            logging.error(f"DepthPro 기본 디렉토리가 없습니다: {depthpro_base}")
            raise FileNotFoundError(f"DepthPro directory not found: {depthpro_base}")
        
        # 체크포인트 파일 확인
        if not checkpoint_path.exists():
            logging.error(f"DepthPro 체크포인트가 없습니다: {checkpoint_path}")
            raise FileNotFoundError(f"DepthPro checkpoint not found: {checkpoint_path}")
        
        # Python 경로에 DepthPro 추가
        src_path = str(depthpro_base / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
            logging.info(f"DepthPro 경로 추가: {src_path}")
        
        logging.info(f"DepthPro 기본 경로: {depthpro_base}")
        logging.info(f"DepthPro 체크포인트: {checkpoint_path}")
    
    def _optimize_gpu_memory(self) -> None:
        """GPU 메모리 최적화 설정"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            # CUDA 최적화 설정
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
    
    def _check_gpu_memory(self) -> Tuple[float, float, float]:
        """GPU 메모리 상태 확인"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
            cached = torch.cuda.memory_reserved(0) / 1024**3      # GB
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            return allocated, cached, total
        return 0.0, 0.0, 0.0
    
    def _validate_transform(self, transform: Any) -> Union[transforms.Compose, Callable[[Any], torch.Tensor]]:
        """Transform 객체 타입 검증 - 235줄 오류 해결"""
        if not isinstance(transform, (transforms.Compose, Callable)):
            raise TypeError(f"Expected transforms.Compose or callable, got {type(transform)}")
        return transform
    
    def _safe_to_device(self, tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        """안전한 디바이스 이동 - 238줄, 257줄 오류 해결"""
        if not torch.is_tensor(tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
        
        # 이미 올바른 device에 있으면 원본 반환
        if tensor.device == device:
            return tensor
        
        # CUDA 가용성 확인
        if 'cuda' in str(device) and not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        
        try:
            # 비동기 전송 최적화 (pinned memory → GPU)
            if tensor.is_pinned() and 'cuda' in str(device):
                return tensor.to(device, non_blocking=True)
            return tensor.to(device)
        except Exception as e:
            logging.error(f"Device 이동 실패: {e}")
            return tensor
    
    def _validate_focal_length(self, f_px: Any) -> Optional[torch.Tensor]:
        """f_px 파라미터 검증 및 정규화 - 257줄 오류 해결"""
        if f_px is None:
            return None
        
        # 스칼라 값을 텐서로 변환
        if isinstance(f_px, (int, float)):
            return torch.tensor(f_px, dtype=torch.float32)
        
        # 이미 텐서인 경우 타입 검증
        if torch.is_tensor(f_px):
            if f_px.dtype != torch.float32:
                return f_px.to(torch.float32)
            return f_px
        
        # 기타 타입은 None으로 처리
        logging.warning(f"Unsupported f_px type: {type(f_px)}, setting to None")
        return None
    
    def _safe_model_infer(self, image_tensor: torch.Tensor, f_px: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """안전한 모델 추론 - 259줄 오류 해결"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # 입력 검증
        if not torch.is_tensor(image_tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(image_tensor)}")
        
        # 형상 검증
        if len(image_tensor.shape) not in [3, 4]:
            raise ValueError(f"Expected 3D or 4D tensor, got shape {image_tensor.shape}")
        
        # 모델 평가 모드 확인
        if self.model.training:
            self.model.eval()
        
        # 안전한 추론 실행
        try:
            with torch.no_grad():
                if f_px is not None:
                    prediction = self.model.infer(image_tensor, f_px=f_px)
                else:
                    prediction = self.model.infer(image_tensor)
                return prediction
        except Exception as e:
            raise RuntimeError(f"Model inference failed: {str(e)}")
    
    def _safe_progress_update(self, iterator: Any, metrics: Dict[str, Any], 
                             update_freq: int = 1) -> None:
        """안전한 진행률 업데이트 - 405줄 오류 해결"""
        try:
            # tqdm 객체인지 확인
            if hasattr(iterator, 'set_postfix') and callable(getattr(iterator, 'set_postfix')):
                # 업데이트 빈도 제어로 성능 최적화
                if hasattr(iterator, 'n') and iterator.n % update_freq == 0:
                    iterator.set_postfix(metrics, refresh=True)
                else:
                    iterator.set_postfix(metrics, refresh=False)
        except (AttributeError, TypeError) as e:
            logging.debug(f"Progress update failed (non-critical): {e}")
        except Exception as e:
            logging.warning(f"Unexpected progress update error: {e}")
    
    def _load_model_official(self) -> None:
        """Apple 공식 방법으로 DepthPro 모델 로드"""
        try:
            logging.info("DepthPro 모델 로드 시작 (Apple 공식 방법)...")
            start_time = time.time()
            
            # Apple 공식 방법: create_model_and_transforms()
            model, transform = depth_pro.create_model_and_transforms()
            
            # 타입 검증
            self.transform = self._validate_transform(transform)
            self.model = model.to(self.device)
            
            # 평가 모드 설정 (Apple 공식 권장)
            self.model.eval()
            
            # 메모리 최적화를 위한 gradient 비활성화
            for param in self.model.parameters():
                param.requires_grad = False
            
            self.model_loaded = True
            
            load_time = time.time() - start_time
            logging.info(f"모델 로드 완료 (Apple 공식 방법, {load_time:.2f}초)")
            
            # 메모리 상태 확인
            allocated, cached, total = self._check_gpu_memory()
            logging.info(f"GPU 메모리 - 할당: {allocated:.2f}GB / 캐시: {cached:.2f}GB / 전체: {total:.2f}GB")
            
            # 모델 정보 출력
            total_params = sum(p.numel() for p in self.model.parameters())
            logging.info(f"모델 파라미터 수: {total_params:,}")
            
        except Exception as e:
            logging.error(f"모델 로드 실패: {str(e)}")
            self._provide_troubleshooting_info()
            raise
    
    def _provide_troubleshooting_info(self) -> None:
        """문제 해결 정보 제공"""
        logging.error("DepthPro 모델 로드 실패 - 해결 방법:")
        logging.error("1. 가상환경 활성화: conda activate depth-pro")
        logging.error("2. DepthPro 설치: cd depth-pro && pip install -e .")
        logging.error(f"3. 체크포인트 확인: {CONFIG['checkpoint_path']}")
        logging.error(f"4. 공식 코드 확인: {CONFIG['depthpro_base_dir']}/src/depth_pro")
        logging.error("5. 메모리 확인: GPU 메모리가 충분한지 확인")
    
    def _collect_image_files(self, input_dir: Path) -> List[Path]:
        """이미지 파일 수집"""
        image_files = []
        for path in input_dir.iterdir():
            if path.is_file() and path.suffix.lower() in SUPPORTED_FORMATS:
                image_files.append(path)
        return sorted(image_files)  # 일관된 순서 보장
    
    def _validate_image_file(self, image_path: Path) -> bool:
        """이미지 파일 유효성 검사"""
        try:
            with Image.open(image_path) as img:
                # 기본적인 이미지 정보 확인
                _ = img.format, img.size, img.mode
            return True
        except Exception as e:
            logging.warning(f"이미지 파일 검증 실패: {image_path.name} - {e}")
            return False
    
    def process_image(self, image_path: Union[str, Path], debug: bool = False) -> Dict[str, Any]:
        """Apple 공식 방법으로 이미지 처리 - 모든 오류 해결"""
        if not self.model_loaded:
            return {
                'success': False,
                'error': 'Model not loaded',
                'image_path': str(image_path)
            }
        
        start_time = time.time()
        image_path_str = str(image_path)
        
        try:
            # 경로 유효성 검사
            if not Path(image_path_str).exists():
                return {
                    'success': False,
                    'error': f'Image file does not exist: {image_path_str}',
                    'image_path': image_path_str
                }
            
            if debug:
                logging.info(f"처리 시작: {image_path_str}")
            
            # Apple 공식 방법 1: depth_pro.load_rgb() 사용
            try:
                image, _, f_px = depth_pro.load_rgb(image_path_str)
                
                if debug:
                    logging.info(f"depth_pro.load_rgb 성공")
                    logging.info(f"이미지 타입: {type(image)}")
                    logging.info(f"초점거리 f_px: {f_px}")
                    
            except Exception as e:
                return {
                    'success': False,
                    'error': f'depth_pro.load_rgb failed: {str(e)}',
                    'image_path': image_path_str
                }
            
            # Apple 공식 방법 2: 공식 transform 적용 - 235줄 오류 해결
            try:
                if self.transform is None:
                    raise RuntimeError("Transform not initialized")
                
                # 타입 안전한 transform 호출 - 330줄 빨간줄 해결
                # mypy와 IDE를 위한 타입 캐스팅
                transform_func = cast(Callable[[Any], torch.Tensor], self.transform)
                image_tensor: torch.Tensor = transform_func(image)
                
                # 텐서 타입 검증
                if not torch.is_tensor(image_tensor):
                    raise TypeError(f"Transform must return torch.Tensor, got {type(image_tensor)}")
                
                # 안전한 디바이스 이동 - 238줄 오류 해결
                image_tensor = self._safe_to_device(image_tensor, self.device)
                
                if debug:
                    logging.info(f"공식 transform 후 크기: {image_tensor.shape}")
                    logging.info(f"tensor 타입: {image_tensor.dtype}")
                    logging.info(f"tensor 디바이스: {image_tensor.device}")
                
            except Exception as e:
                return {
                    'success': False,
                    'error': f'Official transform failed: {str(e)}',
                    'image_path': image_path_str
                }
            
            # f_px 검증 및 처리 - 257줄 오류 해결
            try:
                validated_f_px = self._validate_focal_length(f_px)
                if validated_f_px is not None:
                    validated_f_px = self._safe_to_device(validated_f_px, self.device)
                
                if debug:
                    logging.info(f"f_px 검증 완료: {validated_f_px}")
                    
            except Exception as e:
                logging.warning(f"f_px 처리 실패, None으로 설정: {e}")
                validated_f_px = None
            
            # Apple 공식 방법 3: model.infer() 사용 - 259줄 오류 해결
            try:
                prediction: Dict[str, torch.Tensor] = self._safe_model_infer(image_tensor, validated_f_px)
                
                if debug:
                    logging.info(f"model.infer 성공")
                    logging.info(f"prediction 키: {list(prediction.keys()) if isinstance(prediction, dict) else type(prediction)}")
                
            except Exception as e:
                return {
                    'success': False,
                    'error': f'model.infer failed: {str(e)}',
                    'image_path': image_path_str
                }
            
            # Apple 공식 방법 4: 결과 추출
            try:
                # Apple 공식 문서에 따른 결과 추출
                depth_map = prediction["depth"]  # Depth in [m]
                focal_length_px = prediction["focallength_px"]  # Focal length in pixels
                
                # Tensor를 NumPy로 안전하게 변환
                if hasattr(depth_map, 'cpu'):
                    depth_map = depth_map.cpu()
                if hasattr(depth_map, 'detach'):
                    depth_map = depth_map.detach()
                if hasattr(depth_map, 'numpy'):
                    depth_map = depth_map.numpy()
                
                # 스칼라 값 안전한 추출
                if hasattr(focal_length_px, 'item'):
                    focal_length_px = focal_length_px.item()
                elif hasattr(focal_length_px, 'cpu'):
                    focal_length_px = focal_length_px.cpu().item()
                elif isinstance(focal_length_px, (int, float)):
                    focal_length_px = float(focal_length_px)
                else:
                    focal_length_px = 0.0
                
                # 2D 배열로 변환 (필요 시)
                while depth_map.ndim > 2:
                    depth_map = depth_map[0]
                
                if debug:
                    logging.info(f"최종 깊이맵 크기: {depth_map.shape}")
                    logging.info(f"깊이 범위: {depth_map.min():.4f} ~ {depth_map.max():.4f} m")
                    logging.info(f"초점거리: {focal_length_px:.2f} px")
                
            except Exception as e:
                return {
                    'success': False,
                    'error': f'Result extraction failed: {str(e)}',
                    'image_path': image_path_str
                }
            
            # 메모리 정리
            try:
                del image_tensor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 주기적 가비지 컬렉션
                self.processed_count += 1
                if self.processed_count % CONFIG["gc_frequency"] == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
            except Exception as e:
                logging.warning(f"Memory cleanup warning: {str(e)}")
            
            process_time = time.time() - start_time
            
            # 원본 이미지 정보 안전한 처리 (PIL Image 형태로 변환)
            try:
                if hasattr(image, 'size'):
                    original_size = image.size
                    rgb_image = image
                else:
                    # numpy array인 경우 PIL로 변환
                    if isinstance(image, np.ndarray):
                        if image.ndim == 3 and image.shape[2] == 3:
                            rgb_image = Image.fromarray((image * 255).astype(np.uint8))
                        else:
                            rgb_image = Image.fromarray(image.astype(np.uint8))
                        original_size = rgb_image.size
                    else:
                        # 기본값
                        rgb_image = Image.open(image_path_str)
                        original_size = rgb_image.size
            except Exception:
                rgb_image = Image.open(image_path_str)
                original_size = rgb_image.size
            
            return {
                'rgb_image': rgb_image,
                'depth_map': depth_map,
                'focal_length': float(focal_length_px),
                'raw_prediction': prediction,
                'original_size': original_size,
                'process_time': process_time,
                'success': True,
                'image_path': image_path_str
            }
            
        except Exception as e:
            logging.error(f"이미지 처리 실패: {image_path_str} - {e}")
            return {
                'success': False,
                'error': str(e),
                'image_path': image_path_str
            }
    
    def process_image_list(self, image_paths: List[Path], show_progress: bool = True) -> List[Dict[str, Any]]:
        """이미지 리스트 처리 (진행률 표시 포함) - 405줄 오류 해결"""
        results = []
        processing_times = []
        
        iterator = tqdm(image_paths, desc="Apple DepthPro 깊이 추정", unit="img") if show_progress else image_paths
        
        for i, image_path in enumerate(iterator):
            # 개별 이미지 처리 (Apple 공식 방법)
            result = self.process_image(image_path)
            
            # None 결과 처리
            if result is None:
                result = {
                    'success': False,
                    'error': 'Process returned None',
                    'image_path': str(image_path)
                }
            
            results.append(result)
            
            # 성공한 경우 통계 업데이트
            if result.get("success", False):
                process_time = result.get("process_time", 0)
                if process_time > 0:
                    processing_times.append(process_time)
                    
                    # 최근 10개 평균으로 ETA 계산
                    if len(processing_times) >= 10:
                        recent_avg = np.mean(processing_times[-10:])
                    else:
                        recent_avg = np.mean(processing_times)
                    
                    remaining = len(image_paths) - i - 1
                    eta_seconds = remaining * recent_avg
                    
                    # 안전한 진행률 정보 업데이트 - 405줄 오류 해결
                    if show_progress:
                        try:
                            metrics = {
                                'time': f"{process_time:.1f}s",
                                'avg': f"{recent_avg:.1f}s",
                                'ETA': f"{eta_seconds/60:.1f}m",
                                'method': 'Apple Official'
                            }
                            self._safe_progress_update(iterator, metrics, update_freq=5)
                        except Exception as e:
                            logging.debug(f"Progress update failed: {e}")
                    
                    # 느린 처리 경고
                    if process_time > 10:
                        logging.warning(f"처리 시간 길음: {image_path.name} - {process_time:.1f}초")
            
            # 메모리 상태 주기적 확인
            if i % 20 == 0 and i > 0:
                allocated, cached, total = self._check_gpu_memory()
                if allocated > total * 0.9:  # 90% 이상 사용 시 경고
                    logging.warning(f"GPU 메모리 사용량 높음: {allocated:.2f}GB / {total:.2f}GB")
        
        return results
    
    def process_single_folder(self, input_dir: Path, output_dir: Path) -> Tuple[int, int, float]:
        """단일 폴더 처리 (Apple 공식 방법)"""
        
        # 폴더 존재 확인
        if not input_dir.exists():
            logging.error(f"입력 폴더가 존재하지 않습니다: {input_dir}")
            return 0, 0, 0.0
        
        # 출력 디렉토리 생성
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 이미지 파일 수집
        image_files = self._collect_image_files(input_dir)
        
        if not image_files:
            logging.warning(f"처리할 이미지가 없습니다: {input_dir}")
            return 0, 0, 0.0
        
        logging.info(f"{len(image_files)}개 이미지 발견")
        
        # 이미지 유효성 검사
        valid_images = []
        for image_path in image_files:
            if self._validate_image_file(image_path):
                valid_images.append(image_path)
        
        invalid_count = len(image_files) - len(valid_images)
        if invalid_count > 0:
            logging.warning(f"{invalid_count}개 무효한 이미지 제외")
        
        if not valid_images:
            logging.error("유효한 이미지가 없습니다")
            return 0, 0, 0.0
        
        # 예상 시간 계산 (Apple 공식 방법은 더 빠름)
        estimated_time = len(valid_images) * 1.5 / 60
        logging.info(f"예상 처리 시간: {estimated_time:.1f}분 (Apple 공식 방법)")
        
        # 처리 시작
        start_time = time.time()
        
        # 깊이 추정 실행 (Apple 공식 방법)
        all_results = self.process_image_list(valid_images, show_progress=True)
        
        # 결과 저장
        logging.info("결과 저장 중...")
        processed_count = 0
        error_count = 0
        
        save_iterator = tqdm(all_results, desc="저장") if len(all_results) > 10 else all_results
        
        for result in save_iterator:
            try:
                if result is not None and result.get("success", False):
                    # image_path 추출 (다양한 형태 대응)
                    image_path_value = result.get('image_path')
                    if image_path_value is not None and isinstance(image_path_value, (str, Path)):
                        filename_stem = Path(image_path_value).stem
                    else:
                        filename_stem = f"image_{processed_count}"
                    
                    self.save_depth_results(result, output_dir, filename_stem)
                    processed_count += 1
                else:
                    error_count += 1
                    if result is not None:
                        error_info = result.get('error', 'unknown error')
                        image_info = result.get('image_path', 'unknown path')
                    else:
                        error_info = 'result is None'
                        image_info = 'unknown path'
                    logging.error(f"처리 실패: {image_info} - {error_info}")
            except Exception as e:
                error_count += 1
                logging.error(f"저장 실패: {str(e)}")
        
        process_time = time.time() - start_time
        
        return processed_count, error_count, process_time
    
    def process_all_folders(self) -> Tuple[int, int, float]:
        """모든 폴더 처리 (여성, 남성, 검증)"""
        
        # 전체 처리 통계
        total_processed = 0
        total_errors = 0
        total_start_time = time.time()
        
        # 각 폴더별 처리
        for subdir_idx, subdir in enumerate(CONFIG["subdirs"], 1):
            logging.info(f"\n{'='*60}")
            logging.info(f"[{subdir_idx}/{len(CONFIG['subdirs'])}] {subdir} 폴더 처리 시작")
            logging.info(f"{'='*60}")
            
            # 입출력 경로 설정
            input_dir = CONFIG["base_dir"] / CONFIG["input_folder_base"] / subdir
            output_dir = CONFIG["base_dir"] / CONFIG["output_folder_base"] / subdir
            
            # 폴더 존재 확인
            if not input_dir.exists():
                logging.warning(f"입력 폴더가 존재하지 않습니다: {input_dir}")
                continue
            
            # 폴더 처리
            try:
                processed, errors, process_time = self.process_single_folder(input_dir, output_dir)
                
                # 통계 업데이트
                total_processed += processed
                total_errors += errors
                
                # 폴더별 결과 출력
                total_images = processed + errors
                avg_time = process_time / total_images if total_images > 0 else 0
                success_rate = (processed / total_images * 100) if total_images > 0 else 0
                
                logging.info(f"\n{subdir} 폴더 처리 완료:")
                logging.info(f"  ✓ 성공: {processed}개")
                logging.info(f"  ✗ 실패: {errors}개")
                logging.info(f"  ⏱ 처리 시간: {process_time/60:.1f}분")
                logging.info(f"  📊 평균 시간: {avg_time:.2f}초/이미지")
                logging.info(f"  📈 성공률: {success_rate:.1f}%")
                
                # GPU 메모리 상태 확인
                allocated, cached, total_mem = self._check_gpu_memory()
                logging.info(f"  🖥 GPU 메모리: {allocated:.2f}GB / {total_mem:.2f}GB")
                
            except Exception as e:
                logging.error(f"{subdir} 폴더 처리 중 오류 발생: {str(e)}")
                continue
            
            # 폴더 간 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        total_time = time.time() - total_start_time
        
        return total_processed, total_errors, total_time
    
    def save_depth_results(self, results: Dict[str, Any], output_dir: Path, filename_stem: str) -> None:
        """깊이 추정 결과를 다양한 형식으로 저장"""
        if not results.get("success", False):
            return
        
        depth_map = results["depth_map"]
        
        # 1. Numpy 배열로 저장 (.npy) - 원본 수치값
        depth_npy_path = output_dir / f"{filename_stem}.npy"
        np.save(depth_npy_path, depth_map)
        
        # 2. 시각화용 이미지 저장 (선택적)
        if CONFIG.get("save_visualization", True):
            # 2-1. 흑백 깊이맵 (.png)
            depth_gray_path = output_dir / f"{filename_stem}_gray.png"
            
            # 0-255 범위로 정규화
            depth_min, depth_max = depth_map.min(), depth_map.max()
            if depth_max > depth_min:
                depth_normalized = (depth_map - depth_min) / (depth_max - depth_min)
                depth_normalized = 1.0 - depth_normalized # 역전 방지용
            else:
                depth_normalized = np.zeros_like(depth_map)
            
            depth_gray = (depth_normalized * 255).astype(np.uint8)
            Image.fromarray(depth_gray, mode='L').save(depth_gray_path)
            
            # 2-2. 컬러 깊이맵 (.png) - DepthPro 스타일
            try:
                import matplotlib.pyplot as plt
                import matplotlib.cm as cm
                
                # 컬러맵 옵션들
                colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'turbo']
                
                for cmap_name in colormaps:
                    try:
                        # 컬러맵 적용
                        colormap = cm.get_cmap(cmap_name)
                        colored_depth = colormap(depth_normalized)
                        
                        # RGBA를 RGB로 변환 (알파 채널 제거)
                        colored_depth_rgb = (colored_depth[:, :, :3] * 255).astype(np.uint8)
                        
                        # PIL Image로 변환 후 저장
                        color_image = Image.fromarray(colored_depth_rgb, mode='RGB')
                        color_path = output_dir / f"{filename_stem}_{cmap_name}.png"
                        color_image.save(color_path)
                        
                    except Exception as e:
                        logging.debug(f"컬러맵 {cmap_name} 저장 실패: {e}")
                        continue
                
                # 기본 컬러 버전 (magma - Apple DepthPro 추정 기본값) - 메인 출력
                try:
                    main_colormap = cm.get_cmap('magma')
                    main_colored = main_colormap(depth_normalized)
                    main_colored_rgb = (main_colored[:, :, :3] * 255).astype(np.uint8)
                    main_color_image = Image.fromarray(main_colored_rgb, mode='RGB')
                    
                    # 메인 컬러 이미지 저장
                    main_color_path = output_dir / f"{filename_stem}.png"
                    main_color_image.save(main_color_path)
                    
                except Exception as e:
                    logging.warning(f"메인 컬러맵 저장 실패: {e}")
                    # 실패 시 흑백으로 대체
                    Image.fromarray(depth_gray, mode='L').save(output_dir / f"{filename_stem}.png")
            
            except ImportError:
                logging.warning("matplotlib이 설치되지 않음 - 흑백 이미지만 저장")
                # matplotlib 없으면 흑백만 저장
                Image.fromarray(depth_gray, mode='L').save(output_dir / f"{filename_stem}.png")
            
            except Exception as e:
                logging.warning(f"컬러 시각화 실패: {e} - 흑백으로 대체")
                Image.fromarray(depth_gray, mode='L').save(output_dir / f"{filename_stem}.png")
        
        # 3. 메타데이터 JSON 저장 (선택적)
        if CONFIG.get("save_metadata", True):
            depth_min, depth_max = depth_map.min(), depth_map.max()
            metadata = {
                "filename": filename_stem,
                "original_size": results.get("original_size"),
                "depth_shape": list(depth_map.shape),
                "depth_range": {
                    "min": float(depth_min),
                    "max": float(depth_max)
                },
                "process_time": results.get("process_time", 0),
                "focal_length": results.get("focal_length", 0),
                "model_info": {
                    "name": "DepthPro",
                    "method": "Apple Official API",
                    "version": "depth_pro.create_model_and_transforms()",
                    "inference": "model.infer(image, f_px=f_px)",
                    "preprocessing": "depth_pro.load_rgb()"
                },
                "files": {
                    "depth_npy": f"{filename_stem}.npy",
                    "depth_png_color": f"{filename_stem}.png",
                    "depth_png_gray": f"{filename_stem}_gray.png" if CONFIG.get("save_visualization") else None,
                    "depth_png_viridis": f"{filename_stem}_viridis.png" if CONFIG.get("save_visualization") else None,
                    "depth_png_plasma": f"{filename_stem}_plasma.png" if CONFIG.get("save_visualization") else None,
                    "depth_png_inferno": f"{filename_stem}_inferno.png" if CONFIG.get("save_visualization") else None,
                    "depth_png_magma": f"{filename_stem}_magma.png" if CONFIG.get("save_visualization") else None,
                    "depth_png_turbo": f"{filename_stem}_turbo.png" if CONFIG.get("save_visualization") else None
                }
            }
            
            metadata_path = output_dir / f"{filename_stem}.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)


# ===== 유틸리티 함수들 =====
def setup_logging(base_dir: Path) -> None:
    """로깅 시스템 설정"""
    log_file = base_dir / "depthpro_apple_official_multi.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(log_file), encoding='utf-8')
        ],
        force=True
    )


def check_folder_permissions():
    """폴더 권한 및 존재 여부 확인"""
    logging.info("=" * 50)
    logging.info("Apple DepthPro 공식 다중 폴더 환경 확인")
    logging.info("=" * 50)
    
    base_dir = CONFIG["base_dir"]
    input_base_dir = base_dir / CONFIG["input_folder_base"]
    output_base_dir = base_dir / CONFIG["output_folder_base"]
    depthpro_base = CONFIG["depthpro_base_dir"]
    checkpoint_path = CONFIG["checkpoint_path"]
    
    logging.info(f"기본 디렉토리: {base_dir}")
    logging.info(f"입력 기본 폴더: {input_base_dir}")
    logging.info(f"출력 기본 폴더: {output_base_dir}")
    logging.info(f"처리할 하위 폴더: {CONFIG['subdirs']}")
    logging.info(f"DepthPro 기본 경로: {depthpro_base}")
    logging.info(f"체크포인트: {checkpoint_path}")
    
    all_good = True
    
    # 1. DepthPro 설치 확인
    if depthpro_base.exists():
        src_dir = depthpro_base / "src" / "depth_pro"
        egg_info = depthpro_base / "src" / "depth_pro.egg-info"
        
        if src_dir.exists():
            logging.info(f"✓ DepthPro 소스 코드 OK: {src_dir}")
        else:
            logging.error(f"✗ DepthPro 소스 코드 없음: {src_dir}")
            all_good = False
            
        if egg_info.exists():
            logging.info(f"✓ DepthPro 설치 정보 OK: {egg_info}")
        else:
            logging.warning(f"⚠ DepthPro 설치 정보 없음: {egg_info}")
    else:
        logging.error(f"✗ DepthPro 기본 경로 없음: {depthpro_base}")
        all_good = False
    
    # 2. 체크포인트 확인
    if checkpoint_path.exists():
        file_size = checkpoint_path.stat().st_size / (1024**3)  # GB
        logging.info(f"✓ 체크포인트 OK: {checkpoint_path} ({file_size:.2f}GB)")
    else:
        logging.error(f"✗ 체크포인트 없음: {checkpoint_path}")
        all_good = False
    
    # 3. 기본 디렉토리 확인
    if not base_dir.exists():
        logging.error(f"기본 디렉토리가 존재하지 않습니다: {base_dir}")
        return False
    
    # 4. 각 하위 폴더 확인
    for subdir in CONFIG["subdirs"]:
        input_dir = input_base_dir / subdir
        
        if input_dir.exists():
            try:
                # 폴더 읽기 권한 확인
                list(input_dir.iterdir())
                image_count = len([f for f in input_dir.iterdir() 
                                 if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS])
                logging.info(f"✓ {subdir} 폴더 OK: {input_dir} (이미지: {image_count}개)")
            except PermissionError:
                logging.error(f"✗ {subdir} 폴더 권한 없음: {input_dir}")
                all_good = False
            except Exception as e:
                logging.error(f"✗ {subdir} 폴더 오류: {input_dir} - {e}")
                all_good = False
        else:
            logging.warning(f"⚠ {subdir} 폴더 없음: {input_dir}")
    
    # 5. 출력 폴더 확인/생성
    try:
        output_base_dir.mkdir(parents=True, exist_ok=True)
        # 쓰기 권한 확인
        test_file = output_base_dir / "test_write.tmp"
        test_file.write_text("test")
        test_file.unlink()
        logging.info(f"✓ 출력 기본 폴더 OK: {output_base_dir}")
    except Exception as e:
        logging.error(f"✗ 출력 기본 폴더 오류: {output_base_dir} - {e}")
        all_good = False
    
    # 6. DepthPro 모듈 임포트 테스트
    try:
        import depth_pro
        logging.info("✓ DepthPro 모듈 임포트 성공")
    except ImportError as e:
        logging.error(f"✗ DepthPro 모듈 임포트 실패: {e}")
        all_good = False
    
    return all_good


# ===== 메인 함수 =====
def main():
    """메인 실행 함수 - Apple DepthPro 공식 방법 다중 폴더 처리"""
    
    # 로깅 설정
    setup_logging(CONFIG["base_dir"])
    
    logging.info("=" * 70)
    logging.info("Apple DepthPro 공식 방법 다중 폴더 시각화 시스템 시작")
    logging.info("=" * 70)
    
    # 환경 정보 출력
    logging.info(f"PyTorch 버전: {torch.__version__}")
    logging.info(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    logging.info(f"처리할 폴더: {CONFIG['subdirs']}")
    logging.info(f"입력 기본 경로: {CONFIG['base_dir']}/{CONFIG['input_folder_base']}")
    logging.info(f"출력 기본 경로: {CONFIG['base_dir']}/{CONFIG['output_folder_base']}")
    logging.info(f"DepthPro 경로: {CONFIG['depthpro_base_dir']}")
    logging.info(f"체크포인트: {CONFIG['checkpoint_path']}")
    
    # 환경 확인
    if not check_folder_permissions():
        logging.error("환경 확인 실패 - 처리를 중단합니다")
        return
    
    # Apple DepthPro 공식 시각화 모델 초기화 (전체 처리에서 한 번만)
    try:
        visualizer = DepthProOfficialVisualizer()
    except Exception as e:
        logging.error(f"DepthPro 초기화 실패: {e}")
        return
    
    if not visualizer.model_loaded:
        logging.error("DepthPro 모델 로드 실패")
        return
    
    # 전체 처리 통계
    total_start_time = time.time()
    
    # 모든 폴더 처리 (Apple 공식 방법)
    try:
        total_processed, total_errors, total_time = visualizer.process_all_folders()
        
        # 전체 결과 통계
        total_images = total_processed + total_errors
        avg_total_time = total_time / total_images if total_images > 0 else 0
        total_success_rate = (total_processed / total_images * 100) if total_images > 0 else 0
        
        logging.info(f"\n{'='*80}")
        logging.info("🎉 Apple DepthPro 공식 방법 다중 폴더 처리 완료!")
        logging.info(f"{'='*80}")
        
        logging.info(f"📊 전체 통계:")
        logging.info(f"  • 총 처리된 이미지: {total_images}개")
        logging.info(f"  • 성공: {total_processed}개")
        logging.info(f"  • 실패: {total_errors}개")
        logging.info(f"  • 전체 처리 시간: {total_time/60:.1f}분 ({total_time/3600:.2f}시간)")
        logging.info(f"  • 평균 처리 시간: {avg_total_time:.2f}초/이미지")
        logging.info(f"  • 전체 성공률: {total_success_rate:.1f}%")
        
        # GPU 메모리 상태 확인
        allocated, cached, total_mem = visualizer._check_gpu_memory()
        logging.info(f"  🖥 최종 GPU 메모리: {allocated:.2f}GB / {total_mem:.2f}GB")
        
    except Exception as e:
        logging.error(f"다중 폴더 처리 중 오류 발생: {str(e)}")
        return
    
    # 결과 요약
    if total_processed > 0:
        logging.info(f"\n📁 결과 저장 위치: {CONFIG['base_dir']}/{CONFIG['output_folder_base']}")
        
        for subdir in CONFIG["subdirs"]:
            output_dir = CONFIG["base_dir"] / CONFIG["output_folder_base"] / subdir
            if output_dir.exists():
                image_count = len([f for f in output_dir.iterdir() if f.suffix == '.png'])
                logging.info(f"  ├── {subdir}/ ({image_count}개)")
        
        logging.info(f"\n📄 생성된 파일 형식 (Apple 공식 방법):")
        logging.info(f"  • .npy: 원본 깊이 맵 (numpy 배열)")
        if CONFIG.get("save_visualization"):
            logging.info(f"  • .png: Apple DepthPro 공식 시각화 (magma 색상맵)")
            logging.info(f"  • _gray.png: 흑백 깊이맵")
            logging.info(f"  • _magma/plasma/viridis/inferno/turbo.png: 다양한 색상맵")
        if CONFIG.get("save_metadata"):
            logging.info(f"  • .json: 메타데이터 (Apple 공식 정보 포함)")
        
        # 성능 분석
        if total_time > 0 and total_images > 0:
            images_per_minute = total_images / (total_time / 60)
            logging.info(f"\n⚡ Apple 공식 방법 성능 분석:")
            logging.info(f"  • 처리 속도: {images_per_minute:.1f}개/분")
            logging.info(f"  • 방법: depth_pro.load_rgb() + model.infer()")
            logging.info(f"  • 정확도: Apple 공식 API 기준")
            logging.info(f"  • 예상 1000개 처리 시간: {1000/images_per_minute:.1f}분")
        
        logging.info("\n✅ 다음 단계:")
        logging.info("  1. 결과 확인: 각 폴더의 .png 파일들을 확인해보세요")
        logging.info("  2. 품질 검증: Apple 공식 방법으로 처리된 깊이 맵 품질을 확인해보세요")
        logging.info("  3. Apple DepthPro 공식 시각화: 2x2 레이아웃으로 전문적 분석")
        logging.info("  4. 메타데이터: .json 파일에서 Apple 공식 처리 정보 확인")
    
    if total_errors > 0:
        logging.warning(f"⚠️  {total_errors}개 이미지 처리 실패")
        logging.warning("로그 파일을 확인하여 실패 원인을 분석해보세요")


# ===== 테스트 함수 =====
def test_single_image(image_path: Optional[str] = None, debug: bool = True):
    """단일 이미지 Apple 공식 방법 테스트"""
    logging.info("=" * 50)
    logging.info("Apple DepthPro 공식 방법 단일 이미지 테스트")
    logging.info("=" * 50)
    
    # 설정 확인
    setup_logging(CONFIG["base_dir"])
    
    # 이미지 경로 설정 또는 자동 찾기
    if image_path is None:
        # 자동으로 테스트 이미지 찾기
        for subdir in CONFIG["subdirs"]:
            test_folder = CONFIG["base_dir"] / CONFIG["input_folder_base"] / subdir
            if test_folder.exists():
                for ext in ['.jpg', '.jpeg', '.png']:
                    image_files = list(test_folder.glob(f"*{ext}"))
                    if image_files:
                        image_path = str(image_files[0])
                        break
                if image_path:
                    break
        
        if image_path is None:
            logging.error(f"테스트할 이미지를 찾을 수 없습니다")
            return
    
    test_image = Path(image_path)
    
    # 경로 확인
    if not test_image.exists():
        logging.error(f"테스트 이미지가 존재하지 않습니다: {test_image}")
        return
    
    # Apple DepthPro 공식 시각화 모델 초기화
    try:
        visualizer = DepthProOfficialVisualizer()
    except Exception as e:
        logging.error(f"Apple DepthPro 초기화 실패: {e}")
        return
    
    if not visualizer.model_loaded:
        logging.error("모델 로드 실패")
        return
    
    # 단일 이미지 처리 (Apple 공식 방법)
    logging.info(f"테스트 이미지: {test_image}")
    result = visualizer.process_image(str(test_image), debug=debug)
    
    if result is not None and result.get("success", False):
        logging.info("✅ Apple 공식 방법 처리 성공!")
        depth_map = result.get('depth_map')
        if depth_map is not None:
            logging.info(f"  • 깊이 맵 크기: {depth_map.shape}")
            depth_min, depth_max = depth_map.min(), depth_map.max()
            logging.info(f"  • 깊이 범위: {depth_min:.4f} ~ {depth_max:.4f} m")
        
        focal_length = result.get('focal_length', 0)
        logging.info(f"  • Apple 추정 초점거리: {focal_length:.2f} px")
        
        process_time = result.get('process_time', 0)
        logging.info(f"  • 처리 시간: {process_time:.2f}초")
        
        # 테스트 결과 저장
        output_dir = CONFIG["base_dir"] / "test_output_apple_official_multi"
        try:
            output_dir.mkdir(exist_ok=True, parents=True)
            visualizer.save_depth_results(result, output_dir, "test_depth_apple")
            logging.info(f"  • 저장 위치: {output_dir}")
        except Exception as e:
            logging.error(f"결과 저장 실패: {e}")
        
    else:
        if result is not None:
            error_msg = result.get('error', 'Unknown error')
        else:
            error_msg = 'Process returned None'
        logging.error(f"❌ Apple 공식 방법 처리 실패: {error_msg}")


if __name__ == "__main__":
    # 설정 및 로깅 초기화
    setup_logging(CONFIG["base_dir"])
    
    # 실행 모드 선택
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "test":
            # 테스트 모드
            test_image_path: Optional[str] = sys.argv[2] if len(sys.argv) > 2 else None
            test_single_image(test_image_path, debug=True)
            
        elif mode == "check":
            # Apple DepthPro 환경 확인 모드
            check_folder_permissions()
            
        elif mode == "main":
            # 메인 처리 모드
            if check_folder_permissions():
                main()
            else:
                logging.error("Apple DepthPro 환경 확인 실패 - 메인 처리를 중단합니다")
        else:
            logging.error(f"알 수 없는 모드: {mode}")
            logging.info("사용법:")
            logging.info("  python script.py test [image_path]  # Apple 공식 방법 단일 이미지 테스트")
            logging.info("  python script.py check             # Apple DepthPro 환경 확인")
            logging.info("  python script.py main              # Apple 공식 방법 다중 폴더 처리")
    else:
        # 기본 실행: Apple DepthPro 환경 확인 후 메인 처리
        logging.info("Apple DepthPro 환경 확인 중...")
        if check_folder_permissions():
            logging.info("Apple DepthPro 환경 정상 - 다중 폴더 처리 시작")
            main()
        else:
            logging.error("Apple DepthPro 환경 확인 실패")
            logging.info("\n문제 해결 방법:")
            logging.info(f"1. DepthPro 설치 확인: {CONFIG['depthpro_base_dir']}")
            logging.info(f"2. 체크포인트 확인: {CONFIG['checkpoint_path']}")
            logging.info(f"3. 입력 폴더들 확인:")
            for subdir in CONFIG["subdirs"]:
                logging.info(f"   - {CONFIG['base_dir']}/{CONFIG['input_folder_base']}/{subdir}")
            logging.info("4. 가상환경 활성화: conda activate depth-pro")
            logging.info("5. 테스트 실행: python script.py test")

# 시간될때 필요시 감마 추가
# https://claude.ai/chat/7de7f942-df97-4499-b455-b2a3e10051dc 여기서 작업했음