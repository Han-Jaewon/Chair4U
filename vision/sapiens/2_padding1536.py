from PIL import Image, ImageOps
from pathlib import Path
import os
from typing import List, Union, Tuple
from tqdm import tqdm


def pad_to_1536_preserve_orientation(image: Image.Image, fill: Union[int, Tuple[int, int, int]] = (0, 0, 0)) -> Image.Image:
    """
    입력 이미지 1536x1536 정사각형으로 변환
    EXIF orientation을 무시하여 원본 방향을 그대로 유지 -> 없는거 고려해야 함
    
    Args:
        image: PIL Image 객체
        fill: 패딩 색상 (RGB 튜플 또는 단일 값)
        
    Returns:
        1536x1536 크기의 PIL Image (원본 방향 유지)
    """
    target_size = 1536
    
    # 현재 이미지 크기
    current_width, current_height = image.size
    
    # 긴 변을 기준으로 스케일링 비율 계산
    max_dimension = max(current_width, current_height)
    scale_ratio = target_size / max_dimension
    
    # 새로운 크기 계산
    new_width = int(current_width * scale_ratio)
    new_height = int(current_height * scale_ratio)
    
    # 고품질 리샘플링으로 크기 조정
    if scale_ratio != 1.0:
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # 1536x1536 캔버스 생성
    if image.mode == 'RGBA':
        canvas = Image.new('RGBA', (target_size, target_size), fill + (255,) if len(fill) == 3 else fill)
    elif image.mode == 'RGB':
        canvas = Image.new('RGB', (target_size, target_size), fill)
    else:
        # 그레이스케일 등 다른 모드
        canvas = Image.new(image.mode, (target_size, target_size), fill[0] if isinstance(fill, tuple) else fill)
    
    # 중앙 배치 좌표 계산
    paste_x = (target_size - new_width) // 2
    paste_y = (target_size - new_height) // 2
    
    # 이미지를 캔버스 중앙에 배치
    if image.mode == 'RGBA':
        canvas.paste(image, (paste_x, paste_y), image)
    else:
        canvas.paste(image, (paste_x, paste_y))
    
    return canvas


def load_image_raw(image_path: Path) -> Image.Image:
    """
    스마트폰 사진 픽셀 데이터 로드
    PIL의 자동 EXIF 회전을 강제로 되돌림
    
    Args:
        image_path: 이미지 파일 경로
        
    Returns:
        저장된 그대로의 픽셀 데이터 (감마에서 본 것과 동일한 방향)
    """
    try:
        # PIL이 자동 회전시키기 전에 EXIF 정보 먼저 확인
        orientation = 1
        try:
            with Image.open(image_path) as temp_img:
                exif = temp_img._getexif()
                if exif and 274 in exif:
                    orientation = exif[274]
        except:
            pass
        
        # 이제 이미지 로드 (PIL이 자동 회전 적용함)
        with open(image_path, 'rb') as f:
            img = Image.open(f)
            img.load()
            
            # 이미지 모드 정리
            if img.mode == 'P':
                if 'transparency' in img.info:
                    img = img.convert('RGBA')
                else:
                    img = img.convert('RGB')
            elif img.mode in ('CMYK', 'YCbCr'):
                img = img.convert('RGB')
        
        # 핵심: PIL이 자동으로 회전시킨 것을 역회전으로 원상복구
        if orientation != 1:
            print(f"  EXIF 자동회전 역변환: {image_path.name} (orientation: {orientation})")
            
            if orientation == 2:
                # 좌우 반전
                img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            elif orientation == 3:
                # 180도 회전
                img = img.transpose(Image.Transpose.ROTATE_180)
            elif orientation == 4:
                # 상하 반전
                img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            elif orientation == 5:
                # 90도 반시계 + 좌우반전
                img = img.transpose(Image.Transpose.ROTATE_90).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            elif orientation == 6:
                # 90도 시계방향 회전을 역으로 되돌림 (90도 반시계방향)
                img = img.transpose(Image.Transpose.ROTATE_270)
            elif orientation == 7:
                # 90도 시계 + 좌우반전
                img = img.transpose(Image.Transpose.ROTATE_270).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            elif orientation == 8:
                # 90도 반시계방향 회전을 역으로 되돌림 (90도 시계방향)
                img = img.transpose(Image.Transpose.ROTATE_90)
        
        # 새로운 이미지 객체로 복사 (메타데이터 제거)
        pixel_data = list(img.getdata())
        clean_img = Image.new(img.mode, img.size)
        clean_img.putdata(pixel_data)
        
        return clean_img
        
    except Exception as e:
        raise Exception(f"이미지 로드 실패 ({image_path.name}): {str(e)}")


def process_images_to_1536(
    input_dirs: List[Path],
    output_base_dir: Path,
    dir_mapping: dict = None,
    fill_color: Tuple[int, int, int] = (0, 0, 0),
    supported_formats: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
) -> None:
    """
    지정된 디렉토리의 이미지들을 1536x1536으로 패딩 처리
    EXIF 회전을 완전히 무시하여 원본 방향을 유지
    
    Args:
        input_dirs: 입력 디렉토리 경로 리스트
        output_base_dir: 출력 베이스 디렉토리
        dir_mapping: 디렉토리 매핑 (선택사항)
        fill_color: 패딩 색상 (RGB)
        supported_formats: 지원하는 파일 확장자
    """
    
    # 출력 디렉토리 생성
    output_dir = output_base_dir / "padding1536"
    output_dir.mkdir(exist_ok=True)
    
    # 처리할 파일 목록 수집
    all_files = []
    
    for input_dir in input_dirs:
        if not input_dir.exists():
            print(f"⚠️ 디렉토리 없음: {input_dir}")
            continue
        
        # 출력 폴더명 결정
        if dir_mapping and str(input_dir) in dir_mapping:
            output_folder_name = dir_mapping[str(input_dir)]
        else:
            output_folder_name = input_dir.name
        
        current_output_dir = output_dir / output_folder_name
        current_output_dir.mkdir(exist_ok=True)
        
        # 이미지 파일 수집
        for file_path in input_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_formats:
                all_files.append((file_path, current_output_dir))
    
    if not all_files:
        print("❌ 처리할 이미지 파일이 없습니다.")
        return
    
    print(f"총 {len(all_files)}개 파일을 처리합니다.")
    print(f"출력 경로: {output_dir}")
    print("-" * 50)
    
    # 처리 통계
    success_count = 0
    error_count = 0
    
    # 진행 표시와 함께 처리
    with tqdm(all_files, desc="이미지 패딩 처리", unit="파일") as pbar:
        for file_path, output_folder in pbar:
            try:
                # 파일명 표시 업데이트
                display_name = file_path.name
                if len(display_name) > 25:
                    display_name = display_name[:22] + "..."
                
                pbar.set_postfix({
                    "파일": display_name,
                    "성공": success_count,
                    "오류": error_count
                })
                
                # 이미지 로드 (픽셀 데이터만 추출해서 회전 완전 차단)
                image = load_image_raw(file_path)
                
                # 1536x1536 패딩 처리
                padded_image = pad_to_1536_preserve_orientation(image, fill_color)
                
                # 출력 파일 경로 및 포맷 결정
                # 알파 채널 유무에 따라 최적의 포맷 선택
                if padded_image.mode == 'RGBA':
                    # 투명도가 있는 이미지 → PNG로 저장 (알파 채널 보존)
                    output_path = output_folder / f"{file_path.stem}.png"
                    padded_image.save(output_path, optimize=True, compress_level=6)
                elif file_path.suffix.lower() in ['.webp']:
                    # WebP → WebP 유지
                    output_path = output_folder / file_path.name
                    padded_image.save(output_path, quality=90, method=6)
                elif file_path.suffix.lower() in ['.png'] and padded_image.mode == 'RGB':
                    # 투명도 없는 PNG → JPEG로 변환 (용량 절약)
                    output_path = output_folder / f"{file_path.stem}.jpg"
                    padded_image.save(output_path, quality=90, optimize=True)
                else:
                    # 기타: 원본 확장자 유지
                    output_path = output_folder / file_path.name
                    if file_path.suffix.lower() in ['.jpg', '.jpeg']:
                        padded_image.save(output_path, quality=90, optimize=True)
                    else:
                        padded_image.save(output_path, optimize=True)
                
                success_count += 1
                
            except Exception as e:
                error_count += 1
                tqdm.write(f"❌ 오류: {file_path.name} - {str(e)}")
    
    # 결과 출력
    print(f"\n처리 완료")
    print(f"  성공: {success_count}개")
    print(f"  오류: {error_count}개")
    print(f"  출력: {output_dir}")


def main():
    """메인 실행 함수"""
    
    # 입력 디렉토리 설정
    input_directories = [
        Path(r"C:\Users\grace\OneDrive\Desktop\dataset\gamma\women"),
        Path(r"C:\Users\grace\OneDrive\Desktop\dataset\gamma\men"),
        Path(r"C:\Users\grace\OneDrive\Desktop\dataset\gamma\validation"),
    ]
    
    # 출력 베이스 디렉토리
    output_base = Path(r"C:\Users\grace\OneDrive\Desktop\dataset")
    
    # 디렉토리 매핑 (입력 경로 -> 출력 폴더명)
    dir_mapping = {
        str(input_directories[0]): "women",
        str(input_directories[1]): "men", 
        str(input_directories[2]): "validation",
    }
    
    # 배치 처리 실행
    process_images_to_1536(
        input_dirs=input_directories,
        output_base_dir=output_base,
        dir_mapping=dir_mapping,
        fill_color=(0, 0, 0),  # 검은색 패딩
        supported_formats=('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.heic', '.heif')
    )


if __name__ == "__main__":
    # 필수 라이브러리 확인
    try:
        from PIL import Image
        from tqdm import tqdm
        print(f"✅ PIL/Pillow 버전: {Image.__version__ if hasattr(Image, '__version__') else 'Unknown'}")
    except ImportError as e:
        print(f"필수 라이브러리 누락: {e}")
        print("설치 명령: pip install pillow tqdm")
        exit(1)
    
    main()