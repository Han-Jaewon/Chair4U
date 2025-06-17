"""
사피언스 깊이 모델 - 키 측정용 데이터만 추출
깊이 맵(.npy)과 깊이 시각화 이미지만 생성
"""

import subprocess
import sys
import shutil
import os
import time
import cv2
import numpy as np
from pathlib import Path

def check_folder(folder_path, folder_name):
    """폴더 존재 및 이미지 개수 확인"""
    print(f"[{folder_name}] 폴더 확인")
    
    if not folder_path.exists():
        print(f"폴더 없음: {folder_path}")
        return False
    
    images = list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.png")) + list(folder_path.glob("*.jpeg"))
    print(f"이미지: {len(images)}개")
    
    if len(images) == 0:
        print(f"이미지 없음")
        return False
    
    print(f"처리 가능: {len(images)}개 이미지")
    return True

def check_dependencies(script_path, checkpoint_path, seg_base_path):
    """필수 의존성 파일들 확인"""
    print("의존성 파일 확인")
    
    if not Path(script_path).exists():
        print(f"깊이 추정 스크립트를 찾을 수 없습니다: {script_path}")
        return False
    print(f"스크립트 파일: {script_path}")
    
    if not Path(checkpoint_path).exists():
        print(f"체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")
        return False
    print(f"체크포인트 파일: {checkpoint_path}")
    
    if not seg_base_path.exists():
        print(f"세그멘테이션 기본 폴더를 찾을 수 없습니다: {seg_base_path}")
        return False
    print(f"세그멘테이션 기본 폴더: {seg_base_path}")
    
    return True

def save_depth_only(original_img, depth_map, output_path, seg_dir):
    """
    키 측정용 깊이 데이터만 저장
    - .npy: 정확한 깊이 값 (키 계산용)
    - .jpg: 깊이 시각화 (원본 + 깊이맵만)
    """
    # 세그멘테이션 마스크 로드
    image_name = os.path.basename(output_path)
    mask_path = os.path.join(
        seg_dir,
        image_name.replace(".png", ".npy")
        .replace(".jpg", ".npy")
        .replace(".jpeg", ".npy"),
    )
    
    if os.path.exists(mask_path):
        mask = np.load(mask_path)
    else:
        # 마스크가 없으면 전체 영역을 전경으로 처리
        mask = np.ones((original_img.shape[0], original_img.shape[1]), dtype=bool)
    
    # 깊이 맵 저장 (키 계산용 - 정확한 수치 데이터)
    save_path = (
        output_path.replace(".png", ".npy")
        .replace(".jpg", ".npy")
        .replace(".jpeg", ".npy")
    )
    np.save(save_path, depth_map)
    
    # 깊이 시각화 생성 (원본 + 깊이만)
    depth_map_masked = depth_map.copy()
    depth_map_masked[~mask] = np.nan
    depth_foreground = depth_map_masked[mask]
    
    # 깊이 시각화 처리
    processed_depth = np.full((mask.shape[0], mask.shape[1], 3), 100, dtype=np.uint8)
    
    if len(depth_foreground) > 0:
        min_val, max_val = np.min(depth_foreground), np.max(depth_foreground)
        if max_val > min_val:  # 분모가 0이 되는 것을 방지
            depth_normalized_foreground = 1 - (
                (depth_foreground - min_val) / (max_val - min_val)
            )
            depth_normalized_foreground = (depth_normalized_foreground * 255.0).astype(np.uint8)
            
            # 키 측정에 최적화된 컬러맵 사용 (INFERNO)
            depth_colored_foreground = cv2.applyColorMap(
                depth_normalized_foreground, cv2.COLORMAP_INFERNO
            )
            depth_colored_foreground = depth_colored_foreground.reshape(-1, 3)
            processed_depth[mask] = depth_colored_foreground
    
    # 원본 + 깊이맵만 결합 (표면 법선 제외)
    vis_image = np.concatenate([original_img, processed_depth], axis=1)
    cv2.imwrite(output_path, vis_image)
    
    return save_path, output_path

def process_depth_estimation_custom(input_folder, output_folder, seg_folder, folder_name, 
                                  script, checkpoint, work_dir, batch_size=1, fp16=True):
    """
    커스텀 깊이 추정 - 키 측정용 데이터만 생성
    """
    print(f"[{folder_name}] 키 측정용 깊이 추정 실행")
    
    # 출력 폴더 생성
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # 입력 이미지 개수 확인
    input_images = list(input_folder.glob("*.jpg")) + list(input_folder.glob("*.png")) + list(input_folder.glob("*.jpeg"))
    total_images = len(input_images)
    
    # 임시 출력 폴더 생성 (원본 스크립트 결과용)
    temp_output = output_folder.parent / f"temp_{folder_name}"
    temp_output.mkdir(parents=True, exist_ok=True)
    
    # vis_depth.py 실행
    cmd = [
        sys.executable,
        script,
        checkpoint,
        "--input", str(input_folder),
        "--output_root", str(temp_output),
        "--device", "cuda:0",
        "--batch_size", str(batch_size),
        "--shape", "1024", "768"
    ]
    
    if seg_folder and seg_folder.exists():
        cmd.extend(["--seg_dir", str(seg_folder)])
        print(f"세그멘테이션: {seg_folder}")
    
    if fp16:
        cmd.append("--fp16")
    
    print(f"입력: {input_folder}")
    print(f"출력: {output_folder}")
    print(f"총 이미지: {total_images}개")
    print(f"목적: 키 측정용 깊이 데이터만 추출")
    print(f"예상 소요시간: {total_images * 8}초")
    
    try:
        print(f"깊이 추정")
        start_time = time.time()
        
        # 원본 스크립트 실행
        process = subprocess.Popen(
            cmd, 
            cwd=work_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # 실시간 로그
        for line in iter(process.stdout.readline, ''):
            line = line.strip()
            if line and ("FPS:" in line or "Total inference time:" in line):
                print(f" {line}")
        
        return_code = process.wait(timeout=7200)
        
        if return_code == 0:
            print(f"원본 처리 완료")
            
            # 키 측정용 데이터만 추출
            print(f"키 측정용 데이터 추출 중...")
            processed_count = 0
            
            for img_file in input_images:
                try:
                    # 원본 이미지 로드
                    orig_img = cv2.imread(str(img_file))
                    
                    # 임시 결과에서 깊이 맵 로드
                    base_name = img_file.stem
                    temp_npy = temp_output / f"{base_name}.npy"
                    
                    if temp_npy.exists():
                        depth_map = np.load(temp_npy)
                        
                        # 키 측정용 데이터 저장
                        final_output = output_folder / f"{base_name}.jpg"
                        save_depth_only(orig_img, depth_map, str(final_output), str(seg_folder))
                        processed_count += 1
                
                except Exception as e:
                    print(f"{img_file.name} 처리 실패: {e}")
            
            # 임시 폴더 정리
            shutil.rmtree(temp_output, ignore_errors=True)
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            print(f"처리된 이미지: {processed_count}/{total_images}개")
            print(f"총 소요시간: {elapsed_time:.1f}초")
            
            return True
        else:
            print(f"깊이 추정 실패 (종료 코드: {return_code})")
            return False
            
    except Exception as e:
        print(f"처리 중 오류: {e}")
        return False

def check_depth_results(output_folder, folder_name):
    """키 측정용 결과 확인"""
    print(f"[{folder_name}] 키 측정용 결과 확인")
    
    npy_files = list(output_folder.glob("*.npy"))
    image_files = list(output_folder.glob("*.jpg"))
    
    print(f"깊이 데이터(.npy): {len(npy_files)}개")
    print(f"깊이 시각화(.jpg): {len(image_files)}개")
    
    if len(npy_files) > 0:
        print(f"키 측정용 데이터 생성 성공!")
        
        # 파일 크기 확인
        total_size = sum(f.stat().st_size for f in npy_files) / (1024 * 1024)
        print(f"깊이 데이터 총 크기: {total_size:.1f}MB")
        
        return True
    else:
        print(f"키 측정용 데이터 없음")
        return False

def get_segmentation_folder(base_path, folder_name):
    """세그멘테이션 폴더 경로 찾기"""
    seg_path = Path(r"C:\Users\grace\OneDrive\Desktop\dataset\seg") / folder_name
    
    if not seg_path.exists():
        print(f"세그멘테이션 폴더 없음: {seg_path}")
        return None
    
    npy_files = list(seg_path.glob("*.npy"))
    
    if len(npy_files) > 0:
        print(f"세그멘테이션 데이터: {len(npy_files)}개")
        return seg_path
    else:
        print(f"세그멘테이션 데이터(.npy) 없음")
        return None

def main():
    print("=" * 80)
    
    # 경로 설정
    base_input = Path(r"C:\Users\grace\OneDrive\Desktop\dataset\padding1536")
    base_output = Path(r"C:\Users\grace\OneDrive\Desktop\dataset\sapiensdepth")
    seg_base = Path(r"C:\Users\grace\OneDrive\Desktop\dataset\seg")
    
    # vis_depth.py 설정
    script = r"C:\Users\grace\OneDrive\Desktop\sapiens\lite\demo\vis_depth.py"
    checkpoint = r"C:\Users\grace\OneDrive\Desktop\sapiens\torchscript\depth\checkpoints\sapiens_0.3b\sapiens_0.3b_render_people_epoch_100_torchscript.pt2"
    work_dir = r"C:\Users\grace\OneDrive\Desktop\sapiens"
    
    # 매개변수 설정
    BATCH_SIZE = 1
    USE_FP16 = True
    
    # 의존성 확인
    if not check_dependencies(script, checkpoint, seg_base):
        return
    
    # 처리할 폴더들
    folders = ["women", "men", "validation"]
    
    results = {}
    total_start_time = time.time()
    
    # 각 폴더 처리
    for folder_name in folders:
        print(f"\n{'='*80}")
        print(f"[{folder_name}] 키 측정용 깊이 추정 시작")
        
        input_folder = base_input / folder_name
        output_folder = base_output / folder_name
        seg_folder = get_segmentation_folder(base_input, folder_name)
        
        # 폴더 확인
        if not check_folder(input_folder, folder_name):
            results[folder_name] = "폴더 없음"
            continue
        
        # 키 측정용 깊이 추정 실행
        process_success = process_depth_estimation_custom(
            input_folder, output_folder, seg_folder, folder_name, 
            script, checkpoint, work_dir, 
            batch_size=BATCH_SIZE, fp16=USE_FP16
        )
        
        if not process_success:
            results[folder_name] = "처리 실패"
            continue
        
        # 결과 확인
        result_success = check_depth_results(output_folder, folder_name)
        results[folder_name] = "성공" if result_success else "결과 불완전"
        
        print(f"   🏁 [{folder_name}] 완료: {results[folder_name]}")
    
    # 최종 요약
    total_time = time.time() - total_start_time
    print(f"\n{'='*80}")
    print(f"총 소요시간: {total_time:.1f}초 ({total_time/60:.1f}분)")
    print(f"결과 위치: {base_output}")
    
    print(f"최종 결과 요약:")
    total_images = 0
    for folder_name, status in results.items():
        output_folder = base_output / folder_name
        
        if output_folder.exists():
            npy_count = len(list(output_folder.glob("*.npy")))
            img_count = len(list(output_folder.glob("*.jpg")))
            total_images += npy_count
            print(f"  📏 {folder_name}: {status} - 깊이데이터 {npy_count}개, 시각화 {img_count}개")
        else:
            print(f"  ❌ {folder_name}: {status}")
    
    success_count = sum(1 for status in results.values() if status == "성공")
    print(f"성공: {success_count}/{len(folders)}개 폴더")
    print(f"총 키 측정용 데이터: {total_images}개")

if __name__ == "__main__":
    main()