"""
세그멘테이션
"""

import subprocess
import sys
import shutil
import os
from pathlib import Path

def check_folder(folder_path, folder_name):
    """
    폴더 존재 및 이미지 개수 확인
    """
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

def process_segmentation(input_folder, output_folder, folder_name, script, checkpoint, work_dir):
    """
    세그멘테이션 스크립트 실행 및 결과 확인
    """
    print(f"[{folder_name}] 세그멘테이션 실행")
    
    # 출력 폴더 생성
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # vis_seg.py에 맞는 명령어 구성
    cmd = [
        sys.executable,
        script,
        checkpoint,
        "--input", str(input_folder),
        "--output-root", str(output_folder),
        "--device", "cuda:0",
        "--batch_size", "1",
        "--shape", "1024", "768",
        "--opacity", "0.5"   
    ]
    
    print(f"   입력: {input_folder}")
    print(f"   출력: {output_folder}")
    
    # 실행
    try:
        result = subprocess.run(cmd, cwd=work_dir, check=True, timeout=3600)  # 1시간 타임아웃
        print(f"세그멘테이션 실행 완료")
        return True
    except subprocess.CalledProcessError as e:
        print(f"세그멘테이션 실행 실패: {e}")
        return False
    except subprocess.TimeoutExpired:
        print(f"세그멘테이션 시간 초과 (1시간)")
        return False

def check_segmentation_results(output_folder, folder_name):
    """
    세그멘테이션 결과 파일 확인
    """
    print(f"\n📊 [{folder_name}] 결과 확인")
    
    # vis_seg.py가 생성하는 파일들:
    # - .png (시각화된 세그멘테이션 이미지)
    # - .npy (마스크 데이터)
    # - _seg.npy (세그멘테이션 데이터)
    
    seg_images = list(output_folder.glob("*.png")) + list(output_folder.glob("*.jpg"))
    npy_files = list(output_folder.glob("*.npy"))
    seg_npy_files = list(output_folder.glob("*_seg.npy"))
    
    print(f"   세그멘테이션 이미지: {len(seg_images)}개")
    print(f"   마스크 파일(.npy): {len(npy_files)}개")
    print(f"   세그멘테이션 파일(_seg.npy): {len(seg_npy_files)}개")
    
    if len(seg_images) > 0:
        print(f"성공! 세그멘테이션 이미지 생성")
        return True
    elif len(npy_files) > 0:
        print(f"NPY 파일만 생성 (이미지 저장 실패)")
        return False
    else:
        print(f"결과 없음")
        return False

def main():
    base_input = Path(r"C:\Users\grace\OneDrive\Desktop\dataset\padding1536")
    base_output = Path(r"C:\Users\grace\OneDrive\Desktop\dataset\seg")
    
    # 세그멘테이션 스크립트 설정
    script = r"C:\Users\grace\OneDrive\Desktop\sapiens\lite\demo\vis_seg.py"
    checkpoint = r"C:\Users\grace\OneDrive\Desktop\sapiens\sapiens_lite_host\torchscript\seg\checkpoints\sapiens_1b\sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2"
    work_dir = r"C:\Users\grace\OneDrive\Desktop\sapiens"
    
    # 스크립트 파일 존재 확인
    if not Path(script).exists():
        print(f"❌ 세그멘테이션 스크립트를 찾을 수 없음: {script}")
        return
    
    if not Path(checkpoint).exists():
        print(f"체크포인트 찾을 수 없음: {checkpoint}")
        return
    
    # 처리할 폴더들 (영어명 직접 사용)
    folders = ["women", "men", "validation"]
    
    results = {}
    
    # 각 폴더 처리
    for folder_name in folders:
        print(f"\n{'='*60}")
        print(f"[{folder_name}] 세그멘테이션 처리 시작")
        
        input_folder = base_input / folder_name
        output_folder = base_output / folder_name
        
        # 단계 1: 폴더 확인
        if not check_folder(input_folder, folder_name):
            print(f"[{folder_name}] 폴더 문제 - 건너뜀")
            results[folder_name] = "폴더 없음"
            continue
        
        # 단계 2: 세그멘테이션 실행
        process_success = process_segmentation(input_folder, output_folder, folder_name, script, checkpoint, work_dir)
        if not process_success:
            print(f"[{folder_name}] 처리 실패")
            results[folder_name] = "처리 실패"
            continue
        
        # 단계 3: 결과 확인
        result_success = check_segmentation_results(output_folder, folder_name)
        if result_success:
            results[folder_name] = "성공"
        else:
            results[folder_name] = "결과 불완전"
        
        print(f"[{folder_name}] 완료: {results[folder_name]}")
    
    # 최종 요약
    print(f"\n{'='*60}")
    print(f"결과 위치: {base_output}")
    
    print(f"최종 결과 요약:")
    for folder_name, status in results.items():
        output_folder = base_output / folder_name
        
        if output_folder.exists():
            seg_count = len(list(output_folder.glob("*.png"))) + len(list(output_folder.glob("*.jpg")))
            npy_count = len(list(output_folder.glob("*.npy")))
            print(f"{folder_name}: {status} - 이미지 {seg_count}개, NPY {npy_count}개")
        else:
            print(f"{folder_name}: {status}")
    
    # 성공한 폴더 개수
    success_count = sum(1 for status in results.values() if status == "성공")
    print(f"성공: {success_count}/{len(folders)}개 폴더")
    
    if success_count == len(folders):
        print("성공")
    else:
        print("로그 확인 필요")
        
        # 실패한 폴더들 안내
        failed_folders = [name for name, status in results.items() if status != "성공"]
        if failed_folders:
            print(f"실패한 폴더: {', '.join(failed_folders)}")

if __name__ == "__main__":
    main()