from pathlib import Path
from PIL import Image

# 처리할 베이스 폴더들
base_dirs = [
    r"C:\Users\grace\OneDrive\Desktop\dataset\padding1536"
]

# 하위 카테고리
categories = ["validation", "men", "women"]

for base in base_dirs:
    base_path = Path(base)
    for cat in categories:
        dir_path = base_path / cat
        if not dir_path.is_dir():
            print(f"폴더 없음: {dir_path}")
            continue

        print(f"\n처리 폴더: {dir_path}")

        # 1) WebP → JPG 변환 & 원본 삭제
        for webp in dir_path.glob("*.webp"):
            try:
                img = Image.open(webp).convert("RGB")
                jpg = webp.with_suffix(".jpg")
                img.save(jpg, "JPEG")
                webp.unlink()
                print(f" WebP→JPG: {webp.name} → {jpg.name}")
            except Exception as e:
                print(f" WebP 변환 실패: {webp.name} ({e})")

        # 2) .jpeg/.JPEG → .jpg 확장자 통일
        for f in dir_path.iterdir():
            suf = f.suffix.lower()
            if suf == ".jpeg":
                new_f = f.with_suffix(".jpg")
                try:
                    f.rename(new_f)
                    print(f"확장자 통일: {f.name} → {new_f.name}")
                except Exception as e:
                    print(f"확장자 통일 실패: {f.name} ({e})")