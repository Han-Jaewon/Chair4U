# check_db_data.py
from database import SessionLocal, Person, Chair, ChairSpecification

db = SessionLocal()

print("=== DB 데이터 확인 ===\n")

# 1. Person 확인
print("1. Person 데이터:")
persons = db.query(Person).limit(5).all()
for p in persons:
    print(f"   ID: {p.person_id}, Name: {p.image_name}, Height: {p.human_height}mm")

# 2. Chair 확인
print("\n2. Chair 데이터 (처음 10개):")
chairs = db.query(Chair).limit(10).all()
for c in chairs:
    print(f"   ID: {c.chair_id}, Brand: {c.brand_name}, Product: {c.product_name}")

# 3. 실제 존재하는 chair_id 가져오기
chair_ids = db.query(Chair.chair_id).limit(5).all()
chair_ids = [c[0] for c in chair_ids]
print(f"\n3. 테스트용 의자 ID: {chair_ids}")

# 4. ChairSpecification 확인
print("\n4. ChairSpecification 데이터 확인:")
specs = db.query(ChairSpecification).limit(5).all()
print(f"   Specification 개수: {len(specs)}")
if specs:
    print(f"   첫 번째 spec - Chair ID: {specs[0].chair_id}")

db.close()