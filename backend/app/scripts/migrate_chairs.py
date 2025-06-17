import pandas as pd
import sys
from pathlib import Path
from sqlalchemy.orm import Session
from sqlalchemy import text

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.database import SessionLocal, engine
from app.models import Chair, ChairSpecification

def fix_database_constraints():
    """데이터베이스 제약조건 수정"""
    with engine.connect() as conn:
        try:
            # 제약조건 제거
            conn.execute(text("ALTER TABLE chair_specifications DROP CONSTRAINT IF EXISTS valid_seat_height"))
            conn.commit()
            print("✅ 제약조건 제거 완료")
            
            # 새 제약조건 추가
            conn.execute(text("""
                ALTER TABLE chair_specifications 
                ADD CONSTRAINT valid_seat_height CHECK (
                    seat_height_max IS NULL 
                    OR seat_height_min IS NULL 
                    OR seat_height_max >= seat_height_min
                )
            """))
            conn.commit()
            print("새 제약조건 추가 완료")
        except Exception as e:
            print(f"제약조건 수정 중 오류: {e}")

def safe_convert_value(value, target_type, default=None):
    """안전한 타입 변환 함수"""
    if pd.isna(value) or value is None or str(value).strip() == '':
        return default
    
    try:
        if target_type == 'int':
            if isinstance(value, str):
                value = value.replace(',', '').replace('원', '').strip()
            return int(float(value))
        elif target_type == 'float':
            if isinstance(value, str):
                value = value.replace(',', '').strip()
            return float(value)
        elif target_type == 'str':
            return str(value).strip()
        elif target_type == 'bool':
            if isinstance(value, str):
                return value.upper() == 'O'
            return bool(value)
    except (ValueError, TypeError, AttributeError):
        return default
    
    return default

def load_chair_data(filepath: str):
    """Excel 파일에서 의자 데이터 로드"""
    try:
        df = pd.read_excel(filepath)
        print(f"총 {len(df)}개의 의자 데이터 로드됨")
        
        # 데이터 미리보기
        print("\n 데이터 컬럼:")
        print(df.columns.tolist())
        
        return df
    except Exception as e:
        print(f"파일 로드 실패: {str(e)}")
        raise

def prepare_chair_data(df: pd.DataFrame):
    """데이터 전처리 및 분리"""
    # 불린 값 변환
    bool_mappings = {
        '헤드레스트 유무': 'has_headrest',
        '팔걸이 유무': 'has_armrest',
        '요추지지대 유무': 'has_lumbar_support',
        '높이 조절 레버 유무': 'has_height_adjustment',
        '틸팅 여부': 'has_tilting'
    }
    
    for ko_col, en_col in bool_mappings.items():
        if ko_col in df.columns:
            df[en_col] = df[ko_col].apply(lambda x: True if str(x).upper() == 'O' else False)
    
    # 기본 정보 컬럼
    basic_cols_mapping = {
        '링크': 'product_url',
        '브랜드명': 'brand_name',
        '제품명': 'product_name',
        '가격': 'price',
        '별점': 'rating',
        '리뷰 갯수': 'review_count',
        '등받이 곧/꺾': 'backrest_type'
    }
    
    # 치수 정보 컬럼
    spec_cols_mapping = {
        'h8_지면-좌석 높이_MIN': 'seat_height_min',
        'h8_지면-좌석 높이_MAX': 'seat_height_max',
        'b3_좌석 가로 길이': 'seat_width',
        't4_좌석 세로 길이 일반': 'seat_depth',
        'b4_등받이 가로 길이': 'backrest_width',
        'h7_등받이 세로 길이': 'backrest_height'
    }
    
    return df, basic_cols_mapping, spec_cols_mapping, bool_mappings

def clear_existing_data(db: Session):
    """기존 데이터 삭제"""
    try:
        spec_count = db.query(ChairSpecification).count()
        chair_count = db.query(Chair).count()
        
        if chair_count > 0:
            response = input(f"\n 기존 데이터가 있습니다:\n"
                           f"  - 의자: {chair_count}개\n"
                           f"  - 사양: {spec_count}개\n"
                           f"기존 데이터를 삭제하고 새로 입력하시겠습니까? (y/n): ")
            
            if response.lower() != 'y':
                return False
            
            print("기존 데이터 삭제 중...")
            db.query(ChairSpecification).delete()
            db.query(Chair).delete()
            db.commit()
            print("기존 데이터 삭제 완료")
        
        return True
    except Exception as e:
        print(f"데이터 삭제 중 오류: {str(e)}")
        db.rollback()
        return False

def migrate_to_database(df: pd.DataFrame, db: Session):
    """데이터베이스로 마이그레이션"""
    df, basic_cols_mapping, spec_cols_mapping, bool_mappings = prepare_chair_data(df)
    
    success_count = 0
    error_count = 0
    errors = []
    
    print(f"\n 마이그레이션 시작 ({len(df)}개 항목)...")
    
    for idx, row in df.iterrows():
        try:
            # Chair 객체 생성
            chair_data = {}
            
            # 기본 정보 매핑
            for ko_col, en_col in basic_cols_mapping.items():
                if ko_col in df.columns:
                    value = row.get(ko_col)
                    
                    if en_col == 'product_url':
                        chair_data[en_col] = safe_convert_value(value, 'str', None)
                    elif en_col == 'brand_name':
                        chair_data[en_col] = safe_convert_value(value, 'str', '알 수 없음')
                    elif en_col == 'product_name':
                        chair_data[en_col] = safe_convert_value(value, 'str', f'의자_{idx + 1}')
                    elif en_col == 'price':
                        chair_data[en_col] = safe_convert_value(value, 'int', None)
                    elif en_col == 'review_count':
                        chair_data[en_col] = safe_convert_value(value, 'int', None)
                    elif en_col == 'rating':
                        rating_value = safe_convert_value(value, 'float', None)
                        if rating_value is not None:
                            # rating 값을 0-5 범위로 정규화
                            if rating_value > 5:
                                if rating_value <= 100:
                                    rating_value = rating_value / 20.0
                                else:
                                    rating_value = 5.0
                            rating_value = max(0.0, min(rating_value, 5.0))
                            chair_data[en_col] = round(rating_value, 1)
                        else:
                            chair_data[en_col] = None
                    elif en_col == 'backrest_type':
                        value_str = safe_convert_value(value, 'str', None)
                        if value_str and value_str not in ['nan', 'None', 'null', '']:
                            chair_data[en_col] = value_str
                        else:
                            chair_data[en_col] = None
                else:
                    if en_col == 'brand_name':
                        chair_data[en_col] = '알 수 없음'
                    elif en_col == 'product_name':
                        chair_data[en_col] = f'의자_{idx + 1}'
            
            # 불린 값 추가
            for ko_col, en_col in bool_mappings.items():
                if ko_col in df.columns:
                    value = row.get(ko_col)
                    chair_data[en_col] = safe_convert_value(value, 'bool', False)
                else:
                    chair_data[en_col] = False
            
            # Chair 생성
            chair = Chair(**chair_data)
            db.add(chair)
            db.flush()
            
            # ChairSpecification 생성
            spec_data = {'chair_id': chair.chair_id}
            has_spec = False
            
            # 좌석 높이 데이터 수집
            height_min = None
            height_max = None
            
            for ko_col, en_col in spec_cols_mapping.items():
                if ko_col in df.columns:
                    value = row.get(ko_col)
                    converted_value = safe_convert_value(value, 'int', None)
                    
                    if en_col == 'seat_height_min':
                        height_min = converted_value
                    elif en_col == 'seat_height_max':
                        height_max = converted_value
                    
                    if converted_value is not None:
                        spec_data[en_col] = converted_value
                        has_spec = True
            
            # 좌석 높이 min/max 검증 및 수정
            if height_min is not None and height_max is not None:
                if height_min > height_max:
                    # 값을 서로 바꿈
                    spec_data['seat_height_min'] = height_max
                    spec_data['seat_height_max'] = height_min
                    if error_count < 5:  # 처음 5개만 경고
                        print(f"행 {idx + 1}: 좌석 높이 min({height_min}) > max({height_max}) → 값 교체")
            
            # 최소한 하나의 사양이라도 있으면 저장
            if has_spec:
                chair_spec = ChairSpecification(**spec_data)
                db.add(chair_spec)
            
            success_count += 1
            
            # 진행 상황 표시
            if (idx + 1) % 1000 == 0:  # 1000개마다 표시
                print(f"  처리 중... {idx + 1}/{len(df)} ({success_count}개 성공)")
                db.commit()  # 중간 커밋
                
        except Exception as e:
            error_count += 1
            error_msg = f"행 {idx + 1} 처리 중 오류: {str(e)}"
            errors.append(error_msg)
            if error_count <= 5:
                print(f"  ❌ {error_msg}")
            db.rollback()
            
            # 새 세션으로 계속
            db = SessionLocal()
            continue
    
    # 최종 커밋
    try:
        db.commit()
        print(f"\n 마이그레이션 완료!")
        print(f"  성공: {success_count}개")
        print(f"  실패: {error_count}개")
        
        if error_count > 5:
            print(f"\n  (총 {error_count}개 오류 중 처음 5개만 표시됨)")
            
    except Exception as e:
        db.rollback()
        print(f"최종 커밋 중 오류 발생: {str(e)}")

def verify_migration(db: Session):
    """마이그레이션 결과 검증"""
    print("\n 데이터베이스 검증:")
    
    # 의자 수 확인
    chair_count = db.query(Chair).count()
    spec_count = db.query(ChairSpecification).count()
    
    print(f"  - 총 의자 수: {chair_count}개")
    print(f"  - 총 사양 수: {spec_count}개")
    
    # 잘못된 높이 데이터 확인
    from sqlalchemy import and_
    invalid_heights = db.query(ChairSpecification).filter(
        and_(
            ChairSpecification.seat_height_min.isnot(None),
            ChairSpecification.seat_height_max.isnot(None),
            ChairSpecification.seat_height_min > ChairSpecification.seat_height_max
        )
    ).count()
    
    if invalid_heights > 0:
        print(f"잘못된 높이 데이터: {invalid_heights}개")
    else:
        print(f"모든 높이 데이터 정상")

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("🪑 Chair4U 의자 데이터 마이그레이션")
    print("=" * 60)
    
    # 제약조건 수정
    print("\n데이터베이스 제약조건 수정 중...")
    fix_database_constraints()
    
    # 데이터 파일 경로
    data_path = Path(__file__).parent.parent.parent / "data" / "ohouse_chair.xlsx"
    
    if not data_path.exists():
        print(f"\n파일을 찾을 수 없습니다: {data_path}")
        return
    
    # 데이터 로드
    try:
        df = load_chair_data(str(data_path))
    except Exception as e:
        print(f"데이터 로드 실패: {str(e)}")
        return
    
    # 데이터베이스 세션
    db = SessionLocal()
    
    try:
        # 기존 데이터 처리
        if not clear_existing_data(db):
            print("\n마이그레이션 취소됨")
            return
        
        # 마이그레이션 실행
        migrate_to_database(df, db)
        
        # 결과 검증
        verify_migration(db)
        
    except Exception as e:
        print(f"\n❌ 마이그레이션 중 오류: {str(e)}")
        import traceback
        traceback.print_exc()
        db.rollback()
    finally:
        db.close()
        print("\n" + "=" * 60)
        print("마이그레이션 프로세스 종료")
        print("=" * 60)

if __name__ == "__main__":
    main()