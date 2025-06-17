import pandas as pd
import psycopg2
from psycopg2 import sql
import numpy as np
from datetime import datetime

# 데이터베이스 연결 설정
DB_CONFIG = {
    'host': 'localhost',
    'database': 'chair4u_db',
    'user': 'chair4u_user',
    'password': 'inisw06',
    'port': 5432
}

def connect_to_db():
    """PostgreSQL 데이터베이스에 연결"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("데이터베이스 연결 성공")
        return conn
    except Exception as e:
        print(f"데이터베이스 연결 실패: {e}")
        return None

def create_tables(conn):
    """o_chairs와 o_chair_specifications 테이블 생성"""
    cursor = conn.cursor()
    
    # o_chairs 테이블 생성
    create_chairs_table = """
    CREATE TABLE IF NOT EXISTS o_chairs (
        chair_id SERIAL PRIMARY KEY,
        product_url TEXT,
        brand_name VARCHAR(100) NOT NULL,
        product_name VARCHAR(255) NOT NULL,
        price INTEGER,
        rating NUMERIC(3,1),
        review_count INTEGER,
        has_headrest BOOLEAN DEFAULT FALSE,
        has_armrest BOOLEAN DEFAULT FALSE,
        has_lumbar_support BOOLEAN DEFAULT FALSE,
        has_height_adjustment BOOLEAN DEFAULT FALSE,
        has_tilting BOOLEAN DEFAULT FALSE,
        backrest_type VARCHAR(10) CHECK (backrest_type IN ('곧', '꺾', NULL)),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    
    # O_chair_specifications 테이블 생성
    create_specs_table = """
    CREATE TABLE IF NOT EXISTS o_chair_specifications (
        spec_id SERIAL PRIMARY KEY,
        chair_id INTEGER UNIQUE NOT NULL REFERENCES o_chairs(chair_id) ON DELETE CASCADE,
        seat_height_min INTEGER,
        seat_height_max INTEGER,
        seat_width INTEGER,
        seat_depth INTEGER,
        backrest_width INTEGER,
        backrest_height INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT O_valid_seat_height CHECK (
            seat_height_max IS NULL OR seat_height_min IS NULL OR 
            seat_height_max >= seat_height_min
        )
    );
    """
    
    try:
        cursor.execute(create_chairs_table)
        cursor.execute(create_specs_table)
        conn.commit()
        print("테이블 생성 완료!")
    except Exception as e:
        print(f"테이블 생성 실패: {e}")
        conn.rollback()
    finally:
        cursor.close()

def clear_tables(conn):
    """기존 데이터 삭제"""
    cursor = conn.cursor()
    try:
        cursor.execute("TRUNCATE TABLE o_chair_specifications CASCADE;")
        cursor.execute("TRUNCATE TABLE o_chairs CASCADE;")
        conn.commit()
        print("기존 데이터 삭제 완료!")
    except Exception as e:
        print(f"데이터 삭제 실패: {e}")
        conn.rollback()
    finally:
        cursor.close()

def convert_boolean(value):
    """O/X를 Boolean으로 변환"""
    if pd.isna(value) or value == '':
        return False
    return value == 'O'

def convert_numeric(value):
    """숫자 변환 (NULL 처리)"""
    if pd.isna(value) or value == '':
        return None
    return int(value) if isinstance(value, (int, float)) else None

def convert_rating(value):
    """별점 변환 (NULL 처리)"""
    if pd.isna(value) or value == '':
        return None
    try:
        return float(value)
    except:
        return None

def convert_backrest_type(value):
    """등받이 타입 변환"""
    if pd.isna(value) or value == '' or value not in ['곧', '꺾']:
        return None
    return value

def insert_data_from_excel(conn, excel_path):
    """엑셀 파일에서 데이터를 읽어 PostgreSQL에 삽입"""
    cursor = None
    try:
        # 엑셀 파일 읽기
        df = pd.read_excel(excel_path)
        print(f"엑셀 파일 읽기 완료: {len(df)}개 행")
        
        cursor = conn.cursor()
        
        # 데이터 삽입
        success_count = 0
        error_count = 0
        
        for index, row in df.iterrows():
            try:
                # chair_id는 1부터 시작
                chair_id = index + 1
                
                # o_chairs 테이블 데이터 준비
                chair_data = {
                    'chair_id': chair_id,
                    'product_url': row.get('링크', ''),
                    'brand_name': row.get('브랜드명', ''),
                    'product_name': row.get('제품명', ''),
                    'price': convert_numeric(row.get('가격')),
                    'rating': convert_rating(row.get('별점')),
                    'review_count': convert_numeric(row.get('리뷰 갯수')),
                    'has_headrest': convert_boolean(row.get('헤드레스트 유무')),
                    'has_armrest': convert_boolean(row.get('팔걸이 유무')),
                    'has_lumbar_support': convert_boolean(row.get('요추지지대 유무')),
                    'has_height_adjustment': convert_boolean(row.get('높이 조절 레버 유무')),
                    'has_tilting': convert_boolean(row.get('틸팅 여부')),
                    'backrest_type': convert_backrest_type(row.get('등받이 곧/꺾'))
                }
                
                # o_chairs에 삽입
                insert_chair_query = """
                INSERT INTO o_chairs (chair_id, product_url, brand_name, product_name, 
                                     price, rating, review_count, has_headrest, 
                                     has_armrest, has_lumbar_support, has_height_adjustment, 
                                     has_tilting, backrest_type)
                VALUES (%(chair_id)s, %(product_url)s, %(brand_name)s, %(product_name)s,
                        %(price)s, %(rating)s, %(review_count)s, %(has_headrest)s,
                        %(has_armrest)s, %(has_lumbar_support)s, %(has_height_adjustment)s,
                        %(has_tilting)s, %(backrest_type)s)
                """
                cursor.execute(insert_chair_query, chair_data)
                
                # o_chair_specifications 테이블 데이터 준비
                spec_data = {
                    'chair_id': chair_id,
                    'seat_height_min': convert_numeric(row.get('h8_지면-좌석 높이_MIN')),
                    'seat_height_max': convert_numeric(row.get('h8_지면-좌석 높이_MAX')),
                    'seat_width': convert_numeric(row.get('b3_좌석 가로 길이')),
                    'seat_depth': convert_numeric(row.get('t4_좌석 세로 길이 일반')),
                    'backrest_width': convert_numeric(row.get('b4_등받이 가로 길이')),
                    'backrest_height': convert_numeric(row.get('h7_등받이 세로 길이'))
                }
                
                # o_chair_specifications에 삽입
                insert_spec_query = """
                INSERT INTO o_chair_specifications (chair_id, seat_height_min, seat_height_max,
                                                   seat_width, seat_depth, backrest_width,
                                                   backrest_height)
                VALUES (%(chair_id)s, %(seat_height_min)s, %(seat_height_max)s,
                        %(seat_width)s, %(seat_depth)s, %(backrest_width)s,
                        %(backrest_height)s)
                """
                cursor.execute(insert_spec_query, spec_data)
                
                success_count += 1
                
                # 진행 상황 출력
                if (index + 1) % 100 == 0: # 100개 마다
                    print(f"진행 중... {index + 1}/{len(df)} 완료")
                    
            except Exception as e:
                error_count += 1
                print(f"행 {index + 1} 삽입 실패: {e}")
                continue
        
        # 커밋
        conn.commit()
        print(f"\n데이터 삽입 완료!")
        print(f"성공: {success_count}개")
        print(f"실패: {error_count}개")
        
    except Exception as e:
        print(f"데이터 삽입 중 오류 발생: {e}")
        if conn:
            conn.rollback()
    finally:
        if cursor:
            cursor.close()

def verify_data(conn):
    """삽입된 데이터 검증"""
    cursor = conn.cursor()
    
    try:
        # o_chairs 테이블 확인
        cursor.execute("SELECT COUNT(*) FROM o_chairs;")
        chair_count = cursor.fetchone()[0]
        
        # o_chair_specifications 테이블 확인
        cursor.execute("SELECT COUNT(*) FROM o_chair_specifications;")
        spec_count = cursor.fetchone()[0]
        
        print(f"\n데이터 검증:")
        print(f"o_chairs 테이블: {chair_count}개 행")
        print(f"o_chair_specifications 테이블: {spec_count}개 행")
        
        # 샘플 데이터 출력
        print("\n샘플 데이터 (처음 3개):")
        cursor.execute("""
            SELECT c.chair_id, c.brand_name, c.product_name, c.price, 
                   s.seat_height_min, s.seat_height_max
            FROM o_chairs c
            JOIN o_chair_specifications s ON c.chair_id = s.chair_id
            ORDER BY c.chair_id
            LIMIT 3;
        """)
        
        for row in cursor.fetchall():
            print(f"ID: {row[0]}, 브랜드: {row[1]}, 제품명: {row[2][:30]}..., "
                  f"가격: {row[3]}, 좌석높이: {row[4]}-{row[5]}")
            
    except Exception as e:
        print(f"데이터 검증 실패: {e}")
    finally:
        cursor.close()

def main():
    """메인 실행 함수"""
    import os
    
    # 현재 스크립트의 디렉토리 경로
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 엑셀 파일 경로 (여러 경로 시도)
    possible_paths = [
        'ohouse_chair.xlsx',  # 현재 디렉토리
        '../data/ohouse_chair.xlsx',  # 상위 디렉토리의 data 폴더
        os.path.join(script_dir, 'ohouse_chair.xlsx'),  # 스크립트와 같은 디렉토리
        os.path.join(script_dir, '..', 'data', 'ohouse_chair.xlsx'),  # 스크립트 상위의 data 폴더
        os.path.join(script_dir, '..', '..', 'data', 'ohouse_chair.xlsx'),  # 더 상위의 data 폴더
    ]
    
    excel_path = None
    for path in possible_paths:
        if os.path.exists(path):
            excel_path = path
            print(f"엑셀 파일 발견: {excel_path}")
            break
    
    if not excel_path:
        print("엑셀 파일을 찾을 수 없습니다.")
        print("시도한 경로:")
        for path in possible_paths:
            print(f"  - {path}")
        print("\n현재 디렉토리:", os.getcwd())
        print("스크립트 디렉토리:", script_dir)
        return
    
    # 데이터베이스 연결
    conn = connect_to_db()
    if not conn:
        return
    
    try:
        # 1. 테이블 생성
        create_tables(conn)
        
        # 2. 기존 데이터 삭제 (선택사항)
        response = input("\n기존 데이터를 삭제하시겠습니까? (y/n): ")
        if response.lower() == 'y':
            clear_tables(conn)
        
        # 3. 엑셀 데이터 삽입
        print(f"\n엑셀 파일 '{excel_path}'에서 데이터를 읽어옵니다...")
        insert_data_from_excel(conn, excel_path)
        
        # 4. 데이터 검증
        verify_data(conn)
        
    finally:
        # 연결 종료
        conn.close()
        print("\n데이터베이스 연결 종료")

if __name__ == "__main__":
    main()