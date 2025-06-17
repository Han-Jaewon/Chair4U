# models.py
from sqlalchemy import (
    Column, Integer, String, Text, Numeric, Boolean,
    TIMESTAMP, ForeignKey, CheckConstraint, UniqueConstraint
)
from sqlalchemy.sql import func
from project.backend.app.database import Base

class Person(Base):
    __tablename__ = 'persons'

    person_id = Column(Integer, primary_key=True, index=True)  # SERIAL
    image_name = Column(String(255), nullable=False)
    human_height = Column(Integer, nullable=False)
    buttock_popliteal_length = Column(Numeric(10, 2), nullable=False)  # A
    popliteal_height = Column(Numeric(10, 2), nullable=False)           # B
    hip_breadth = Column(Numeric(10, 2), nullable=False)                # C
    sitting_height = Column(Numeric(10, 2), nullable=False)             # F
    shoulder_breadth = Column(Numeric(10, 2), nullable=False)           # G
    created_at = Column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    updated_at = Column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )

class Chair(Base):
    __tablename__ = 'chairs'

    chair_id = Column(Integer, primary_key=True, index=True)  # SERIAL
    product_url = Column(Text)
    brand_name = Column(String(100), nullable=False)
    product_name = Column(String(255), nullable=False)
    price = Column(Integer)
    rating = Column(Numeric(2, 1))
    review_count = Column(Integer)
    has_headrest = Column(Boolean, default=False)
    has_armrest = Column(Boolean, default=False)
    has_lumbar_support = Column(Boolean, default=False)
    has_height_adjustment = Column(Boolean, default=False)
    has_tilting = Column(Boolean, default=False)
    backrest_type = Column(
        String(10),
        CheckConstraint(
            "(backrest_type IN ('곧','꺾') OR backrest_type IS NULL)",
            name="check_backrest_type"
        )
    )
    created_at = Column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    updated_at = Column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )

class ChairSpecification(Base):
    __tablename__ = 'chair_specifications'
    __table_args__ = (
        CheckConstraint(
            "seat_height_max IS NULL OR seat_height_min IS NULL OR seat_height_max >= seat_height_min",
            name="check_valid_seat_height"
        ),
    )

    spec_id = Column(Integer, primary_key=True, index=True)  # SERIAL
    chair_id = Column(
        Integer,
        ForeignKey('chairs.chair_id', ondelete='CASCADE'),
        unique=True,
        nullable=False
    )
    seat_height_min = Column(Integer)  # h8_지면-좌석 높이_MIN
    seat_height_max = Column(Integer)  # h8_지면-좌석 높이_MAX
    seat_width = Column(Integer)       # b3_좌석 가로 길이
    seat_depth = Column(Integer)       # t4_좌석 세로 길이 일반
    backrest_width = Column(Integer)   # b4_등받이 가로 길이
    backrest_height = Column(Integer)  # h7_등받이 세로 길이
    created_at = Column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    updated_at = Column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )

class Recommendation(Base):
    __tablename__ = 'recommendations'
    __table_args__ = (
        UniqueConstraint('person_id', 'chair_id', name='uix_person_chair'),
        UniqueConstraint('person_id', 'rank', name='uix_person_rank'),
        CheckConstraint(
            "match_score >= 0 AND match_score <= 100",
            name="check_match_score_range"
        ),
        CheckConstraint(
            "rank > 0",
            name="check_rank_positive"
        ),
    )

    rec_id = Column(Integer, primary_key=True, index=True)  # SERIAL
    person_id = Column(
        Integer,
        ForeignKey('persons.person_id', ondelete='CASCADE'),
        nullable=False
    )
    chair_id = Column(
        Integer,
        ForeignKey('chairs.chair_id', ondelete='CASCADE'),
        nullable=False
    )
    match_score = Column(Numeric(5, 2), nullable=False)
    rank = Column(Integer, nullable=False)
    created_at = Column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=False
    )