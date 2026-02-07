"""
Data Models
===========

Mục đích:
    Định nghĩa data models cho database.

Thư viện sử dụng:
    - sqlalchemy: ORM models
    - Hoặc dataclasses: Simple models

Models cần định nghĩa:
    1. Violation (SQLAlchemy Model)
       - id: int (primary key, auto increment)
       - plate_text: str (biển số xe)
       - timestamp: datetime (thời điểm vi phạm)
       - image_path: str (đường dẫn ảnh vi phạm)
       - confidence: float (độ tin cậy OCR)
       - zone_id: str (vùng vi phạm)
       - created_at: datetime (thời điểm tạo record)

SQLAlchemy Example:
    from sqlalchemy import Column, Integer, String, Float, DateTime
    from sqlalchemy.ext.declarative import declarative_base
    
    Base = declarative_base()
    
    class Violation(Base):
        __tablename__ = 'violations'
        id = Column(Integer, primary_key=True)
        plate_text = Column(String)
        ...

TODO:
    [ ] Định nghĩa Violation model
    [ ] Add indexes cho plate_text và timestamp
    [ ] Add relationship nếu cần thêm tables
"""
