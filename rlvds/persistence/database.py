"""
Database Connection & Operations
================================

Mục đích:
    Quản lý kết nối SQLite database.

Thư viện sử dụng:
    - sqlite3: Built-in SQLite
    - Hoặc sqlalchemy: ORM (recommended)

Input:
    - database_url: str (ví dụ: "sqlite:///data/rlvds.db")

Output:
    - Database connection/session

Classes cần implement:
    1. Database
       - __init__(url: str)
       - connect() -> None
       - disconnect() -> None
       - get_session() -> Session (nếu dùng SQLAlchemy)
       - create_tables() -> None
       - execute(query: str, params: dict) -> Any

Cách sử dụng SQLAlchemy:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    engine = create_engine(url)
    Session = sessionmaker(bind=engine)
    session = Session()

Tables cần tạo:
    1. violations
       - id: INTEGER PRIMARY KEY
       - plate_text: VARCHAR
       - timestamp: DATETIME
       - image_path: VARCHAR
       - confidence: FLOAT
       - zone_id: VARCHAR

TODO:
    [ ] Setup SQLAlchemy hoặc sqlite3
    [ ] Implement connection management
    [ ] Create tables on first run
    [ ] Add connection pooling (optional)
"""
