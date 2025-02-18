from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class HumanDetectorDatabase:
    def __init__(self, database_url):
        self.engine = create_engine(database_url)
        self.Session = sessionmaker(bind = self.engine)
                
    def create_tables(self):
        Base.metadata.create_all(self.engine)

    def drop_tables(self):
        Base.metadata.drop_all(self.engine)

    def add_record(self, record):
        with self.Session() as session:
            session.add(record)
            session.commit()

    def get_record_by_id(self, model, record_id):
        with self.Session() as session:
            return session.query(model).get(record_id)

    def get_all_records(self, model):
        with self.Session() as session:
            return session.query(model).all()

    def update_record(self, model, record_id, **kwargs):
        with self.Session() as session:
            record = session.query(model).get(record_id)
            if record:
                for key, value in kwargs.items():
                    setattr(record, key, value)
                session.commit()

    def delete_record(self, model, record_id):
        with self.Session() as session:
            record = session.query(model).get(record_id)
            if record:
                session.delete(record)
                session.commit()
        
class PredictionRecord(Base):
    __tablename__ = 'Predictions'

    query_id = Column(Integer, primary_key = True)
    time = Column(String(32))
    
    query_image_file = Column(String(64))
    result_image_file = Column(String(64))
    num_humans = Column(Integer)
    