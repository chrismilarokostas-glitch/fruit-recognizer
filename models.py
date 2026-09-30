from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from datetime import datetime
from database import Base

class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    fruit_name = Column(String, index=True)
    confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    gradcam_image = Column(Text, nullable=True)