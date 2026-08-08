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
    # Base64 data URI του Grad-CAM heatmap - αποθηκεύεται μαζί με κάθε πρόβλεψη
    # ώστε να είναι διαθέσιμο αργότερα από το Ιστορικό, όχι μόνο τη στιγμή της πρόβλεψης.
    gradcam_image = Column(Text, nullable=True)