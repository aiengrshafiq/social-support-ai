# storage/models.py
import enum
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Enum, ForeignKey, JSON
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import datetime

Base = declarative_base()

class ApplicationStatus(enum.Enum):
    PENDING = "PENDING"
    AWAITING_DOCUMENTS = "AWAITING_DOCUMENTS"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    PENDING_REVIEW = "PENDING_REVIEW"
    APPROVED = "APPROVED"
    DECLINED = "DECLINED"

class Applicant(Base):
    __tablename__ = "applicants"
    
    id = Column(Integer, primary_key=True, index=True)
    emirates_id = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True)
    phone_number = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    applications = relationship("Application", back_populates="applicant")

class Application(Base):
    __tablename__ = "applications"
    
    id = Column(Integer, primary_key=True, index=True)
    applicant_id = Column(Integer, ForeignKey("applicants.id"), nullable=False)
    status = Column(Enum(ApplicationStatus), default=ApplicationStatus.PENDING, nullable=False)
    
    # This will store the final validated JSON data for this application
    validated_data = Column(JSON) 
    
    # This stores the final decision and explanation
    decision_summary = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    applicant = relationship("Applicant", back_populates="applications")

# Add other models here as needed (e.g., Documents, FamilyMembers)