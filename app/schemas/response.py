from pydantic import BaseModel
from typing import List, Optional

class Prediction(BaseModel):
    predictedPrice: int
    priceRatio: float
    confidence: float
    failureProbability: float

class AuctionPrediction(BaseModel):
    failureProbability: float
    expectedSuccessAttempt: int
    optimalBidPrice: int

class RiskAssessment(BaseModel):
    riskLevel: str  # "LOW", "MEDIUM", "HIGH"
    riskFactors: List[str]

class SimilarCase(BaseModel):
    caseId: str
    appraisalValue: int
    finalPrice: int
    priceRatio: float
    failCount: int

class AuctionAnalysisResponse(BaseModel):
    prediction: Prediction
    auctionPrediction: AuctionPrediction
    riskAssessment: RiskAssessment
    similarCases: List[SimilarCase]

    class Config:
        schema_extra = {
            "example": {
                "prediction": {
                    "predictedPrice": 340000000,
                    "priceRatio": 0.8,
                    "confidence": 0.87,
                    "failureProbability": 0.35
                },
                "auctionPrediction": {
                    "failureProbability": 0.35,
                    "expectedSuccessAttempt": 3,
                    "optimalBidPrice": 350000000
                },
                "riskAssessment": {
                    "riskLevel": "MEDIUM",
                    "riskFactors": [
                        "임차인 점유 중",
                        "유찰 횟수 높음"
                    ]
                },
                "similarCases": [
                    {
                        "caseId": "20190520058123",
                        "appraisalValue": 430000000,
                        "finalPrice": 344000000,
                        "priceRatio": 0.8,
                        "failCount": 2
                    }
                ]
            }
        }
