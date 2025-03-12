from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class PricePredictionRequest:
    """가격 예측 요청 스키마"""
    age: int  # 건물 나이
    floor: int  # 층수
    area: float  # 면적(m²)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PricePredictionRequest':
        return cls(
            age=data.get('age'),
            floor=data.get('floor'),
            area=data.get('area')
        )

@dataclass
class PricePredictionResponse:
    """가격 예측 응답 스키마"""
    predicted_price: float
    confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'predicted_price': self.predicted_price,
            'confidence': self.confidence
        }
