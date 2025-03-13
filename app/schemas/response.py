from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Literal
from enum import Enum


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass
class Scores:
    total: float  # 종합 점수 (0-100)
    value: float  # 가치 점수 (0-1)
    location: float  # 입지 점수 (0-1)
    legal: float  # 법적 안정성 점수 (0-1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total': self.total,
            'value': self.value,
            'location': self.location,
            'legal': self.legal
        }


@dataclass
class PriceAnalysis:
    market_ratio: float  # 시세 대비 비율
    expected_bid_price: float  # 예상 낙찰가
    expected_return: float  # 예상 수익률

    def to_dict(self) -> Dict[str, Any]:
        return {
            'marketRatio': self.market_ratio,
            'expectedBidPrice': self.expected_bid_price,
            'expectedReturn': self.expected_return
        }


@dataclass
class RiskAnalysis:
    risk_level: RiskLevel  # 위험 수준
    risk_factors: List[str]  # 주요 위험 요소

    def to_dict(self) -> Dict[str, Any]:
        return {
            'riskLevel': self.risk_level.value,
            'riskFactors': self.risk_factors
        }


@dataclass
class Analysis:
    price_analysis: PriceAnalysis
    risk_analysis: RiskAnalysis

    def to_dict(self) -> Dict[str, Any]:
        return {
            'priceAnalysis': self.price_analysis.to_dict(),
            'riskAnalysis': self.risk_analysis.to_dict()
        }


@dataclass
class AuctionAnalysisResponse:
    success: bool
    scores: Optional[Scores] = None
    analysis: Optional[Analysis] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {'success': self.success}

        if self.scores:
            result['scores'] = self.scores.to_dict()

        if self.analysis:
            result['analysis'] = self.analysis.to_dict()

        if self.error_message:
            result['errorMessage'] = self.error_message

        return result
