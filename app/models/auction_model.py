import joblib
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
from typing import List, Dict, Any, Tuple

from app.schemas.request import AuctionAnalysisRequest
from app.schemas.response import (
    AuctionAnalysisResponse, Scores, Analysis,
    PriceAnalysis, RiskAnalysis, RiskLevel
)


class AuctionPriceModel:
    """경매 가격 예측 모델"""

    def __init__(self):
        self.model = None
        self.model_path = Path(__file__).parent.parent.parent / "model_artifacts" / "auction_price_model.joblib"
        self.load_model()

    def load_model(self):
        """저장된 모델을 로드하거나 새 모델을 만듭니다"""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        else:
            # 저장된 모델이 없을 경우 기본 모델 생성
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            # 여기서는 간단한 예시 학습만 진행 (실제로는 데이터셋을 활용할 것)
            self._train_default_model()

    def _train_default_model(self):
        """기본 모델 학습 (실제 프로젝트에서는 외부 학습 스크립트로 대체)"""
        # 예시 데이터
        X = np.array([
            # 감정가, 청구금액, 면적, 위도, 경도 순
            [500000000, 300000000, 85, 37.5, 127.0],
            [700000000, 500000000, 115, 37.6, 127.1],
            [300000000, 200000000, 60, 37.4, 126.9]
        ])
        y = np.array([450000000, 650000000, 280000000])  # 예측 낙찰가

        self.model.fit(X, y)

        # 모델 저장
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)

    def _extract_features(self, request: AuctionAnalysisRequest) -> np.ndarray:
        """요청 데이터에서 모델 입력 특성 추출"""
        features = [
            request.basic_info.appraised_value,
            request.basic_info.claim_amount,
            request.property_info.size,
            request.property_info.coordinates.latitude,
            request.property_info.coordinates.longitude
        ]
        return np.array([features])

    def analyze(self, request: AuctionAnalysisRequest) -> AuctionAnalysisResponse:
        """경매 물건 종합 분석 수행"""
        if not self.model:
            self.load_model()

        try:
            # 특성 추출
            features = self._extract_features(request)

            # 예상 낙찰가 예측
            expected_bid_price = float(self.model.predict(features)[0])

            # 시세 대비 비율 계산 (가정: 시세는 감정가의 1.2배)
            market_price = request.basic_info.appraised_value * 1.2
            market_ratio = expected_bid_price / market_price

            # 예상 수익률 계산 (단순화된 계산)
            expected_return = (market_price - expected_bid_price) / expected_bid_price

            # 가치 점수 계산 (0-1 사이)
            value_score = min(1.0, max(0.0, 1.0 - market_ratio + 0.3))

            # 입지 점수 계산 (예시 - 실제로는 위치 데이터 기반 모델 필요)
            # 강남/서초/송파 지역이면 높은 점수
            location_score = 0.7  # 기본값
            address = request.property_info.address.lower()
            if '강남' in address or '서초' in address or '송파' in address:
                location_score = 0.9
            elif '강북' in address or '노원' in address:
                location_score = 0.6

            # 법적 안정성 점수 계산 (예시 - 실제로는 더 복잡한 규칙 필요)
            legal_score = 0.8  # 기본값
            case_type = request.basic_info.case_type.lower()
            if '임의경매' in case_type:
                legal_score = 0.9
            elif '강제경매' in case_type:
                legal_score = 0.7

            # 전체 점수 계산 (0-100)
            total_score = (value_score * 0.4 + location_score * 0.4 + legal_score * 0.2) * 100

            # 위험 요소 분석
            risk_factors = []
            risk_level = RiskLevel.LOW

            if expected_return < 0.1:
                risk_factors.append("수익률이 낮음 (10% 미만)")
                risk_level = RiskLevel.MEDIUM

            if market_ratio > 0.9:
                risk_factors.append("시세 대비 높은 낙찰가 예상")
                risk_level = RiskLevel.HIGH

            if legal_score < 0.7:
                risk_factors.append("법적 안정성 우려")

            if not risk_factors:
                risk_factors.append("특별한 위험 요소 없음")

            # 결과 구성
            scores = Scores(
                total=total_score,
                value=value_score,
                location=location_score,
                legal=legal_score
            )

            price_analysis = PriceAnalysis(
                market_ratio=market_ratio,
                expected_bid_price=expected_bid_price,
                expected_return=expected_return
            )

            risk_analysis = RiskAnalysis(
                risk_level=risk_level,
                risk_factors=risk_factors
            )

            analysis = Analysis(
                price_analysis=price_analysis,
                risk_analysis=risk_analysis
            )

            return AuctionAnalysisResponse(
                success=True,
                scores=scores,
                analysis=analysis
            )

        except Exception as e:
            return AuctionAnalysisResponse(
                success=False,
                error_message=f"분석 중 오류 발생: {str(e)}"
            )
