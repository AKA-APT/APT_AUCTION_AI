import os
import joblib
import numpy as np
import pandas as pd
from app.services.preprocessing import preprocess_input_data
from app.services.feature_engineering import (
    calculate_optimal_bid, analyze_risk_factors,
    determine_risk_level
)

class AuctionPredictionPipeline:
    """
    경매 예측 파이프라인
    """
    def __init__(self, model_dir='model_artifacts'):
        self.model_dir = model_dir
        self.preprocessor = None
        self.price_model = None
        self.failure_model = None
        self.attempts_model = None
        self.load_models()

    def load_models(self):
        """
        저장된 모델과 전처리 파이프라인 로드
        """
        # 전처리 파이프라인 로드
        preprocessor_path = os.path.join(self.model_dir, 'preprocessor.joblib')
        if os.path.exists(preprocessor_path):
            self.preprocessor = joblib.load(preprocessor_path)
        else:
            raise FileNotFoundError(f"전처리 파이프라인을 찾을 수 없습니다: {preprocessor_path}")

        # 낙찰가 예측 모델 로드
        price_model_path = os.path.join(self.model_dir, 'price_model.joblib')
        if os.path.exists(price_model_path):
            self.price_model = joblib.load(price_model_path)

        # 유찰 확률 모델 로드
        failure_model_path = os.path.join(self.model_dir, 'failure_model.joblib')
        if os.path.exists(failure_model_path):
            self.failure_model = joblib.load(failure_model_path)

        # 예상 낙찰 회차 모델 로드
        attempts_model_path = os.path.join(self.model_dir, 'attempts_model.joblib')
        if os.path.exists(attempts_model_path):
            self.attempts_model = joblib.load(attempts_model_path)

    def predict(self, input_data):
        """
        경매 결과 예측
        """
        # 입력 데이터를 DataFrame으로 변환
        input_df = preprocess_input_data(input_data)

        # 데이터 전처리
        processed_data = self.preprocessor.transform(input_df)

        # 예측 수행
        results = {}

        # 낙찰가 예측 (모델이 있는 경우)
        if self.price_model is not None:
            predicted_price = self.price_model.predict(processed_data)[0]
            appraisal_value = input_df['appraisalValue'].values[0]
            price_ratio = predicted_price / appraisal_value

            # 예측 정보 추가
            results["prediction"] = {
                "predictedPrice": int(predicted_price),
                "priceRatio": float(price_ratio),
            }
        else:
            # 모델이 없는 경우 더미 값
            results["prediction"] = {
                "predictedPrice": 0,
                "priceRatio": 0,
                "confidence": 0
            }

        # 유찰 확률 예측 (모델이 있는 경우)
        failure_prob = 0.5  # 기본값
        if self.failure_model is not None:
            failure_prob = self.failure_model.predict_proba(processed_data)[0]
            results["prediction"]["failureProbability"] = float(failure_prob)
        else:
            results["prediction"]["failureProbability"] = failure_prob

        # 예상 낙찰 회차 예측 (모델이 있는 경우)
        expected_attempts = 1  # 기본값
        if self.attempts_model is not None:
            expected_attempts = self.attempts_model.predict(processed_data)[0]

        # 최적 입찰가 계산
        predicted_price = results["prediction"]["predictedPrice"]
        optimal_bid = calculate_optimal_bid(predicted_price, failure_prob, input_df)

        # 경매 예측 정보 추가
        results["auctionPrediction"] = {
            "failureProbability": float(failure_prob),
            "expectedSuccessAttempt": int(expected_attempts),
            "optimalBidPrice": int(optimal_bid)
        }

        # 위험 요소 분석
        risk_factors = analyze_risk_factors(input_df, predicted_price, failure_prob)
        risk_level = determine_risk_level(risk_factors, failure_prob, results["prediction"]["priceRatio"])

        # 위험 평가 정보 추가
        results["riskAssessment"] = {
            "riskLevel": risk_level,
            "riskFactors": risk_factors
        }

        return results

# 서비스 함수들
def predict_auction_results(auction_data):
    """
    경매 결과 예측 서비스 함수
    """
    pipeline = AuctionPredictionPipeline()
    return pipeline.predict(auction_data)

def validate_auction_data(auction_data):
    """
    경매 데이터 검증 서비스 함수
    """
    try:
        # 데이터프레임으로 변환 시도
        input_df = preprocess_input_data(auction_data)
        return {
            "status": "valid",
            "message": "입력 데이터가 유효합니다.",
            "fields": list(input_df.columns)
        }
    except Exception as e:
        return {
            "status": "invalid",
            "error": f"유효하지 않은 입력 데이터: {str(e)}"
        }
