from app.models.ml_models import PriceModel
from app.schemas.request import PricePredictionRequest, PricePredictionResponse

class PredictorService:
    """예측 서비스 클래스"""

    def __init__(self):
        self.price_model = PriceModel()

    def predict_price(self, request: PricePredictionRequest) -> PricePredictionResponse:
        """가격 예측 로직 실행"""
        # 특성 추출
        features = [request.age, request.floor, request.area]

        # 모델 예측 수행
        predicted_price = self.price_model.predict(features)

        # 응답 생성
        return PricePredictionResponse(
            predicted_price=predicted_price,
            confidence=0.85  # 실제로는 모델에서 불확실성 계산 필요
        )
