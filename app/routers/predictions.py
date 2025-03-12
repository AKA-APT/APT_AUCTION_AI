from flask import Blueprint, request, jsonify
from app.services.predictor import PredictorService
from app.schemas.request import PricePredictionRequest

predictions_bp = Blueprint('predictions', __name__)

@predictions_bp.route('/predict-price', methods=['POST'])
def predict_price():
    """가격 예측 API 엔드포인트"""
    try:
        # 요청 데이터 파싱
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # 요청 검증 및 객체 생성
        try:
            pred_request = PricePredictionRequest.from_dict(data)
        except (KeyError, TypeError, ValueError) as e:
            return jsonify({"error": f"Invalid input: {str(e)}"}), 400

        # 예측 서비스 호출
        service = PredictorService()
        result = service.predict_price(pred_request)

        # 응답 반환
        return jsonify(result.to_dict()), 200

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
