from flask import Blueprint, request, jsonify
from app.services.auction_service import AuctionAnalysisService
from app.schemas.request import AuctionAnalysisRequest

predictions_bp = Blueprint('predictions', __name__)

@predictions_bp.route('/analyze-auction', methods=['POST'])
def analyze_auction():
    """경매 분석 API 엔드포인트"""
    try:
        # 요청 데이터 파싱
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "errorMessage": "No input data provided"}), 400

        # 요청 객체 생성
        auction_request = AuctionAnalysisRequest.from_dict(data)

        # 서비스 호출
        service = AuctionAnalysisService()
        result = service.analyze_auction(auction_request)

        # 응답 반환
        return jsonify(result.to_dict()), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "errorMessage": f"Analysis failed: {str(e)}"
        }), 500
