from app.models.auction_model import AuctionPriceModel
from app.schemas.request import AuctionAnalysisRequest
from app.schemas.response import AuctionAnalysisResponse


class AuctionAnalysisService:
    """경매 분석 서비스"""

    def __init__(self):
        self.auction_model = AuctionPriceModel()

    def analyze_auction(self, request: AuctionAnalysisRequest) -> AuctionAnalysisResponse:
        """경매 물건 종합 분석 수행"""
        try:
            # 모델을 통한 분석 실행
            response = self.auction_model.analyze(request)
            return response

        except Exception as e:
            # 오류 처리
            return AuctionAnalysisResponse(
                success=False,
                error_message=f"분석 처리 중 오류: {str(e)}"
            )
