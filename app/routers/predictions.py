from fastapi import APIRouter, HTTPException
from app.schemas.request import AuctionAnalysisRequest
from app.schemas.response import AuctionAnalysisResponse
from app.services.auction_service import predict_auction_results, validate_auction_data

router = APIRouter(
    prefix="/api/v1",
    tags=["predictions"],
)

@router.post("/predict-auction", response_model=AuctionAnalysisResponse)
async def predict_auction(request: AuctionAnalysisRequest):
    """
    경매 결과 예측 API
    """
    try:
        # 요청 데이터를 딕셔너리로 변환
        request_dict = request.dict()

        # 예측 수행
        result = predict_auction_results(request_dict)

        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"예측 실패: {str(e)}"
        )

@router.post("/validate-input")
async def validate_input(request: AuctionAnalysisRequest):
    """
    입력 데이터 검증 API
    """
    try:
        # 요청 데이터를 딕셔너리로 변환
        request_dict = request.dict()

        # 데이터 검증
        validation_result = validate_auction_data(request_dict)

        if validation_result["status"] == "valid":
            return validation_result
        else:
            raise HTTPException(
                status_code=400,
                detail=validation_result["error"]
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"유효하지 않은 입력 데이터: {str(e)}"
        )
