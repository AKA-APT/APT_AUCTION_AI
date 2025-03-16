from fastapi import FastAPI
from app.routers import predictions

app = FastAPI(
    title="경매 예측 AI API",
    description="XGBoost 기반 경매 유찰 확률 및 낙찰가 예측 API",
    version="1.0.0"
)

# 라우터 등록
app.include_router(predictions.router)

@app.get("/")
async def root():
    return {
        "message": "경매 예측 AI API",
        "version": "1.0",
        "endpoints": {
            "predict": "/api/v1/predict-auction",
            "validate": "/api/v1/validate-input"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
