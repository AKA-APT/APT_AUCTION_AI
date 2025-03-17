```
pip install -r requirements.txt
```

1.	모델 학습:
```
# Windows에서 set, Linux/macOS에서 export
set MONGO_URI=      .env의 MONGO_URI 값
python train_models.py --mongo-uri $env:MONGO_URI --db-name auction_db --collection auction_data
```

2.	API 서버 실행:
```
uvicorn main:app --reload
```

3.	API 테스트:
```
curl -X POST http://localhost:8000/api/v1/predict-auction \
-H "Content-Type: application/json" \
-d @example_request.json
```
