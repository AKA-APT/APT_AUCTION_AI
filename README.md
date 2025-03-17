```
pip install -r requirements.txt
```

1.	모델 학습:
```
python train_models.py
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
