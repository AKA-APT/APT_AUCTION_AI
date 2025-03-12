import joblib
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path

class PriceModel:
    """부동산 가격 예측 모델"""

    def __init__(self):
        self.model = None
        self.model_path = Path(__file__).parent.parent.parent / "model_artifacts" / "price_model.joblib"
        self.load_model()

    def load_model(self):
        """저장된 모델을 로드하거나 새 모델을 만듭니다"""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        else:
            # 저장된 모델이 없을 경우 기본 모델 생성
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            # 간단한 예시 데이터로 학습 (실제로는 이 부분에 실제 데이터 학습 코드가 들어감)
            X = np.array([[30, 1, 1000], [20, 2, 800], [40, 3, 1500]])  # 예: 나이, 층수, 면적
            y = np.array([500, 400, 700])  # 예: 가격(단위: 만원)
            self.model.fit(X, y)
            # 모델 저장
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(self.model, self.model_path)

    def predict(self, features):
        """가격 예측 수행"""
        if not self.model:
            self.load_model()

        # 입력 검증
        if not isinstance(features, list) or len(features) != 3:
            raise ValueError("Features must be a list with 3 elements: [age, floor, area]")

        # 예측 수행
        features_array = np.array([features])
        prediction = self.model.predict(features_array)[0]

        return float(prediction)
