import numpy as np
import pandas as pd
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.base import clone
import numpy as np
import types
import joblib
import os

class AuctionPriceModel:
    """
    경매 낙찰가 예측 모델
    """
    def __init__(self, params=None):
        self.model = None
        self.params = params or {
            'n_estimators': 500,
            'learning_rate': 0.05,
            'max_depth': 7,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'reg:squarederror',
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42
        }

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        모델 학습
        """
        # 모델 초기화
        self.model = XGBRegressor(**self.params)

        # 검증 세트가 제공된 경우
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                eval_metric='rmse',
                early_stopping_rounds=50,
                verbose=True
            )
        else:
            self.model.fit(X_train, y_train)

        return self

    def predict(self, X):
        """
        낙찰가 예측
        """
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")

        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        """
        모델 평가
        """
        y_pred = self.predict(X_test)

        # 평가 메트릭 계산
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        return {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        }

    def save(self, filepath):
        """
        모델 저장
        """
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다.")

        # 디렉토리 확인
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # 모델 저장
        joblib.dump(self.model, filepath)

    def update_model(self, X_train, y_train, X_val=None, y_val=None):
        """
        기존 모델을 새로운 데이터로 증분 학습합니다.
        """
        # 모델 유형에 따라 다른 증분 학습 방법 적용
        if hasattr(self.model, 'partial_fit'):
            # 온라인 학습 지원 모델 (SGDRegressor 등)
            self.model.partial_fit(X_train, y_train)
        else:
            # 온라인 학습을 지원하지 않는 모델
            # 1. 기존 모델과 새 모델의 앙상블 구성
            new_model = clone(self.model)
            new_model.fit(X_train, y_train)

            # 앙상블 가중치 설정 (검증 데이터로 최적화)
            if X_val is not None and y_val is not None:
                # 기존 모델과 새 모델의 예측 결합
                pred_old = self.model.predict(X_val)
                pred_new = new_model.predict(X_val)

                # 최적 가중치 찾기 (간단한 그리드 서치)
                best_score = float('-inf')
                best_weight = 0.5

                for w in np.linspace(0, 1, 11):
                    pred_ensemble = w * pred_old + (1-w) * pred_new
                    score = -mean_squared_error(y_val, pred_ensemble)

                    if score > best_score:
                        best_score = score
                        best_weight = w

                # 앙상블 가중치 저장
                self.ensemble_weights = [best_weight, 1-best_weight]
                self.ensemble_models = [self.model, new_model]

                # 예측 메서드 오버라이드
                def weighted_predict(self, X):
                    pred_old = self.ensemble_models[0].predict(X)
                    pred_new = self.ensemble_models[1].predict(X)
                    return self.ensemble_weights[0] * pred_old + self.ensemble_weights[1] * pred_new

                self.predict = types.MethodType(weighted_predict, self)
            else:
                # 검증 데이터가 없으면 새 모델로 대체
                self.model = new_model

    @classmethod
    def load(cls, filepath):
        """
        저장된 모델 로드
        """
        instance = cls()
        instance.model = joblib.load(filepath)
        return instance


class AuctionFailureModel:
    """
    경매 유찰 확률 예측 모델
    """
    def __init__(self, params=None):
        self.model = None
        # 유찰=1, 낙찰=0 예측을 위한 이진 분류 모델
        self.params = params or {
            'n_estimators': 300,
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_child_weight': 2,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'binary:logistic',
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42
        }

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        모델 학습
        """
        # 클래스 불균형 처리
        pos_count = np.sum(y_train == 1)
        neg_count = np.sum(y_train == 0)
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

        # 파라미터 업데이트
        self.params['scale_pos_weight'] = scale_pos_weight

        # 모델 초기화
        self.model = XGBClassifier(**self.params)

        # 검증 세트가 제공된 경우
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                eval_metric='auc',
                early_stopping_rounds=50,
                verbose=True
            )
        else:
            self.model.fit(X_train, y_train)

        return self

    def predict_proba(self, X):
        """
        유찰 확률 예측
        """
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")

        # 클래스 1(유찰)의 확률 반환
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X):
        """
        유찰 여부 예측 (이진 분류)
        """
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")

        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        """
        모델 평가
        """
        y_pred = self.predict(X_test)
        y_prob = self.predict_proba(X_test)

        # 평가 메트릭 계산
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        return {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'AUC': auc
        }

    def save(self, filepath):
        """
        모델 저장
        """
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다.")

        # 디렉토리 확인
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # 모델 저장
        joblib.dump(self.model, filepath)

    def update_model(self, X_train, y_train, X_val=None, y_val=None):
        """
        기존 모델을 새로운 데이터로 증분 학습합니다.
        """
        # 모델 유형에 따라 다른 증분 학습 방법 적용
        if hasattr(self.model, 'partial_fit'):
            # 온라인 학습 지원 모델 (SGDRegressor 등)
            self.model.partial_fit(X_train, y_train)
        else:
            # 온라인 학습을 지원하지 않는 모델
            # 1. 기존 모델과 새 모델의 앙상블 구성
            new_model = clone(self.model)
            new_model.fit(X_train, y_train)

            # 앙상블 가중치 설정 (검증 데이터로 최적화)
            if X_val is not None and y_val is not None:
                # 기존 모델과 새 모델의 예측 결합
                pred_old = self.model.predict(X_val)
                pred_new = new_model.predict(X_val)

                # 최적 가중치 찾기 (간단한 그리드 서치)
                best_score = float('-inf')
                best_weight = 0.5

                for w in np.linspace(0, 1, 11):
                    pred_ensemble = w * pred_old + (1-w) * pred_new
                    score = -mean_squared_error(y_val, pred_ensemble)

                    if score > best_score:
                        best_score = score
                        best_weight = w

                # 앙상블 가중치 저장
                self.ensemble_weights = [best_weight, 1-best_weight]
                self.ensemble_models = [self.model, new_model]

                # 예측 메서드 오버라이드
                def weighted_predict(self, X):
                    pred_old = self.ensemble_models[0].predict(X)
                    pred_new = self.ensemble_models[1].predict(X)
                    return self.ensemble_weights[0] * pred_old + self.ensemble_weights[1] * pred_new

                self.predict = types.MethodType(weighted_predict, self)
            else:
                # 검증 데이터가 없으면 새 모델로 대체
                self.model = new_model

    @classmethod
    def load(cls, filepath):
        """
        저장된 모델 로드
        """
        instance = cls()
        instance.model = joblib.load(filepath)
        return instance


class AuctionAttemptsModel:
    """
    경매 예상 낙찰 회차 예측 모델
    """
    def __init__(self, params=None):
        self.model = None
        self.params = params or {
            'n_estimators': 400,
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_child_weight': 2,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'reg:squarederror',
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42
        }

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        모델 학습
        """
        # 모델 초기화
        self.model = XGBRegressor(**self.params)

        # 검증 세트가 제공된 경우
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                eval_metric='mae',
                early_stopping_rounds=50,
                verbose=True
            )
        else:
            self.model.fit(X_train, y_train)

        return self

    def predict(self, X):
        """
        예상 낙찰 회차 예측
        """
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")

        # 예측 값을 반올림하여 정수화
        predictions = self.model.predict(X)
        return np.maximum(1, np.round(predictions)).astype(int)

    def evaluate(self, X_test, y_test):
        """
        모델 평가
        """
        y_pred = self.predict(X_test)

        # 평가 메트릭 계산
        mae = mean_absolute_error(y_test, y_pred)

        return {
            'MAE': mae
        }

    def save(self, filepath):
        """
        모델 저장
        """
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다.")

        # 디렉토리 확인
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # 모델 저장
        joblib.dump(self.model, filepath)

    def update_model(self, X_train, y_train, X_val=None, y_val=None):
        """
        기존 모델을 새로운 데이터로 증분 학습합니다.
        """
        # 모델 유형에 따라 다른 증분 학습 방법 적용
        if hasattr(self.model, 'partial_fit'):
            # 온라인 학습 지원 모델 (SGDRegressor 등)
            self.model.partial_fit(X_train, y_train)
        else:
            # 온라인 학습을 지원하지 않는 모델
            # 1. 기존 모델과 새 모델의 앙상블 구성
            new_model = clone(self.model)
            new_model.fit(X_train, y_train)

            # 앙상블 가중치 설정 (검증 데이터로 최적화)
            if X_val is not None and y_val is not None:
                # 기존 모델과 새 모델의 예측 결합
                pred_old = self.model.predict(X_val)
                pred_new = new_model.predict(X_val)

                # 최적 가중치 찾기 (간단한 그리드 서치)
                best_score = float('-inf')
                best_weight = 0.5

                for w in np.linspace(0, 1, 11):
                    pred_ensemble = w * pred_old + (1-w) * pred_new
                    score = -mean_squared_error(y_val, pred_ensemble)

                    if score > best_score:
                        best_score = score
                        best_weight = w

                # 앙상블 가중치 저장
                self.ensemble_weights = [best_weight, 1-best_weight]
                self.ensemble_models = [self.model, new_model]

                # 예측 메서드 오버라이드
                def weighted_predict(self, X):
                    pred_old = self.ensemble_models[0].predict(X)
                    pred_new = self.ensemble_models[1].predict(X)
                    return self.ensemble_weights[0] * pred_old + self.ensemble_weights[1] * pred_new

                self.predict = types.MethodType(weighted_predict, self)
            else:
                # 검증 데이터가 없으면 새 모델로 대체
                self.model = new_model

    @classmethod
    def load(cls, filepath):
        """
        저장된 모델 로드
        """
        instance = cls()
        instance.model = joblib.load(filepath)
        return instance
