import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

class AuctionDataPreprocessor(BaseEstimator, TransformerMixin):
    """
    경매 데이터 전처리 클래스
    """
    def __init__(self):
        self.categorical_cols = [
            'courtName', 'caseType', 'propertyType',
            'state', 'city', 'dong', 'floorInfo'
        ]
        self.numerical_cols = [
            'appraisalValue', 'minimumBidPrice', 'failCount',
            'totalArea', 'latitude', 'longitude',
            'avgAuctionPriceRate3m', 'avgAuctionPriceRate12m', 'avgFailCount3m'
        ]
        self.binary_cols = ['isOccupied']

    def fit(self, X, y=None):
        # 수치형 변수 전처리 파이프라인
        self.numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # 범주형 변수 전처리 파이프라인
        self.categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # 이진 변수 인코더
        self.binary_encoder = LabelEncoder()

        # 컬럼 트랜스포머 생성
        self.preprocessor = ColumnTransformer([
            ('numerical', self.numerical_pipeline, self.numerical_cols),
            ('categorical', self.categorical_pipeline, self.categorical_cols),
        ], remainder='drop')

        # 데이터에 컬럼 트랜스포머 적용
        self.preprocessor.fit(X)

        # 이진 변수 인코더 적용
        if 'isOccupied' in X.columns:
            self.binary_encoder.fit(X['isOccupied'])

        return self

    def transform(self, X):
        # 특성 추가
        X_copy = X.copy()
        X_copy = self._engineer_features(X_copy)

        # 전처리 파이프라인 적용
        X_transformed = self.preprocessor.transform(X_copy)

        # 이진 변수 처리
        if 'isOccupied' in X_copy.columns:
            X_occupied = self.binary_encoder.transform(X_copy['isOccupied']).reshape(-1, 1)
            X_transformed = np.hstack((X_transformed, X_occupied))

        return X_transformed

    def _engineer_features(self, X):
        """
        특성 가공
        """
        # 감정가 대비 최저입찰가 비율
        if 'appraisalValue' in X.columns and 'minimumBidPrice' in X.columns:
            X['bidRatio'] = X['minimumBidPrice'] / X['appraisalValue']

        # 로그 변환 (큰 값을 가진 변수)
        for col in ['appraisalValue', 'minimumBidPrice']:
            if col in X.columns:
                X[f'{col}_log'] = np.log1p(X[col])

        return X


def create_preprocessing_pipeline():
    """
    전처리 파이프라인 생성
    """
    return AuctionDataPreprocessor()


def preprocess_input_data(data_dict):
    """
    API 입력 데이터를 DataFrame으로 변환
    """
    # 중첩된 구조의 입력 데이터를 플랫한 형식으로 변환
    flat_data = {}

    # auctionInfo 필드 처리
    auction_info = data_dict.get('auctionInfo', {})
    flat_data['caseId'] = auction_info.get('caseId')
    flat_data['courtName'] = auction_info.get('courtName')
    flat_data['caseType'] = auction_info.get('caseType')
    flat_data['appraisalValue'] = auction_info.get('appraisalValue')
    flat_data['minimumBidPrice'] = auction_info.get('minimumBidPrice')
    flat_data['failCount'] = auction_info.get('failCount', 0)

    # propertyInfo 필드 처리
    property_info = data_dict.get('propertyInfo', {})
    flat_data['propertyType'] = property_info.get('propertyType')

    # location 정보
    location = property_info.get('location', {})
    flat_data['state'] = location.get('state')
    flat_data['city'] = location.get('city')
    flat_data['dong'] = location.get('dong')
    flat_data['latitude'] = location.get('latitude')
    flat_data['longitude'] = location.get('longitude')

    # 건물 세부정보
    building_details = property_info.get('buildingDetails', {})
    flat_data['structure'] = building_details.get('structure')
    flat_data['totalArea'] = building_details.get('totalArea')
    flat_data['floorInfo'] = building_details.get('floorInfo')

    # 점유 정보
    occupancy_info = property_info.get('occupancyInfo', {})
    flat_data['isOccupied'] = occupancy_info.get('isOccupied', False)
    flat_data['tenantCount'] = occupancy_info.get('tenantCount', 0)

    # 시장 정보
    market_info = data_dict.get('marketInfo', {})
    flat_data['avgAuctionPriceRate3m'] = market_info.get('avgAuctionPriceRate3m')
    flat_data['avgAuctionPriceRate12m'] = market_info.get('avgAuctionPriceRate12m')
    flat_data['avgFailCount3m'] = market_info.get('avgFailCount3m')

    # DataFrame으로 변환
    return pd.DataFrame([flat_data])
