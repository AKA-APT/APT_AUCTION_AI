import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import joblib
import os
import logging
import argparse
from datetime import datetime

# 로컬 모듈 임포트
from app.services.preprocessing import create_preprocessing_pipeline
from app.models.auction_model import AuctionPriceModel, AuctionFailureModel, AuctionAttemptsModel
from app.services.feature_engineering import create_train_dataset_from_mongo

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_models(mongo_uri, db_name, collection_name, output_dir='model_artifacts'):
    """
    MongoDB 데이터로부터 모델 학습
    """
    logger.info("MongoDB 연결 중...")
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]

    logger.info("학습 데이터셋 생성 중...")
    df = create_train_dataset_from_mongo(collection)
    logger.info(f"데이터셋 크기: {df.shape}")

    # 특성과 타겟 분리
    logger.info("특성과 타겟 분리 중...")
    X = df.drop(['actualPrice', 'wasFailure'], axis=1, errors='ignore')
    y_price = df['actualPrice'] if 'actualPrice' in df.columns else None
    y_failure = df['wasFailure'] if 'wasFailure' in df.columns else None

    # 시간 기반 분할 (최근 데이터를 테스트 세트로)
    logger.info("데이터 분할 중...")

    # 데이터를 날짜 기준으로 정렬 (사건번호에 날짜 정보가 포함된 경우)
    if 'caseId' in X.columns:
        X['year'] = X['caseId'].str[:4].astype(int)  # 첫 4자리를 연도로 가정
        sorted_indices = X['year'].argsort()
        X = X.iloc[sorted_indices].drop('year', axis=1)
        if y_price is not None:
            y_price = y_price.iloc[sorted_indices]
        if y_failure is not None:
            y_failure = y_failure.iloc[sorted_indices]

    # 데이터 분할
    test_size = 0.15
    val_size = 0.15 / (1 - test_size)  # 전체에서 테스트 제외 후 계산

    X_train_val, X_test = train_test_split(X, test_size=test_size, shuffle=False)

    if y_price is not None:
        y_price_train_val, y_price_test = train_test_split(y_price, test_size=test_size, shuffle=False)
        X_price_train, X_price_val, y_price_train, y_price_val = train_test_split(
            X_train_val, y_price_train_val, test_size=val_size, shuffle=False
        )

    if y_failure is not None:
        y_failure_train_val, y_failure_test = train_test_split(y_failure, test_size=test_size, shuffle=False)
        X_failure_train, X_failure_val, y_failure_train, y_failure_val = train_test_split(
            X_train_val, y_failure_train_val, test_size=val_size, shuffle=False
        )

    # 전처리 파이프라인 생성 및 적용
    logger.info("전처리 파이프라인 적용 중...")
    preprocessor = create_preprocessing_pipeline()

    # 전처리 파이프라인 학습
    preprocessor.fit(X_train_val)

    # 데이터 변환
    X_train_transformed = preprocessor.transform(X_train_val)
    X_test_transformed = preprocessor.transform(X_test)

    if y_price is not None:
        logger.info("낙찰가 예측 모델 학습 중...")
        price_model = AuctionPriceModel()
        price_model.train(
            preprocessor.transform(X_price_train),
            y_price_train,
            preprocessor.transform(X_price_val),
            y_price_val
        )

        # 모델 평가
        price_metrics = price_model.evaluate(preprocessor.transform(X_test), y_price_test)
        logger.info(f"낙찰가 예측 모델 성능: {price_metrics}")

        # 모델 저장
        price_model_path = os.path.join(output_dir, 'price_model.joblib')
        price_model.save(price_model_path)
        logger.info(f"낙찰가 예측 모델 저장: {price_model_path}")

    if y_failure is not None:
        logger.info("유찰 확률 예측 모델 학습 중...")
        failure_model = AuctionFailureModel()
        failure_model.train(
            preprocessor.transform(X_failure_train),
            y_failure_train,
            preprocessor.transform(X_failure_val),
            y_failure_val
        )

        # 모델 평가
        failure_metrics = failure_model.evaluate(preprocessor.transform(X_test), y_failure_test)
        logger.info(f"유찰 확률 예측 모델 성능: {failure_metrics}")

        # 모델 저장
        failure_model_path = os.path.join(output_dir, 'failure_model.joblib')
        failure_model.save(failure_model_path)
        logger.info(f"유찰 확률 예측 모델 저장: {failure_model_path}")

    # 예상 낙찰 회차 모델 학습
    # 실제 데이터에서는 낙찰될 때까지의 회차 정보 필요
    if 'failCount' in X.columns and y_price is not None:
        logger.info("예상 낙찰 회차 모델 학습 중...")

        # 낙찰까지의 회차 수를 y로 사용 (현재 failCount + 1)
        y_attempts = X['failCount'] + 1
        y_attempts_train = y_attempts.iloc[:len(X_price_train)]
        y_attempts_val = y_attempts.iloc[len(X_price_train):len(X_price_train)+len(X_price_val)]
        y_attempts_test = y_attempts.iloc[-len(X_test):]

        attempts_model = AuctionAttemptsModel()
        attempts_model.train(
            preprocessor.transform(X_price_train),
            y_attempts_train,
            preprocessor.transform(X_price_val),
            y_attempts_val
        )

        # 모델 평가
        attempts_metrics = attempts_model.evaluate(preprocessor.transform(X_test), y_attempts_test)
        logger.info(f"예상 낙찰 회차 모델 성능: {attempts_metrics}")

        # 모델 저장
        attempts_model_path = os.path.join(output_dir, 'attempts_model.joblib')
        attempts_model.save(attempts_model_path)
        logger.info(f"예상 낙찰 회차 모델 저장: {attempts_model_path}")

    # 전처리 파이프라인 저장
    preprocessor_path = os.path.join(output_dir, 'preprocessor.joblib')
    joblib.dump(preprocessor, preprocessor_path)
    logger.info(f"전처리 파이프라인 저장: {preprocessor_path}")

    logger.info("모델 학습 완료!")
    client.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="경매 예측 모델 학습")
    parser.add_argument('--mongo-uri', type=str, default='mongodb://localhost:27017/', help='MongoDB URI')
    parser.add_argument('--db-name', type=str, default='auction_db', help='데이터베이스 이름')
    parser.add_argument('--collection', type=str, default='auction_data', help='컬렉션 이름')
    parser.add_argument('--output-dir', type=str, default='model_artifacts', help='모델 저장 디렉토리')

    args = parser.parse_args()

    # 저장 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)

    # 모델 학습
    train_models(args.mongo_uri, args.db_name, args.collection, args.output_dir)
