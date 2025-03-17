import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
import joblib
import os
import logging
import argparse
from datetime import datetime
import json

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

def get_training_metadata(output_dir):
    """
    이전 학습 메타데이터를 로드합니다.
    """
    metadata_path = os.path.join(output_dir, 'training_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return {
        "last_trained_date": None,
        "trained_cases": {},
        "model_version": "0.1.0"
    }

def save_training_metadata(metadata, output_dir):
    """
    학습 메타데이터를 저장합니다.
    """
    metadata_path = os.path.join(output_dir, 'training_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)
    logger.info(f"학습 메타데이터 저장 완료: {metadata_path}")

def get_case_signature(case_data):
    """
    사건 데이터의 고유 시그니처를 생성합니다.
    유찰 회차와 최근 공고 정보를 포함하여 변경 여부를 판단할 수 있게 합니다.
    """
    # 경매 회차 정보 추출
    auction_rounds = []
    if 'gdsDspslDxdyLst' in case_data:
        for auction in case_data['gdsDspslDxdyLst']:
            if auction.get('dxdyYmd') and auction.get('tsLwsDspslPrc') is not None:
                auction_rounds.append({
                    'date': auction['dxdyYmd'],
                    'price': auction['tsLwsDspslPrc'],
                    'result': auction.get('auctnDxdyRsltCd')
                })

    # 유찰 횟수 정보
    flbd_ncnt = case_data.get('dspslGdsDxdyInfo', {}).get('flbdNcnt', 0)

    # 시그니처 생성
    signature = {
        'csNo': case_data.get('csBaseInfo', {}).get('csNo'),
        'auction_rounds_count': len(auction_rounds),
        'last_auction_date': auction_rounds[-1]['date'] if auction_rounds else None,
        'last_auction_price': auction_rounds[-1]['price'] if auction_rounds else None,
        'flbdNcnt': flbd_ncnt
    }

    return json.dumps(signature, sort_keys=True)

def is_case_updated(case_data, trained_cases):
    """
    사건이 업데이트되었는지 확인합니다.
    """
    cs_no = case_data.get('csBaseInfo', {}).get('csNo')
    if cs_no not in trained_cases:
        return True  # 새로운 사건

    current_signature = get_case_signature(case_data)
    previous_signature = trained_cases[cs_no]

    return current_signature != previous_signature

def filter_updated_cases(collection, trained_cases):
    """
    업데이트된 사건만 필터링합니다.
    """
    updated_cases = []
    case_signatures = {}

    # 모든 사건 조회
    for case in collection.find({}):
        cs_no = case.get('csBaseInfo', {}).get('csNo')
        if cs_no and is_case_updated(case, trained_cases):
            updated_cases.append(case)
            case_signatures[cs_no] = get_case_signature(case)

    logger.info(f"전체 사건 수: {collection.count_documents({})}, 업데이트된 사건 수: {len(updated_cases)}")
    return updated_cases, case_signatures

def incremental_train_models(mongo_uri, db_name, collection_name, output_dir='model_artifacts'):
    """
    MongoDB 데이터로부터 증분 학습을 수행합니다.
    """
    # 학습 메타데이터 로드
    metadata = get_training_metadata(output_dir)
    trained_cases = metadata.get("trained_cases", {})

    logger.info("MongoDB 연결 중...")
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]

    # 업데이트된 사건만 필터링
    updated_cases, new_case_signatures = filter_updated_cases(collection, trained_cases)

    if not updated_cases:
        logger.info("업데이트된 사건이 없습니다. 학습을 종료합니다.")
        client.close()
        return

    logger.info(f"증분 학습을 위한 사건 수: {len(updated_cases)}")

    # 업데이트된 사건으로 학습 데이터셋 생성
    logger.info("증분 학습 데이터셋 생성 중...")
    df = create_train_dataset_from_mongo(updated_cases)
    logger.info(f"증분 데이터셋 크기: {df.shape}")

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

    # 전처리 파이프라인 로드 또는 생성
    preprocessor_path = os.path.join(output_dir, 'preprocessor.joblib')
    if os.path.exists(preprocessor_path):
        logger.info("기존 전처리 파이프라인 로드 중...")
        preprocessor = joblib.load(preprocessor_path)
    else:
        logger.info("새로운 전처리 파이프라인 생성 중...")
        preprocessor = create_preprocessing_pipeline()
        preprocessor.fit(X_train_val)

    # 데이터 변환
    X_train_transformed = preprocessor.transform(X_train_val)
    X_test_transformed = preprocessor.transform(X_test)

    # 낙찰가 예측 모델 증분 학습
    if y_price is not None:
        price_model_path = os.path.join(output_dir, 'price_model.joblib')

        if os.path.exists(price_model_path):
            logger.info("기존 낙찰가 예측 모델 로드 및 증분 학습 중...")
            price_model = joblib.load(price_model_path)
            price_model.update_model(
                preprocessor.transform(X_price_train),
                y_price_train,
                preprocessor.transform(X_price_val),
                y_price_val
            )
        else:
            logger.info("새로운 낙찰가 예측 모델 학습 중...")
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
        price_model.save(price_model_path)
        logger.info(f"낙찰가 예측 모델 저장: {price_model_path}")

    # 유찰 확률 예측 모델 증분 학습
    if y_failure is not None:
        failure_model_path = os.path.join(output_dir, 'failure_model.joblib')

        if os.path.exists(failure_model_path):
            logger.info("기존 유찰 확률 예측 모델 로드 및 증분 학습 중...")
            failure_model = joblib.load(failure_model_path)
            failure_model.update_model(
                preprocessor.transform(X_failure_train),
                y_failure_train,
                preprocessor.transform(X_failure_val),
                y_failure_val
            )
        else:
            logger.info("새로운 유찰 확률 예측 모델 학습 중...")
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
        failure_model.save(failure_model_path)
        logger.info(f"유찰 확률 예측 모델 저장: {failure_model_path}")

    # 예상 낙찰 회차 모델 증분 학습
    if 'failCount' in X.columns and y_price is not None:
        attempts_model_path = os.path.join(output_dir, 'attempts_model.joblib')

        # 낙찰까지의 회차 수를 y로 사용 (현재 failCount + 1)
        y_attempts = X['failCount'] + 1
        y_attempts_train = y_attempts.iloc[:len(X_price_train)]
        y_attempts_val = y_attempts.iloc[len(X_price_train):len(X_price_train)+len(X_price_val)]
        y_attempts_test = y_attempts.iloc[-len(X_test):]

        if os.path.exists(attempts_model_path):
            logger.info("기존 예상 낙찰 회차 모델 로드 및 증분 학습 중...")
            attempts_model = joblib.load(attempts_model_path)
            attempts_model.update_model(
                preprocessor.transform(X_price_train),
                y_attempts_train,
                preprocessor.transform(X_price_val),
                y_attempts_val
            )
        else:
            logger.info("새로운 예상 낙찰 회차 모델 학습 중...")
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
        attempts_model.save(attempts_model_path)
        logger.info(f"예상 낙찰 회차 모델 저장: {attempts_model_path}")

    # 전처리 파이프라인 저장
    joblib.dump(preprocessor, preprocessor_path)
    logger.info(f"전처리 파이프라인 저장: {preprocessor_path}")

    # 학습 메타데이터 업데이트
    metadata["last_trained_date"] = datetime.now().isoformat()
    metadata["trained_cases"].update(new_case_signatures)
    save_training_metadata(metadata, output_dir)

    logger.info("증분 학습 완료!")
    client.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="경매 예측 모델 증분 학습")
    parser.add_argument('--mongo-uri', type=str, default='mongodb://localhost:27017/', help='MongoDB URI')
    parser.add_argument('--db-name', type=str, default='auction_db', help='데이터베이스 이름')
    parser.add_argument('--collection', type=str, default='auction_data', help='컬렉션 이름')
    parser.add_argument('--output-dir', type=str, default='model_artifacts', help='모델 저장 디렉토리')

    args = parser.parse_args()

    # 저장 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)

    # 증분 학습 실행
    incremental_train_models(args.mongo_uri, args.db_name, args.collection, args.output_dir)

