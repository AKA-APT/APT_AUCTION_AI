import pandas as pd
import numpy as np

def extract_features_from_mongo(mongo_data):
    """
    MongoDB 데이터에서 모델에 필요한 특성 추출
    """
    features = {}

    # 기본 경매 정보
    features['caseId'] = mongo_data.get('csBaseInfo', {}).get('csNo')
    features['courtName'] = mongo_data.get('csBaseInfo', {}).get('cortOfcNm')
    features['caseType'] = mongo_data.get('csBaseInfo', {}).get('csNm')

    # 감정가 및 최저입찰가
    dspsl_gds_info = mongo_data.get('dspslGdsDxdyInfo', {})
    features['appraisalValue'] = dspsl_gds_info.get('aeeEvlAmt')
    features['minimumBidPrice'] = dspsl_gds_info.get('fstPbancLwsDspslPrc')

    # 유찰 횟수 계산
    gdsDspslDxdyLst = mongo_data.get('gdsDspslDxdyLst', [])
    past_auctions = [auction for auction in gdsDspslDxdyLst
                     if auction.get('auctnDxdyRsltCd') is not None]
    features['failCount'] = len(past_auctions)

    # 물건 정보
    property_info = mongo_data.get('gdsDspslObjctLst', [])[0] if mongo_data.get('gdsDspslObjctLst') else {}
    features['propertyType'] = property_info.get('sclDspslGdsLstUsgCd')

    # 위치 정보
    features['state'] = property_info.get('adongSdNm')
    features['city'] = property_info.get('adongSggNm')
    features['dong'] = property_info.get('adongEmdNm')
    features['latitude'] = property_info.get('stYcrd')
    features['longitude'] = property_info.get('stXcrd')

    # 건물 세부 정보
    bld_info = mongo_data.get('bldSdtrDtlLstAll', [[]])[0] if mongo_data.get('bldSdtrDtlLstAll') else []
    building_details = {}
    for bld in bld_info:
        if bld.get('rletDvsDts') == '전유':
            building_details = bld
            break

    # 건물 구조 및 면적 추출
    bld_dtl = building_details.get('bldSdtrDtlDts', '')
    if '철근콘크리트구조' in bld_dtl:
        features['structure'] = '철근콘크리트구조'
    else:
        features['structure'] = '기타'

    # 면적 추출 (숫자만 추출)
    import re
    area_match = re.search(r'(\d+\.\d+)㎡', bld_dtl)
    features['totalArea'] = float(area_match.group(1)) if area_match else None

    # 층 정보
    features['floorInfo'] = property_info.get('bldDtlDts')

    # 점유 정보
    tenants_info = mongo_data.get('dlt_ordTsLserLtn', []) if mongo_data.get('dlt_ordTsLserLtn') else []
    features['isOccupied'] = len(tenants_info) > 0
    features['tenantCount'] = len(tenants_info)

    # 시장 정보
    market_stats = mongo_data.get('aroundDspslStats', [{}])[0] if mongo_data.get('aroundDspslStats') else {}
    features['avgAuctionPriceRate3m'] = market_stats.get('term3MgakPrcRate')
    features['avgAuctionPriceRate12m'] = market_stats.get('term12MgakPrcRate')
    features['avgFailCount3m'] = market_stats.get('term3AvgFlbdNcnt')

    return features

def create_train_dataset_from_mongo(mongo_collection):
    """
    MongoDB 컬렉션에서 학습 데이터셋 생성
    """
    features_list = []

    # mongo_collection이 리스트인지 MongoDB 컬렉션인지 확인
    if isinstance(mongo_collection, list):
        documents = mongo_collection
    else:
        documents = mongo_collection.find()

    for doc in documents:
        # 완료된 경매 건만 선택 (낙찰가가 있는 경우)
        completed_auctions = [auction for auction in doc.get('gdsDspslDxdyLst', [])
                             if auction.get('auctnDxdyRsltCd') == '002' and auction.get('dspslAmt')]

        if completed_auctions:
            # 특성 추출
            features = extract_features_from_mongo(doc)

            # 타겟 변수 (낙찰가) 추가
            final_auction = completed_auctions[-1]  # 마지막 낙찰 정보
            features['actualPrice'] = final_auction.get('dspslAmt')

            # 유찰 여부 타겟 (이전 회차 데이터로 학습)
            past_auctions = doc.get('gdsDspslDxdyLst', [])
            for i in range(len(past_auctions) - 1):
                auction = past_auctions[i]
                if auction.get('auctnDxdyRsltCd'):  # 결과가 있는 경우
                    auction_features = features.copy()
                    auction_features['failCount'] = i
                    # 낙찰=0, 유찰=1로 인코딩
                    auction_features['wasFailure'] = 1 if auction.get('auctnDxdyRsltCd') == '001' else 0
                    features_list.append(auction_features)

    return pd.DataFrame(features_list)

def calculate_optimal_bid(predicted_price, failure_prob, input_data):
    """
    최적 입찰가 계산
    """
    # 기본 로직: 유찰 확률이 높을수록 낮은 입찰가 제안
    min_bid = input_data['minimumBidPrice'].values[0]

    # 유찰 확률에 따른 조정
    if failure_prob > 0.7:  # 유찰 확률이 매우 높은 경우
        return min_bid * 1.03  # 최저가보다 3% 높게
    elif failure_prob > 0.4:  # 유찰 확률이 중간인 경우
        return min(predicted_price * 0.95, min_bid * 1.08)  # 예상가의 95% 또는 최저가 + 8%
    else:  # 유찰 확률이 낮은 경우 (경쟁 예상)
        return min(predicted_price * 0.98, min_bid * 1.15)  # 예상가의 98% 또는 최저가 + 15%

def analyze_risk_factors(input_data, predicted_price, failure_prob):
    """
    위험 요소 분석
    """
    risk_factors = []

    # 감정가 대비 낙찰가 비율 계산
    appraisal_value = input_data['appraisalValue'].values[0]
    price_ratio = predicted_price / appraisal_value

    # 임차인 점유 관련 위험
    if input_data['isOccupied'].values[0]:
        tenant_count = input_data.get('tenantCount', [0]).values[0]
        if tenant_count > 1:
            risk_factors.append(f"다수 임차인 점유 중 ({tenant_count}명)")
        else:
            risk_factors.append("임차인 점유 중")

    # 유찰 횟수 관련 위험
    fail_count = input_data['failCount'].values[0]
    avg_fail_count = input_data.get('avgFailCount3m', [0]).values[0]
    if fail_count > avg_fail_count + 2:
        risk_factors.append(f"유찰 횟수 높음 ({fail_count}회)")

    # 낙찰가율 관련 위험
    if price_ratio < 0.6:
        risk_factors.append("감정가 대비 낙찰가 비율 낮음 (60% 미만)")
    elif price_ratio > 0.9:
        risk_factors.append("감정가에 근접한 낙찰가 예상 (투자수익 저하)")

    # 유찰 확률 관련 위험
    if failure_prob > 0.7:
        risk_factors.append("유찰 가능성 높음")

    # 위험 요소가 없는 경우
    if not risk_factors:
        risk_factors.append("특별한 위험 요소 없음")

    return risk_factors

def determine_risk_level(risk_factors, failure_prob, price_ratio):
    """
    위험 수준 결정
    """
    # 위험 요소 개수
    risk_count = len([factor for factor in risk_factors if "특별한 위험 요소 없음" not in factor])

    # 고위험 조건
    if risk_count >= 3 or failure_prob > 0.8 or price_ratio > 0.95:
        return "HIGH"
    # 중간 위험 조건
    elif risk_count >= 1 or failure_prob > 0.5 or price_ratio > 0.85:
        return "MEDIUM"
    # 저위험 조건
    else:
        return "LOW"
