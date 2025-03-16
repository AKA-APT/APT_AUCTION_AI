from pydantic import BaseModel, Field
from typing import Optional, List

class Location(BaseModel):
    state: str
    city: str
    dong: str
    latitude: float
    longitude: float

class BuildingDetails(BaseModel):
    structure: str
    totalArea: float
    floorInfo: str

class OccupancyInfo(BaseModel):
    isOccupied: bool
    tenantCount: Optional[int] = 0

class PropertyInfo(BaseModel):
    propertyType: str
    location: Location
    buildingDetails: BuildingDetails
    occupancyInfo: OccupancyInfo

class AuctionInfo(BaseModel):
    caseId: str
    courtName: str
    caseType: str
    appraisalValue: int
    minimumBidPrice: int
    failCount: int = 0

class MarketInfo(BaseModel):
    avgAuctionPriceRate3m: Optional[float] = None
    avgAuctionPriceRate12m: Optional[float] = None
    avgFailCount3m: Optional[float] = None

class AuctionAnalysisRequest(BaseModel):
    auctionInfo: AuctionInfo
    propertyInfo: PropertyInfo
    marketInfo: Optional[MarketInfo] = None

    class Config:
        schema_extra = {
            "example": {
                "auctionInfo": {
                    "caseId": "20200130101073",
                    "courtName": "서울중앙지방법원",
                    "caseType": "부동산강제경매",
                    "appraisalValue": 424000000,
                    "minimumBidPrice": 424000000,
                    "failCount": 1
                },
                "propertyInfo": {
                    "propertyType": "20111",
                    "location": {
                        "state": "서울특별시",
                        "city": "서초구",
                        "dong": "양재동",
                        "latitude": 37.4840221129458,
                        "longitude": 127.036695192276
                    },
                    "buildingDetails": {
                        "structure": "철근콘크리트구조",
                        "totalArea": 96.43,
                        "floorInfo": "지층비101호"
                    },
                    "occupancyInfo": {
                        "isOccupied": True,
                        "tenantCount": 1
                    }
                },
                "marketInfo": {
                    "avgAuctionPriceRate3m": 46,
                    "avgAuctionPriceRate12m": 57,
                    "avgFailCount3m": 4.6
                }
            }
        }
