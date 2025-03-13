from dataclasses import dataclass
from typing import Dict, Any, Optional
from datetime import datetime


@dataclass
class Coordinates:
    latitude: float
    longitude: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Coordinates':
        return cls(
            latitude=data.get('latitude'),
            longitude=data.get('longitude')
        )


@dataclass
class BasicInfo:
    court_name: str
    case_number: str
    case_type: str
    appraised_value: float
    claim_amount: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BasicInfo':
        return cls(
            court_name=data.get('courtName'),
            case_number=data.get('caseNumber'),
            case_type=data.get('caseType'),
            appraised_value=float(data.get('appraisedValue')),
            claim_amount=float(data.get('claimAmount'))
        )


@dataclass
class PropertyInfo:
    property_type: str
    building_structure: str
    size: float
    address: str
    coordinates: Coordinates

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PropertyInfo':
        return cls(
            property_type=data.get('propertyType'),
            building_structure=data.get('buildingStructure'),
            size=float(data.get('size')),
            address=data.get('address'),
            coordinates=Coordinates.from_dict(data.get('coordinates', {}))
        )


@dataclass
class DateInfo:
    received_date: str
    decision_date: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DateInfo':
        return cls(
            received_date=data.get('receivedDate'),
            decision_date=data.get('decisionDate')
        )


@dataclass
class AuctionAnalysisRequest:
    basic_info: BasicInfo
    property_info: PropertyInfo
    date_info: DateInfo

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuctionAnalysisRequest':
        return cls(
            basic_info=BasicInfo.from_dict(data.get('basicInfo', {})),
            property_info=PropertyInfo.from_dict(data.get('propertyInfo', {})),
            date_info=DateInfo.from_dict(data.get('dateInfo', {}))
        )
