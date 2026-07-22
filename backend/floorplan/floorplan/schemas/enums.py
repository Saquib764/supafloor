"""Domain enumerations."""

from enum import Enum


class RoomType(str, Enum):
    LIVING = "living"
    DINING = "dining"
    KITCHEN = "kitchen"
    BEDROOM = "bedroom"
    MASTER_BEDROOM = "master_bedroom"
    BATHROOM = "bathroom"
    POWDER = "powder"
    HALLWAY = "hallway"
    CLOSET = "closet"
    STORAGE = "storage"
    UTILITY = "utility"
    BALCONY = "balcony"
    ENTRY = "entry"
    STUDY = "study"
    LAUNDRY = "laundry"
    OTHER = "other"


class ZoneType(str, Enum):
    PUBLIC = "public"
    PRIVATE = "private"
    SERVICE = "service"
    CIRCULATION = "circulation"
    OUTDOOR = "outdoor"
    WET = "wet"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
