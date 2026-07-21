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


class MutationOp(str, Enum):
    INSERT_BALCONY = "insert_balcony"
    INSERT_TOILET = "insert_toilet"
    INSERT_CLOSET = "insert_closet"
    INSERT_UTILITY = "insert_utility"
    INSERT_STORAGE = "insert_storage"
    MERGE_ROOMS = "merge_rooms"
    SPLIT_ROOM = "split_room"
    SHIFT_WET_CORE = "shift_wet_core"
