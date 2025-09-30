__all__ = [
    "CarSufficiency",
    "OccEmp",
    "StudentClass",
    "TimePeriod",
]

from enum import IntEnum, StrEnum


class CarSufficiency(IntEnum):
    NO_CAR = 1
    """No car household"""
    CAR_LIGHT = 2
    """Number of cars less than number of drivers in household"""
    CAR_NEUTRAL = 3
    """Number of cars greater than or equal to number of drivers in houseold"""


class OccEmp(StrEnum):
    GF = "GF"
    """General Office, Full-time"""
    GP = "GP"
    """General Office, Part-time"""
    MF = "MF"
    """Manufacturing, Full-time"""
    MP = "MP"
    """Manufacturing, Part-time"""
    PF = "PF"
    """Professional, Full-time"""
    PP = "PP"
    """Professional, Part-time"""
    SF = "SF"
    """Retail Sales and Service, Full-time"""
    SP = "SP"
    """Retail Sales and Service, Part-time"""
    NOT_EMPLOYED = "O"
    """Not employed"""
    UNKNOWN = "9"
    """Unknown occupation-employment status"""


class StudentClass(StrEnum):
    PRIMARY = "P"
    SECONDARY = "S"
    POST_SECONDARY = "U"
    NOT_A_STUDENT = "O"
    UNKNOWN = "9"


class TimePeriod(IntEnum):
    AM_PEAK = 1
    MIDDAY = 2
    PM_PEAK = 3
    EVENING = 4
    OVERNIGHT = 5
