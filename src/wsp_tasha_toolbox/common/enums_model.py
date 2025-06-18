from enum import IntEnum, StrEnum


class CarSufficiency(IntEnum):
    NO_CAR = 1
    CAR_LIGHT = 2  # Cars < Adults
    CAR_NEUTRAL = 3  # Cars >= Adults


class OccEmp(StrEnum):
    GENERAL_FT = "GF"
    GENERAL_PT = "GP"
    MANUFACTURING_FT = "MF"
    MANUFACTURING_PT = "MP"
    PROFESSIONAL_FT = "PF"
    PROFESSIONAL_PT = "PP"
    SERVICE_FT = "SF"
    SERVICE_PT = "SP"
    NOT_EMPLOYED = "O"
    UNKNOWN = "9"


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
