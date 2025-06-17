from enum import IntEnum, StrEnum


class CarSufficiency(IntEnum):
    NO_CAR = 1
    CAR_LIGHT = 2  # Cars < Adults
    CAR_NEUTRAL = 3  # Cars >= Adults


class TimePeriod(IntEnum):
    AM_PEAK = 1
    MIDDAY = 2
    PM_PEAK = 3
    EVENING = 4
    OVERNIGHT = 5
