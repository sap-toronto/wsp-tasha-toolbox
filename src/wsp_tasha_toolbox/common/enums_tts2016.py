from enum import IntEnum, StrEnum


class DwellingType(IntEnum):
    HOUSE = 1
    APARTMENT = 2
    TOWNHOUSE = 3


class IncomeClass(IntEnum):
    LTE_14999 = 1
    BTWN_15000_AND_39999 = 2
    BTWN_40000_AND_59999 = 3
    BTWN_60000_AND_99999 = 4
    BTWN_100000_AND_124999 = 5
    GTE_125000 = 6


class Gender(StrEnum):
    FEMALE = "F"
    MALE = "M"


class EmploymentStatus(StrEnum):
    FULL_TIME = "F"
    PART_TIME = "P"
    FULL_TIME_AT_HOME = "H"
    PART_TIME_AT_HOME = "J"
    NOT_EMPLOYED = "O"
    UNKNOWN = "9"


class Occupation(StrEnum):
    GENERAL_OFFICE = "G"
    MANUFACTURING = "M"
    PROFESSIONAL = "P"
    RETAIL_SALES_AND_SERVICE = "S"
    NOT_EMPLOYED = "O"
    UNKNOWN = "9"


class StudentStatus(StrEnum):
    FULL_TIME = "F"
    PART_TIME = "P"
    NOT_A_STUDENT = "O"
    UNKNOWN = "9"


class TripPurpose(StrEnum):
    WORK = "W"
    SUBSEQUENT_WORK = "R"
    SCHOOL = "S"
    SUBSEQUENT_SCHOOL = "C"
    DAYCARE = "D"
    FACILITATE_PASSENGER = "F"
    MARKET = "M"
    ENTERTAINMENT = "E"
    OTHER = "O"
    HOME = "H"
    UNKNOWN = "9"
