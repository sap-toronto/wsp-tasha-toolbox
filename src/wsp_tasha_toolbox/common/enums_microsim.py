from enum import StrEnum


class Activity(StrEnum):
    PRIMARY_WORK = "PrimaryWork"
    SECONDARY_WORK = "SecondaryWork"
    WORK_BASED_BUSINESS = "WorkBasedBusiness"
    SCHOOL = "School"
    MARKET = "Market"
    JOINT_MARKET = "JointMarket"
    INDIVIDUAL_OTHER = "IndividualOther"
    JOINT_OTHER = "JointOther"
    HOME = "Home"
    RETURN_FROM_WORK = "ReturnFromWork"


class Mode(StrEnum):
    AUTO = "Auto"
    CARPOOL = "Carpool"
    SCHOOLBUS = "Schoolbus"
    DRIVE_ACCESS_TRANSIT = "DAT"
    WALK_ACCESS_TRANSIT = "WAT"
    WALK = "Walk"
    BIKE = "Bicycle"
    PASSENGER = "Passenger"
    RIDESHARE = "RideShare"
    PASSENGER_ACCESS_TRANSIT = "PAT"
    PASSENGER_EGRESS_TRANSIT = "PET"
    VEHICLE_FOR_HIRE = "VFH"


class Direction(StrEnum):
    AUTO_TO_TRANSIT = "auto2transit"
    TRANSIT_TO_AUTO = "transit2auto"
