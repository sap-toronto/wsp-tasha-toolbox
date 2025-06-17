from enum import StrEnum


class Activity(StrEnum):
    PRIMARY_WORK = "PrimaryWork"  # The main work activity episode going to the assigned work location for the individual
    SECONDARY_WORK = "SecondaryWork"  # A work activity episode not going to the primary work location
    WORK_BASED_BUSINESS = "WorkBasedBusiness"  # A work location not going to the main work location, such as a business meeting
    SCHOOL = "School"  # A school activity episode
    MARKET = "Market"  # A market activity episode
    JOINT_MARKET = "JointMarket"  # A market activity episode that is part of a joint tour
    INDIVIDUAL_OTHER = "IndividualOther"  # An activity episode that is not work, school, or market
    JOINT_OTHER = "JointOther"  # An activity episode that is not work, school, or market and is part of a joint tour
    HOME = "Home"  # The home activity episode
    RETURN_FROM_WORK = "ReturnFromWork"  # A purposeful return home activity episode from work, such as lunch


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
