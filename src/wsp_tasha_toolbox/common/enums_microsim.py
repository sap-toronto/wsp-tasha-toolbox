__all__ = [
    "Activity",
    "Mode",
    "TransitDirection",
]

from enum import StrEnum


class Activity(StrEnum):
    PRIMARY_WORK = "PrimaryWork"
    """The main work activity episode going to the assigned work location for the individual"""
    SECONDARY_WORK = "SecondaryWork"
    """A work activity episode not going to the primary work location"""
    WORK_BASED_BUSINESS = "WorkBasedBusiness"
    """A work location not going to the main work location, such as a business meeting"""
    SCHOOL = "School"
    """A school activity episode"""
    MARKET = "Market"
    """A market activity episode"""
    JOINT_MARKET = "JointMarket"
    """A market activity episode that is part of a joint tour"""
    INDIVIDUAL_OTHER = "IndividualOther"
    """An activity episode that is not work, school, or market"""
    JOINT_OTHER = "JointOther"
    """An activity episode that is not work, school, or market and is part of a joint tour"""
    HOME = "Home"
    """The home activity episode"""
    RETURN_FROM_WORK = "ReturnFromWork"
    """A purposeful return home activity episode from work, such as lunch"""


class Mode(StrEnum):
    AUTO = "Auto"
    """The auto mode represents when a person is the driver inside of a vehicle (be is car, motorcycle or any other personally owned vehicle) that runs on the auto network"""
    RIDESHARE = "RideShare"
    """A dummy mode for persons who are not in charge of their joint trip chain. If the tour-leader chooses to take Auto Drive then this mode will be assigned to the passenger. The passenger will share the same activities as the tour-leader."""
    VEHICLE_FOR_HIRE = "VFH"
    """The mode that serves as a grouping for Taxi and Uber, Lift, and other Vehicle for Hire services"""
    PASSENGER = "Passenger"
    """A dummy mode for persons in a vehicle where the driver is only facilitating the passengers' movements. The driver does not share the same activities as the passenger (e.g., a school kid being driven as a passenger to school; the driver faciliates the kid's activity but doesn't participate in it)"""
    CARPOOL = "Carpool"
    """The carpool mode representing inter-household passenger trips"""
    WALK_ACCESS_TRANSIT = "WAT"
    """The mode that represents trips where both the access and egress legs of the trip are done using active transportation modes, such as walking or cycling, and where at least one public transit vehicle is used for the trip. In pre-V4.1 versions of GTAModel, this mode also represents transit trips for passengers being dropped off at a station/stop."""
    DRIVE_ACCESS_TRANSIT = "DAT"
    """The mode that represents trips where a person drives to a transit station, parks their vehicle, and boards transit."""
    PASSENGER_ACCESS_TRANSIT = "PAT"
    """Available in GTAModel V4.1+, this mode serves as an auto passenger trip to transit stations that have parking lots. This mode does not require the facilitation by a household member, nor the allocation of a household auto."""
    PASSENGER_EGRESS_TRANSIT = "PET"
    """Available in GTAModel V4.1+, this mode serves as an auto passenger trip from transit stations that have parking lots. This mode does not require the facilitation by a household member, nor the allocation of a household auto."""
    WALK = "Walk"
    """The simple mode that represents walking"""
    BIKE = "Bicycle"
    """The simple mode that represents bicycling"""
    SCHOOLBUS = "Schoolbus"
    """The simple mode that represents transport for school activities by school buses"""


class TransitDirection(StrEnum):
    AUTO_TO_TRANSIT = "auto2transit"
    TRANSIT_TO_AUTO = "transit2auto"
