from __future__ import annotations

__all__ = [
    "HouseholdsSchema",
    "PersonsSchema",
    "TripsSchema",
    "TripModesSchema",
    "TripStationsSchema",
    "FacilitatePassengersSchema",
]

from typing import Optional

import numpy as np
import pandera.pandas as pa
from pandera.typing.pandas import Category, Index, Series

from ..common.enums_microsim import Activity, Mode, TransitDirection
from ..common.enums_tts2016 import DwellingType, EmploymentStatus, Gender, IncomeClass, Occupation, StudentStatus


class HouseholdsSchema(pa.DataFrameModel):
    household_id: Index[np.uint64] = pa.Field(
        ge=1,
        coerce=True,
        check_name=True,
        description="The unique identifier for the household",
    )
    home_zone: Series[np.int16] = pa.Field(
        ge=0,
        coerce=True,
        description="The zone of residence",
    )
    weight: Series[np.int64] = pa.Field(
        ge=1,
        coerce=True,
        description="The household's expansion factor to scale the agent to the global population",
    )
    persons: Series[np.int16] = pa.Field(
        ge=1,
        coerce=True,
        description="The number of people living within the household",
    )
    dwelling_type: Series[np.int16] = pa.Field(
        isin=DwellingType,
        coerce=True,
        description="The dwelling/structure type the household lives in",
    )
    vehicles: Series[np.int16] = pa.Field(
        ge=0,
        coerce=True,
        description="The number of vehicles assigned to the household from the auto ownership model",
    )
    income_class: Series[np.int16] = pa.Field(
        isin=IncomeClass,
        coerce=True,
        description="The income class according to TTS2016 that the household belongs to",
    )


class PersonsSchema(pa.DataFrameModel):
    household_id: Index[np.uint64] = pa.Field(
        ge=1,
        coerce=True,
        check_name=True,
        description="The unique identifier for the household",
    )
    person_id: Index[np.uint64] = pa.Field(
        ge=1,
        coerce=True,
        check_name=True,
        description="The unique identifier for the person within the household",
    )
    age: Series[np.int16] = pa.Field(
        ge=0,
        le=100,
        coerce=True,
        description="The person's age (in single years)",
    )
    sex: Series[Category] = pa.Field(
        isin=Gender,
        coerce=True,
        description="The person's gender",
    )
    license: Series[bool] = pa.Field(
        coerce=True,
        description="A flag to indicate if the person holds a driver's license",
    )
    transit_pass: Series[bool] = pa.Field(
        coerce=True,
        description="Unused",
    )
    employment_status: Series[Category] = pa.Field(
        isin=EmploymentStatus,
        coerce=True,
        description="The person's work status",
    )
    occupation: Series[Category] = pa.Field(
        isin=Occupation,
        coerce=True,
        description="The person's occupation category",
    )
    free_parking: Series[bool] = pa.Field(
        coerce=True,
        description="Unused",
    )
    student_status: Series[Category] = pa.Field(
        isin=StudentStatus,
        coerce=True,
        description="The person's student status",
    )
    work_zone: Series[np.int16] = pa.Field(
        ge=0,
        coerce=True,
        description="If the zone is 0, then they have not been assigned a work zone",
    )
    school_zone: Series[np.int16] = pa.Field(
        ge=0,
        coerce=True,
        description="If the zone is 0, then they have not been assigned a school zone",
    )
    weight: Series[np.int64] = pa.Field(
        ge=1,
        coerce=True,
        description="The person's expansion factor to scale the agent to the global population",
    )
    telecommuting: Optional[Series[bool]] = pa.Field(
        coerce=True,
        description="A flag to indicate if the person decided to telecommute on the day of simulation",
    )


class TripsSchema(pa.DataFrameModel):
    household_id: Index[np.uint64] = pa.Field(
        ge=1,
        coerce=True,
        check_name=True,
        description="The unique identifier for the household",
    )
    person_id: Index[np.uint64] = pa.Field(
        ge=1,
        coerce=True,
        check_name=True,
        description="The unique identifier for the person within the household",
    )
    trip_id: Index[np.uint16] = pa.Field(
        ge=1,
        coerce=True,
        check_name=True,
        description="The unique identifier for the trip for the given person",
    )
    o_act: Series[Category] = pa.Field(
        isin=Activity,
        coerce=True,
        description="The activity at the origin of the trip",
    )
    o_zone: Series[np.int16] = pa.Field(
        ge=0,
        coerce=True,
        description="The zone number of the trip's origin",
    )
    d_act: Series[Category] = pa.Field(
        isin=Activity,
        coerce=True,
        description="The activity at the destination of the trip",
    )
    d_zone: Series[np.int16] = pa.Field(
        ge=0,
        coerce=True,
        description="The zone number of the trip's destination",
    )
    weight: Series[np.int64] = pa.Field(
        ge=1,
        coerce=True,
        description="The factor to apply to scale this trip to scale the trip for the global population. This will be the same as the person's weight",
    )
    JointTourRep: Optional[Series[np.int64]] = pa.Field(
        coerce=True,
        description="-1 if this is not a joint tour, otherwise the person id of the person (in the same household) who is making the choices.",
    )
    JointTourRepTripId: Optional[Series[np.int64]] = pa.Field(
        coerce=True,
        description="-1 if this is not a joint tour, otherwise the trip id is of the representative's associated trip.",
    )


class TripModesSchema(pa.DataFrameModel):
    household_id: Index[np.uint64] = pa.Field(
        ge=1,
        coerce=True,
        check_name=True,
        description="The unique identifier for the household",
    )
    person_id: Index[np.uint64] = pa.Field(
        ge=1,
        coerce=True,
        check_name=True,
        description="The unique identifier for the person within the household",
    )
    trip_id: Index[np.uint16] = pa.Field(
        ge=1,
        coerce=True,
        check_name=True,
        description="The unique identifier for the trip for the given person",
    )
    mode: Series[Category] = pa.Field(
        isin=Mode,
        coerce=True,
        description="The name of the mode that was used",
    )
    o_depart: Series[np.float64] = pa.Field(
        coerce=True,
        description="The time that the trip started / departed the origin, relative to 12:00 AM",
    )
    d_arrive: Series[np.float64] = pa.Field(
        ge=240,
        coerce=True,
        description="The time that the trip ended / arrived at the destination, relative to 12:00 AM",
    )
    weight: Series[np.int64] = pa.Field(
        ge=1,
        coerce=True,
        description="The number of times that this mode was selected for the given trip",
    )


class TripStationsSchema(pa.DataFrameModel):
    household_id: Index[np.uint64] = pa.Field(
        ge=1,
        coerce=True,
        check_name=True,
        description="The unique identifier for the household",
    )
    person_id: Index[np.uint64] = pa.Field(
        ge=1,
        coerce=True,
        check_name=True,
        description="The unique identifier for the person within the household",
    )
    trip_id: Index[np.uint16] = pa.Field(
        ge=1,
        coerce=True,
        check_name=True,
        description="The unique identifier for the trip for the given person",
    )
    station: Series[np.int16] = pa.Field(
        ge=0,
        coerce=True,
        description="The zone number for the station",
    )
    direction: Series[Category] = pa.Field(
        isin=TransitDirection,
        coerce=True,
        description="The travel direction of the transit trip. Used to determine if station is used as an access or egress point.",
    )
    weight: Series[np.int64] = pa.Field(
        ge=1,
        coerce=True,
        description="The number of times that the station was selected",
    )
    mode: Optional[Series[Category]] = pa.Field(
        isin=Mode,
        coerce=True,
        description="The mode associated with this station choice. This is required for analyzing PAT/PET modes.",
    )


class FacilitatePassengersSchema(pa.DataFrameModel):
    household_id: Index[np.uint64] = pa.Field(
        ge=1,
        coerce=True,
        check_name=True,
        description="The unique identifier for the household",
    )
    passenger_id: Index[np.uint64] = pa.Field(
        ge=1,
        coerce=True,
        check_name=True,
        description="The unique identifier for the passenger within the household",
    )
    passenger_trip_id: Index[np.uint16] = pa.Field(
        ge=1,
        coerce=True,
        check_name=True,
        description="The unique identifier for the trip for the passenger that is being facilitated",
    )
    driver_id: Series[np.uint64] = pa.Field(
        ge=1,
        coerce=True,
        check_name=True,
        description="The unique identifier for the driver within the household",
    )
    driver_trip_id: Series[np.int16] = pa.Field(
        coerce=True,
        check_name=True,
        description="The unique identifier for the trip that the driver is being diverted from in order to facilitate the trip. -1 if the driver is facilitating the trip from home",
    )
    weight: Series[np.int64] = pa.Field(
        ge=1,
        coerce=True,
        description="The number of times the passenger's trip was facilitated in this manner",
    )
