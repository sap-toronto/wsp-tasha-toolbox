from __future__ import annotations

from typing import Optional

import numpy as np
import pandera.pandas as pa
from pandera.typing.pandas import Category, Index, Series

from ..common.enums_microsim import Activity, Direction, Mode
from ..common.enums_tts2016 import (DwellingType, EmploymentStatus, Gender,
                                    IncomeClass, Occupation, StudentStatus)


class HouseholdsSchema(pa.DataFrameModel):
    household_id: Index[np.uint64] = pa.Field(ge=1, coerce=True, check_name=True)
    home_zone: Series[np.int16] = pa.Field(ge=0, coerce=True)
    weight: Series[np.int64] = pa.Field(ge=1, coerce=True)
    persons: Series[np.int16] = pa.Field(ge=1, coerce=True)
    dwelling_type: Series[np.int16] = pa.Field(isin=DwellingType, coerce=True)
    vehicles: Series[np.int16] = pa.Field(ge=0, coerce=True)
    income_class: Series[np.int16] = pa.Field(isin=IncomeClass, coerce=True)


class PersonsSchema(pa.DataFrameModel):
    household_id: Index[np.uint64] = pa.Field(ge=1, coerce=True, check_name=True)
    person_id: Index[np.uint64] = pa.Field(ge=1, coerce=True, check_name=True)
    age: Series[np.int16] = pa.Field(ge=0, le=100, coerce=True)
    sex: Series[Category] = pa.Field(isin=Gender, coerce=True)
    license: Series[bool] = pa.Field(coerce=True)
    transit_pass: Series[bool] = pa.Field(coerce=True)
    employment_status: Series[Category] = pa.Field(isin=EmploymentStatus, coerce=True)
    occupation: Series[Category] = pa.Field(isin=Occupation, coerce=True)
    free_parking: Series[bool] = pa.Field(coerce=True)
    student_status: Series[Category] = pa.Field(isin=StudentStatus, coerce=True)
    work_zone: Series[np.int16] = pa.Field(ge=0, coerce=True)
    school_zone: Series[np.int16] = pa.Field(ge=0, coerce=True)
    weight: Series[np.int64] = pa.Field(ge=1, coerce=True)
    telecommuting: Optional[Series[bool]] = pa.Field(coerce=True)


class TripsSchema(pa.DataFrameModel):
    household_id: Index[np.uint64] = pa.Field(ge=1, coerce=True, check_name=True)
    person_id: Index[np.uint64] = pa.Field(ge=1, coerce=True, check_name=True)
    trip_id: Index[np.uint16] = pa.Field(ge=1, coerce=True, check_name=True)
    o_act: Series[Category] = pa.Field(isin=Activity, coerce=True)
    o_zone: Series[np.int16] = pa.Field(ge=0, coerce=True)
    d_act: Series[Category] = pa.Field(isin=Activity, coerce=True)
    d_zone: Series[np.int16] = pa.Field(ge=0, coerce=True)
    weight: Series[np.int64] = pa.Field(ge=1, coerce=True)
    JointTourRep: Optional[Series[np.int64]] = pa.Field(coerce=True)
    JointTourRepTripId: Optional[Series[np.int64]] = pa.Field(coerce=True)


class TripModesSchema(pa.DataFrameModel):
    household_id: Index[np.uint64] = pa.Field(ge=1, coerce=True, check_name=True)
    person_id: Index[np.uint64] = pa.Field(ge=1, coerce=True, check_name=True)
    trip_id: Index[np.uint16] = pa.Field(ge=1, coerce=True, check_name=True)
    mode: Series[Category] = pa.Field(isin=Mode, coerce=True)
    o_depart: Series[np.float64] = pa.Field(ge=0, coerce=True)
    d_arrive: Series[np.float64] = pa.Field(ge=0, coerce=True)
    weight: Series[np.int64] = pa.Field(ge=1, coerce=True)


class TripStationsSchema(pa.DataFrameModel):
    household_id: Index[np.uint64] = pa.Field(ge=1, coerce=True, check_name=True)
    person_id: Index[np.uint64] = pa.Field(ge=1, coerce=True, check_name=True)
    trip_id: Index[np.uint16] = pa.Field(ge=1, coerce=True, check_name=True)
    station: Series[np.int16] = pa.Field(ge=0, coerce=True)
    direction: Series[Category] = pa.Field(isin=Direction, coerce=True)
    weight: Series[np.int64] = pa.Field(ge=1, coerce=True)
    mode: Optional[Series[Category]] = pa.Field(isin=Mode, coerce=True)


class FacilitatePassengersSchema(pa.DataFrameModel):
    household_id: Index[np.uint64] = pa.Field(ge=1, coerce=True, check_name=True)
    passenger_id: Index[np.uint64] = pa.Field(ge=1, coerce=True, check_name=True)
    passenger_trip_id: Index[np.uint16] = pa.Field(ge=1, coerce=True, check_name=True)
    driver_id: Series[np.uint64] = pa.Field(ge=1, coerce=True, check_name=True)
    driver_trip_id: Series[np.int16] = pa.Field(coerce=True, check_name=True)
    weight: Series[np.int64] = pa.Field(ge=1, coerce=True)
