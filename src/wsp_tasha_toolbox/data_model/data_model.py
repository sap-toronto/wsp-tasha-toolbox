from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Literal, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from wsp_balsa.logging import ModelLogger, get_model_logger
from wsp_balsa.routines import distance_matrix, read_mdf

from ..common.activity_pairs import activity_pair_mapping
from ..common.enums_data import TimeFormat
from ..common.enums_model import OccEmp, StudentClass
from ..common.enums_tts2016 import (EmploymentStatus, Occupation,
                                    StudentStatus, TripPurpose)
from . import schema as ms


class MicrosimData:
    """Data model for working with microsimulation outputs from a TASHA-based model

    Args:
        households (pd.DataFrame): The household table
        persons (pd.DataFrame): The person table
        trips (pd.DataFrame): The trips table
        trip_modes (pd.DataFrame): The trip modes table
        trip_stations (pd.DataFrame): The trip stations table
        facilitate_passengers (pd.DataFrame, optional): Defaults to ``None``. The facilitate passengers table
        reweight_trips (bool, optional): Defaults to ``True``. A flag to reweight trip modes, trip station, and
            facilitate passenger (if available) tables to match total number of trips modelled in the trip table.
        derive_additional_attributes (bool, optional): Defaults to ``True``. A flag to calculate additional attributes
            useful in model analyses
    """

    time_period_bins = (0, 6, 9, 15, 19, 24, np.inf)
    student_age_range_primary = (5, 13)
    student_age_range_secondary = (14, 17)
    student_age_range_post_secondary = (18, np.inf)

    def __init__(
        self,
        households: pd.DataFrame,
        persons: pd.DataFrame,
        trips: pd.DataFrame,
        trip_modes: pd.DataFrame,
        trip_stations: pd.DataFrame,
        *,
        facilitate_passengers: pd.DataFrame = None,
        reweight_trips: bool = True,
        derive_additional_attributes: bool = True,
    ) -> None:
        self.logger.tip("Loading microsim tables")

        self._households = households
        self._persons = persons
        self._trips = trips
        self._trip_modes = trip_modes
        self._trip_stations = trip_stations
        self._facilitate_passengers = facilitate_passengers
        self._zones: Optional[gpd.GeoDataFrame] = None
        self._impedances: Optional[pd.DataFrame] = None

        if reweight_trips:
            self._adjust_trip_weights()

        if derive_additional_attributes:
            self._classify_times()
            self._derive_household_attributes()
            self._derive_person_attributes()
            self._derive_trip_attributes()

        self.logger.report("Microsim tables successfully loaded!")

    # region Properties

    @property
    def logger(self) -> ModelLogger:
        return get_model_logger(f"wsp_tasha_toolbox.{self.__class__.__name__}")

    @property
    def households(self) -> pd.DataFrame:
        """Modelled microsim households table"""
        return self._households

    @property
    def persons(self) -> pd.DataFrame:
        """Modelled microsim persons table"""
        return self._persons

    @property
    def trips(self) -> pd.DataFrame:
        """Modelled microsim trips table"""
        return self._trips

    @property
    def trip_modes(self) -> pd.DataFrame:
        """Modelled microsim trip modes table"""
        return self._trip_modes

    @property
    def trip_stations(self) -> pd.DataFrame:
        """Modelled microsim trip stations table"""
        return self._trip_stations

    @property
    def facilitate_passengers(self) -> Optional[pd.DataFrame]:
        """Modelled microsim facilitate passengers table"""
        return self._facilitate_passengers

    @property
    def zones(self) -> Optional[gpd.GeoDataFrame]:
        """Model zone system table"""
        return self._zones

    @property
    def impedances(self) -> pd.DataFrame:
        return self._impedances

    # endregion

    # region Loading

    @classmethod
    def from_result_folder(
        cls,
        results_folder: PathLike | str,
        zone_shapefile: PathLike | str,
        *,
        rebuild_indices: bool = True,
        sort_indices: bool = True,
        zone_label: str = "taz",
        coord_unit: float = 0.001,
    ) -> MicrosimData:
        """Initialize a new instance of MicrosimData using files from a model run result folder

        Args:
            results_folder (PathLike | str): Path to the model run result folder
            zone_shapefile (PathLike | str): Path to the zone shapefile to use with the model run
            rebuild_indices (bool, optional): Defaults to ``True``. A flag to rebuild indices in the microsim tables.
            sort_indices (bool, optional): Defaults to ``True``. A flag to sort indices in the microsim tables.
            zone_label (str, optional): Defaults to ``"taz"``. The name of the traffic analysis zone (TAZ) attribute in
                the zone shapefile.
            coord_unit (float, optional): Defaults to ``0.001``. A value to adjust distance values with.
        """
        microsim_folder = Path(results_folder) / "Microsim Results"

        # Load data
        households = cls._load_households(microsim_folder / "households.csv")
        persons = cls._load_persons(microsim_folder / "persons.csv", rebuild_index=rebuild_indices)
        trips = cls._load_trips(microsim_folder / "trips.csv", rebuild_index=rebuild_indices)
        trip_modes = cls._load_trips(microsim_folder / "trip_modes.csv", rebuild_index=rebuild_indices)
        trip_stations = cls._load_trip_stations(microsim_folder / "trip_stations.csv", rebuild_index=rebuild_indices)
        facilitate_passengers_fp = microsim_folder / "facilitate_passenger.csv"
        if facilitate_passengers_fp.exists():
            facilitate_passengers = cls._load_facilitate_passengers(
                facilitate_passengers_fp, rebuild_index=rebuild_indices
            )
        else:
            facilitate_passengers = None

        # Reindex persons
        if rebuild_indices:
            cls._rebuild_microsim_indices(
                households, persons, trips, trip_modes, trip_stations, facilitate_passengers=facilitate_passengers
            )

        # Sort indices
        if sort_indices:
            households.sort_index(inplace=True)
            persons.sort_index(inplace=True)
            trips.sort_index(inplace=True)
            trip_modes.sort_index(inplace=True)
            trip_stations.sort_index(inplace=True)
            if facilitate_passengers is not None:
                facilitate_passengers.sort_index(inplace=True)

        data = cls(
            ms.HouseholdsSchema.validate(households),
            ms.PersonsSchema.validate(persons),
            ms.TripsSchema.validate(trips),
            ms.TripModesSchema.validate(trip_modes),
            ms.TripStationsSchema.validate(trip_stations),
            facilitate_passengers=(
                ms.FacilitatePassengersSchema.validate(facilitate_passengers)
                if facilitate_passengers_fp.exists()
                else None
            ),
        )
        data.attach_zone_system(zone_shapefile, taz_att=zone_label, coord_unit=coord_unit)

        return data

    @staticmethod
    def _load_households(fp: PathLike | str) -> pd.DataFrame:
        spec = {
            "household_id": np.uint64,
        }
        df = pd.read_csv(fp, index_col="household_id", dtype=spec)
        return df

    @staticmethod
    def _load_persons(fp: PathLike | str, *, rebuild_index: bool = False) -> pd.DataFrame:
        spec = {
            "household_id": np.uint64,
            "person_id": str if rebuild_index else np.uint64,
        }
        df = pd.read_csv(fp, index_col=["household_id", "person_id"], dtype=spec)
        return df

    @staticmethod
    def _load_trips(fp: PathLike | str, *, rebuild_index: bool = False) -> pd.DataFrame:
        spec = {
            "household_id": np.uint64,
            "person_id": str if rebuild_index else np.uint64,
            "trip_id": np.uint16,
            "JointTourRep": str if rebuild_index else np.uint64,
        }
        df = pd.read_csv(fp, index_col=["household_id", "person_id", "trip_id"], dtype=spec)
        return df

    @staticmethod
    def _load_trip_modes(fp: PathLike | str, *, rebuild_index: bool = False) -> pd.DataFrame:
        spec = {
            "household_id": np.uint64,
            "person_id": str if rebuild_index else np.uint64,
            "trip_id": np.uint16,
        }
        df = pd.read_csv(fp, index_col=["household_id", "person_id", "trip_id"], dtype=spec)
        return df

    @staticmethod
    def _load_trip_stations(fp: PathLike | str, *, rebuild_index: bool = False) -> pd.DataFrame:
        spec = {
            "household_id": np.uint64,
            "person_id": str if rebuild_index else np.uint64,
            "trip_id": np.uint16,
        }
        df = pd.read_csv(fp, index_col=["household_id", "person_id", "trip_id"], dtype=spec)
        return df

    @staticmethod
    def _load_facilitate_passengers(fp: PathLike | str, *, rebuild_index: bool = False) -> pd.DataFrame:
        spec = {
            "household_id": np.uint64,
            "passenger_id": str if rebuild_index else np.uint64,
            "passenger_trip_id": np.uint16,
            "driver_id": str if rebuild_index else np.uint64,
        }
        df = pd.read_csv(fp, index_col=["household_id", "passenger_id", "passenger_trip_id"], dtype=spec)
        return df

    @staticmethod
    def _rebuild_microsim_indices(
        households: pd.DataFrame,
        persons: pd.DataFrame,
        trips: pd.DataFrame,
        trip_modes: pd.DataFrame,
        trip_stations: pd.DataFrame,
        *,
        facilitate_passengers: pd.DataFrame = None,
    ) -> None:
        """Rebuild invalid microsim indicies in-place"""
        person_idx = persons.index.copy().to_frame()
        person_idx["person_id"] = persons.groupby("household_id")["weight"].cumcount() + 1
        persons.index = pd.MultiIndex.from_frame(person_idx)

        trips_idx = trips.index.copy().to_frame()
        trips_idx["person_id"] = trips.index.droplevel("trip_id").map(person_idx["person_id"])
        trips.index = pd.MultiIndex.from_frame(trips_idx)
        if "JointTourRep" in trips:
            trips["JointTourRep"] = (
                pd.MultiIndex.from_arrays([trips.index.get_level_values("household_id"), trips["JointTourRep"]])
                .map(person_idx["person_id"])
                .fillna(-1)
                .astype(np.int64)
            )

        trip_modes_idx = trip_modes.index.copy().to_frame()
        trip_modes_idx["person_id"] = trip_modes.index.droplevel("trip_id").map(person_idx["person_id"])
        trip_modes.index = pd.MultiIndex.from_frame(trip_modes_idx)

        trip_stations_idx = trip_stations.index.copy().to_frame()
        trip_stations_idx["person_id"] = trip_stations.index.droplevel("trip_id").map(person_idx["person_id"])
        trip_stations.index = pd.MultiIndex.from_frame(trip_stations_idx)

        if facilitate_passengers is not None:
            facilitate_passengers_idx = facilitate_passengers.index.copy().to_frame()
            facilitate_passengers_idx["passenger_id"] = facilitate_passengers.index.droplevel("passenger_trip_id").map(
                person_idx["person_id"]
            )
            facilitate_passengers_idx["driver_id"] = pd.MultiIndex.from_arrays(
                [facilitate_passengers.index.get_level_values("household_id"), facilitate_passengers["driver_id"]]
            ).map(person_idx["person_id"])
            facilitate_passengers["driver_id"] = facilitate_passengers_idx["driver_id"]
            facilitate_passengers.index = pd.MultiIndex.from_frame(facilitate_passengers_idx.drop("driver_id", axis=1))

    # endregion

    # region Finalize initialization

    def _adjust_trip_weights(self) -> None:
        self.logger.info("Adjusting weights in trip mode, station, and facilitate passenger (if available) tables")

        trips = self.trips
        trip_modes = self.trip_modes
        trip_stations = self.trip_stations
        facilitate_passengers = self.facilitate_passengers

        trips["repetitions"] = trip_modes.groupby(["household_id", "person_id", "trip_id"])["weight"].sum()

        trip_modes.rename(columns={"weight": "orig_weight"}, inplace=True)
        trip_modes["weight"] = (
            trip_modes["orig_weight"]
            / trips["repetitions"].reindex(trip_modes.index)
            * trips["weight"].reindex(trip_modes.index)
        )

        if "mode" in trip_stations:
            trip_stations.rename(columns={"weight": "orig_weight"}, inplace=True)
            trip_stations["weight"] = (
                trip_modes.set_index("mode", append=True)
                .loc[trip_stations.set_index("mode", append=True).index, "weight"]
                .droplevel("mode")
            )

        if facilitate_passengers is not None:
            facilitate_passengers.rename(columns={"weight": "orig_weight"}, inplace=True)
            facilitate_passengers["weight"] = (
                facilitate_passengers["orig_weight"]
                / trips["repetitions"].reindex(facilitate_passengers.index)
                * trips["weight"].reindex(facilitate_passengers.index)
            )

    def _classify_times(self, *, time_format: TimeFormat = TimeFormat.MINUTE_DELTA) -> None:
        self.logger.info("Parsing time formats")

        trip_modes = self.trip_modes

        self.logger.debug("Parsing `o_depart`")
        trip_modes["o_depart_hr"] = self._convert_time_to_hours(trip_modes["o_depart"], time_format)

        self.logger.debug("Parsing `d_arrive`")
        trip_modes["d_arrive_hr"] = self._convert_time_to_hours(trip_modes["d_arrive"], time_format)

        self.logger.debug("Classifying `time_period`")
        trip_modes["time_period"] = self._classify_time_periods(trip_modes["o_depart_hr"])

    def _convert_time_to_hours(self, s: pd.Series, time_format: TimeFormat) -> pd.Series:
        if time_format == TimeFormat.MINUTE_DELTA:
            return self._floordiv_minutes(s)
        elif time_format == TimeFormat.COLON_SEP:
            return self._convert_text_to_datetime(s)
        else:
            raise NotImplementedError(time_format)

    def _convert_text_to_datetime(self, s: pd.Series) -> pd.Series:
        colon_count = s.str.count(":")
        filtr = colon_count == 1

        new_time = s.copy()
        new_time.loc[filtr] += ":00"

        filtr = new_time.str.contains("-")
        if filtr.sum() > 0:
            new_time.loc[filtr] = "0:00:00"
            self.logger.warning(f"Found {filtr.sum()} cells with negative time. These have been corrected to 0:00:00")

        time_table = new_time.str.split(":", expand=True).astype(np.int8)
        hours = time_table.iloc[:, 0]

        return hours

    @staticmethod
    def _floordiv_minutes(s: pd.Series) -> pd.Series:
        converted = s.astype(np.float64)
        return (converted // 60).astype(np.int32)

    def _classify_time_periods(self, start_hour: pd.Series) -> pd.Series:
        period = pd.cut(start_hour, self.time_period_bins, right=False, labels=False, include_lowest=True)
        period = period.replace(0, 5).astype("category")
        return period

    def _derive_household_attributes(self) -> None:
        self.logger.info("Deriving additional household attributes")

        self.logger.debug("Processing `drivers`")
        self.households["drivers"] = self._derive_household_drivers()

        self.logger.debug("Processing `auto_suff`")
        self.households["auto_suff"] = self._derive_household_auto_sufficiency()

    def _derive_household_drivers(self) -> pd.Series:
        drivers = self.persons.groupby("household_id")["license"].sum().reindex(self.households.index, fill_value=0)
        return drivers

    def _derive_household_auto_sufficiency(self) -> pd.Series:
        num_cars = self.households["vehicles"]
        num_drivers = self.households["drivers"]
        car_suff = pd.Series(
            np.where(num_cars == 0, 0, (num_cars >= num_drivers) + 1) + 1, index=self.households.index
        )
        return car_suff

    def _derive_person_attributes(self) -> None:
        self.logger.info("Deriving additional person attributes")

        self.logger.debug("Processing `student_class`")
        self.persons["student_class"] = self._derive_person_student_class()

        self.logger.debug("Processing `occ_emp`")
        self.persons["occ_emp"] = self._derive_person_occ_emp()

        self.logger.debug("Processing `work_at_home`")
        self.persons["work_at_home"] = self._derive_person_work_at_home()

    def _derive_person_student_class(self) -> pd.Series:
        stu_class = pd.Series(StudentClass.UNKNOWN, index=self.persons.index, name="student_class")
        stu_class.loc[self.persons["student_status"] == StudentStatus.NOT_A_STUDENT] = StudentClass.NOT_A_STUDENT
        stu_class_groups = {
            StudentClass.PRIMARY: self.student_age_range_primary,
            StudentClass.SECONDARY: self.student_age_range_secondary,
            StudentClass.POST_SECONDARY: self.student_age_range_post_secondary,
        }
        mask_student = self.persons["student_status"].isin({StudentStatus.FULL_TIME, StudentStatus.PART_TIME})
        for label, age_range in stu_class_groups.items():
            mask_age = self.persons["age"].between(*age_range)
            stu_class.loc[mask_student & mask_age] = label
        return stu_class.astype("category")

    def _derive_person_occ_emp(self) -> pd.Series:
        occ_emp = pd.Series(OccEmp.UNKNOWN, index=self.persons.index, name="occ_emp")
        for (occ, emp), subset in self.persons.groupby(["occupation", "employment_status"], observed=True):
            if (occ == Occupation.UNKNOWN) or (emp == EmploymentStatus.UNKNOWN):
                occ_emp.loc[subset.index] = OccEmp.UNKNOWN
            elif (occ == Occupation.NOT_EMPLOYED) or (emp == EmploymentStatus.NOT_EMPLOYED):
                occ_emp.loc[subset.index] = OccEmp.NOT_EMPLOYED
            elif emp == EmploymentStatus.FULL_TIME_AT_HOME:
                occ_emp.loc[subset.index] = f"{occ}{EmploymentStatus.FULL_TIME}"
            elif emp == EmploymentStatus.PART_TIME_AT_HOME:
                occ_emp.loc[subset.index] = f"{occ}{EmploymentStatus.PART_TIME}"
            else:
                occ_emp.loc[subset.index] = f"{occ}{emp}"
        return occ_emp.astype("category")

    def _derive_person_work_at_home(self) -> pd.Series:
        return (
            self.persons["employment_status"]
            .isin([EmploymentStatus.FULL_TIME_AT_HOME, EmploymentStatus.PART_TIME_AT_HOME])
            .rename("work_at_home")
        )

    def _derive_trip_attributes(self) -> None:
        self.logger.info("Deriving additional trip attributes")

        self.logger.debug("Processing `purpose`")
        self.trips["purpose"] = self._derive_trip_purpose()

        # self.logger.debug("Processing `direction`")
        # self.trips["direction"] = self._derive_trip_direction()

    def _derive_trip_purpose(self) -> pd.Series:
        purpose = pd.Series(TripPurpose.OTHER, index=self.trips.index, name="purpose")
        for (o_act, d_act), subset in self.trips.groupby(["o_act", "d_act"], observed=True):
            purpose.loc[subset.index] = activity_pair_mapping[(o_act, d_act)]
        return purpose.astype("category")

    # def _derive_trip_direction(self) -> pd.Series:
    #     pass  # TODO

    # endregion

    # region Additional Data

    def attach_zone_system(
        self,
        zone_shapefile: PathLike | str,
        *,
        taz_att: str = "taz",
        coord_unit: float = 0.001,
    ) -> None:
        """Attach zone system information for analysis

        Args:
            zone_shapefile (PathLike | str): Path to the zone shapefile
            taz_att (str, optional): Defaults to ``"taz"``. Name of the traffic analysis zone (TAZ) attribute
            coord_unit (float, optional): Defaults to ``0.001``. A value to adjust distance values with.
        """
        self.logger.info("Attaching zone system for analysis")

        self._zones = self._load_zone_shapefile(zone_shapefile, taz_att=taz_att)

        self.logger.debug("Calculating standard impedances from zone coordinates")
        zones_mindex = pd.MultiIndex.from_product([self._zones.index] * 2, names=["o", "d"])
        if self._impedances is None:
            self._impedances = pd.DataFrame(index=zones_mindex)
        else:
            self._impedances = self._impedances.reindex(zones_mindex, fill_value=0)
        self._impedances.insert(0, "manhattan", self._calc_std_impedance("manhattan", coord_unit))
        self._impedances.insert(1, "euclidean", self._calc_std_impedance("euclidean", coord_unit))

    @staticmethod
    def _load_zone_shapefile(
        fp: PathLike | str,
        *,
        taz_att: str,
    ) -> gpd.GeoDataFrame:
        zones = gpd.read_file(fp)
        zones = zones.astype({taz_att: np.int64})
        zones.set_index(taz_att, inplace=True)
        zones.sort_index(inplace=True)
        return zones

    def _calc_std_impedance(
        self,
        method: Literal["manhattan", "euclidean", "haversine"],
        coord_unit: float,
    ) -> pd.Series:
        mtx = distance_matrix(
            self.zones.centroid.x, self.zones.centroid.y, tall=True, method=method, coord_unit=coord_unit
        ).rename(method)
        mtx.index.names = ["o", "d"]
        return mtx

    def attach_impedance(
        self,
        name: str,
        impedance_data: pd.Series | PathLike | str,
        *,
        scale_unit: float = 1.0,
        ignore_missing_ods: bool = False,
    ) -> None:
        """Attach impedance values to the impedance table for analysis

        Args:
            name (str): The name for the impedeance values
            impedance_data (pd.Series | PathLike | str): Impedance values in the form of matrix data (formatted as a
                stacked pandas Series) or path to a matrix binary file.
            scale_unit (float, optional): Defaults to ``1.0``. A scalar value to adjust impedance values.
            ignore_missing_ods (bool, optional): Defaults to ``False``. A flag to ignore missing ODs. If ``True``,
                impedance values for missing ODs will be set to zero.
        """
        self.logger.info(f"Attaching `{name}` impedance values")

        if self.zones is None:
            raise RuntimeError("Please attach a zone system first before attaching impedances")

        if not isinstance(impedance_data, pd.Series):
            impedance_data = read_mdf(impedance_data, tall=True)
        impedance_data = impedance_data.reindex(self.impedances.index)

        if not ignore_missing_ods:
            if impedance_data.isna().any():
                raise ValueError(
                    "NaN values found after reindexing impedance data to zone system. Please check if `impedance_data` "
                    "is compatible with the dataset zone system."
                )
        impedance_data.fillna(0, inplace=True)

        impedance_data = impedance_data * scale_unit

        self.impedances[name] = impedance_data

    # endregion
