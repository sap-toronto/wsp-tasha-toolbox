from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from wsp_balsa.logging import ModelLogger, get_model_logger
from wsp_balsa.routines import distance_matrix, read_mdf

from ..common.enums_data import TimeFormat
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
        zones (gpd.GeoDataFrame, optional): Defaults to ``None``. The modelled zone system
        reweight_trips (bool, optional): Defaults to ``True``. A flag to reweight trip modes, trip station, and
            facilitate passenger (if available) tables to match total number of trips modelled in the trip table.
        derive_additional_attributes (bool, optional): Defaults to ``True``. A flag to calculate additional attributes
            useful in model analyses
    """

    time_period_bins = (0, 6, 9, 15, 19, 24, np.inf)

    def __init__(
        self,
        households: pd.DataFrame,
        persons: pd.DataFrame,
        trips: pd.DataFrame,
        trip_modes: pd.DataFrame,
        trip_stations: pd.DataFrame,
        *,
        facilitate_passengers: pd.DataFrame = None,
        zones: gpd.GeoDataFrame = None,
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
        self._zones = zones

        if reweight_trips:
            self._adjust_trip_weights()

        if derive_additional_attributes:
            self._classify_times()

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

    # endregion

    # region Loading

    @classmethod
    def from_result_folder(
        cls,
        results_folder: PathLike | str,
        *,
        rebuild_indices: bool = True,
        sort_indices: bool = True,
    ) -> MicrosimData:
        """Initialize a new instance of MicrosimData using files from a model run result folder

        Args:
            results_folder (PathLike | str): Path to the model run result folder
            rebuild_indices (bool, optional): Defaults to ``True``. A flag to rebuild indices in the microsim tables.
            sort_indices (bool, optional): Defaults to ``True``. A flag to sort indices in the microsim tables.
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

    # endregion
