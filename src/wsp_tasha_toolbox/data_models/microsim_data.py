from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from wsp_balsa.logging import ModelLogger, get_model_logger
from wsp_balsa.routines import distance_matrix, read_mdf

from . import schema as ms


class MicrosimData:

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
    ):
        self._households = households
        self._persons = persons
        self._trips = trips
        self._trip_modes = trip_modes
        self._trip_stations = trip_stations
        self._facilitate_passengers = facilitate_passengers
        self._zones = zones

    @classmethod
    def from_result_folder(cls, results_folder: PathLike | str) -> MicrosimData:
        microsim_folder = Path(results_folder) / "Microsim Results"

        households = cls._load_households(microsim_folder / "households.csv")
        persons = cls._load_persons(microsim_folder / "persons.csv")
        trips = cls._load_trips(microsim_folder / "trips.csv")
        trip_modes = cls._load_trips(microsim_folder / "trip_modes.csv")
        trip_stations = cls._load_trip_stations(microsim_folder / "trip_stations.csv")
        facilitate_passengers_fp = microsim_folder / "facilitate_passenger.csv"
        if facilitate_passengers_fp.exists():
            facilitate_passengers = cls._load_facilitate_passengers(facilitate_passengers_fp)
        else:
            facilitate_passengers = None

        data = cls(
            households,
            persons,
            trips,
            trip_modes,
            trip_stations,
            facilitate_passenger=facilitate_passengers,
        )

        return data

    @property
    def logger(self) -> ModelLogger:
        return get_model_logger(f"wsp_tasha_toolbox.{self.__class__.__name__}")

    @property
    def households(self) -> pd.DataFrame:
        return self._households

    @property
    def persons(self) -> pd.DataFrame:
        return self._persons

    @property
    def trips(self) -> pd.DataFrame:
        return self._trips

    @property
    def trip_modes(self) -> pd.DataFrame:
        return self._trip_modes

    @property
    def trip_stations(self) -> pd.DataFrame:
        return self._trip_stations

    @property
    def facilitate_passengers(self) -> Optional[pd.DataFrame]:
        return self._facilitate_passengers

    @property
    def zones(self) -> Optional[gpd.GeoDataFrame]:
        return self._zones

    @staticmethod
    def _load_households(fp: PathLike | str) -> pd.DataFrame:
        spec = {
            "household_id": np.uint64,
        }
        df = pd.read_csv(fp, index_col="household_id", dtype=spec)
        return ms.HouseholdsSchema.validate(df)

    @staticmethod
    def _load_persons(fp: PathLike | str) -> pd.DataFrame:
        spec = {
            "household_id": np.uint64,
            "person_id": np.uint64,
        }
        df = pd.read_csv(fp, index_col=["household_id", "person_id"], dtype=spec)
        return ms.PersonsSchema.validate(df)

    @staticmethod
    def _load_trips(fp: PathLike | str) -> pd.DataFrame:
        spec = {
            "household_id": np.uint64,
            "person_id": np.uint64,
            "trip_id": np.uint16,
        }
        df = pd.read_csv(fp, index_col=["household_id", "person_id", "trip_id"], dtype=spec)
        return ms.TripsSchema.validate(df, lazy=True)

    @staticmethod
    def _load_trip_modes(fp: PathLike | str) -> pd.DataFrame:
        spec = {
            "household_id": np.uint64,
            "person_id": np.uint64,
            "trip_id": np.uint16,
        }
        df = pd.read_csv(fp, index_col=["household_id", "person_id", "trip_id"], dtype=spec)
        return ms.TripModesSchema.validate(df)

    @staticmethod
    def _load_trip_stations(fp: PathLike | str) -> pd.DataFrame:
        spec = {
            "household_id": np.uint64,
            "person_id": np.uint64,
            "trip_id": np.uint16,
        }
        df = pd.read_csv(fp, index_col=["household_id", "person_id", "trip_id"], dtype=spec)
        return ms.TripStationsSchema.validate(df)

    @staticmethod
    def _load_facilitate_passengers(fp: PathLike | str) -> pd.DataFrame:
        spec = {
            "household_id": np.uint64,
            "passenger_id": np.uint64,
            "passenger_trip_id": np.uint16,
        }
        df = pd.read_csv(fp, index_col=["household_id", "passenger_id", "passenger_trip_id"], dtype=spec)
        return ms.FacilitatePassengersSchema.validate(df)
