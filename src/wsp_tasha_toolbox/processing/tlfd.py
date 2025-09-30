from __future__ import annotations

__all__ = [
    "extract_tlfds",
]

import pandas as pd
from wsp_balsa.routines import tlfd


def extract_tlfds(
    table: pd.DataFrame,
    agg_col: str,
    *,
    impedance_col: str = "impedance",
    bin_start: int = 0,
    bin_end: int = 200,
    bin_step: int = 2,
    orig_col: str = "o_zone",
    dest_col: str = "d_zone",
) -> pd.DataFrame:
    """A function to extract TLFDs from model results.

    Args:
        table (pd.DataFrame): The table from the Microsim results. Ideally, this table would be from
            ``MicrosimData.trips`` or ``MicrosimData.persons``.
        agg_col (str): The name of the column in the table to plot TLFDs by category/group.
        impedance_col (str, optional): Defaults to ``"impedance"``. The column in ``table`` containing the impedances
            to use for calculating TFLDs.
        bin_start (int): Defaults is ``0``. The minimum bin value.
        bin_end (int): Defaults to ``200``. The maximum bin value.
        bin_step (int): Default is ``2``. The size of each bin.
        orig_col (str, optional): Defaults to ``"o_zone"``. The name of the column to use as the origin.
        dest_col (str, optional): Defaults to ``"d_zone"``. The name of the column to use as the destination.

    Returns:
        pd.DataFrame: The TLFDs from the model results trip table in wide format.
    """
    retval = {}
    for label, subset in table.groupby(agg_col):
        df = tlfd(
            subset[impedance_col],
            bin_start=bin_start,
            bin_end=bin_end,
            bin_step=bin_step,
            weights=subset["weight"],
            intrazonal=subset[orig_col] == subset[dest_col],
            label_type="MULTI",
            include_top=True,
        )
        retval[label] = df
    retval = pd.DataFrame(retval)
    retval.columns.name = agg_col

    return retval
