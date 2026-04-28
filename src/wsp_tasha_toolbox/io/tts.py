from __future__ import annotations

__all__ = [
    "read_tts_cross_tabulation_file",
]

import io
import re
from os import PathLike
from pathlib import Path

import pandas as pd

# region Header parsing


def _parse_ct_header_attributes(lines: list[str]) -> tuple[str, str, str | None]:
    """Extract the row, column, and table attribute names from the header lines.

    Args:
        lines: Lines of a TTS Cross Tabulation query result.

    Returns:
        tuple[str, str, str | None]: row attribute name, column attribute name, and optional table attribute name.
    """
    row_att: str | None = None
    col_att: str | None = None
    table_att: str | None = None

    for line in lines:
        stripped = line.strip()
        if (row_att is None) and stripped.startswith("Row:") and ("-" in stripped):
            row_att = stripped.split("-")[-1].strip()
        if (col_att is None) and stripped.startswith("Column:") and ("-" in stripped):
            col_att = stripped.split("-")[-1].strip()
        if (table_att is None) and stripped.startswith("Table:") and ("-" in stripped):
            table_att = stripped.split("-")[-1].strip()

    if (row_att is None) or (col_att is None):
        raise ValueError("Could not find row and column attribute names in the header lines.")

    return row_att, col_att, table_att


# endregion

# region Comma (wide) format parsing


def _parse_ct_comma_block(csv_text: str, row_att: str, col_att: str) -> pd.DataFrame:
    df: pd.DataFrame = pd.read_csv(io.StringIO(csv_text), index_col=0)
    df.index.name = row_att
    df.columns.name = col_att
    df = df.stack().to_frame(name="total")
    df.reset_index(inplace=True)

    return df


def _parse_ct_comma_format(
    lines: list[str],
    row_att: str,
    col_att: str,
    table_att: str | None,
) -> pd.DataFrame:
    sections: list[tuple[str | None, int]] = []
    if table_att is None:
        for i, line in enumerate(lines):
            m = re.match(r"^Table:$", line.strip())
            if m:
                sections.append((None, i + 1))
        if len(sections) != 1:
            raise ValueError("Expected only one 'Table:' section when no table attribute is specified.")
    else:
        for i, line in enumerate(lines):
            m = re.match(r"^Table:\s+(.+)$", line.strip())
            if m and (not m.group(1).endswith(table_att)):
                sections.append((m.group(1).strip(), i + 1))
        if len(sections) == 0:
            raise ValueError("Expected at least one 'Table: <name>' section when a table attribute is specified.")

    tables = []
    for table_val, sec_start in sections:
        csv_start = next((k for k in range(sec_start, len(lines)) if lines[k].strip().startswith(",")), None)
        csv_lines = []
        for ln in lines[csv_start:]:
            if ln.strip():
                csv_lines.append(ln)
            else:
                break

        df = _parse_ct_comma_block("\n".join(csv_lines), row_att, col_att)
        if table_val is not None:
            df.insert(df.columns.get_loc("total"), table_att, table_val)
        tables.append(df)

    return pd.concat(tables, axis=0, ignore_index=True)


# endregion

# region Column (long) format parsing


def _parse_ct_column_block(data_text: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(data_text), sep=r"\s+", skipinitialspace=True)


def _parse_ct_column_format(
    lines: list[str],
    row_att: str,
    col_att: str,
    table_att: str | None,
) -> pd.DataFrame:
    sections: list[tuple[str | None, int]] = []
    if table_att is None:
        for i, line in enumerate(lines):
            m = re.match(r"^COLUMN :\s*.+$", line.strip())
            if m:
                sections.append((None, i))
        if len(sections) != 1:
            raise ValueError("Expected only one section when no table attribute is specified.")
    else:
        for i, line in enumerate(lines):
            m = re.match(r"^TABLE\s*:\s*.+\((.+)\)\s*$", line.strip())
            if m:
                sections.append((m.group(1).strip(), i + 1))
        if len(sections) == 0:
            raise ValueError(
                "Expected at least one 'TABLE : <att> (<name>)' section when a table attribute is specified."
            )

    tables = []
    for j, (table_val, sec_start) in enumerate(sections):
        data_start = next((k for k in range(sec_start + 1, len(lines)) if lines[k].strip()), None)
        sec_end = sections[j + 1][1] if j + 1 < len(sections) else len(lines)
        data_lines = [ln for ln in lines[data_start:sec_end] if ln.strip()]

        df = _parse_ct_column_block("\n".join(data_lines))
        if table_val is not None:
            df.insert(df.columns.get_loc("total"), table_att, table_val)
        tables.append(df)

    return pd.concat(tables, axis=0, ignore_index=True)


# endregion


def read_tts_cross_tabulation(lines: list[str]) -> pd.DataFrame:
    """A function to read a TTS Cross Tabulation query result from a list of lines.

    Args:
        lines (list[str]): Lines of a TTS Cross Tabulation query result.

    Returns:
        pd.DataFrame
    """
    row_att, col_att, table_att = _parse_ct_header_attributes(lines)

    is_column_format = any(line.strip().startswith("ROW :") for line in lines)
    if is_column_format:
        return _parse_ct_column_format(lines, row_att, col_att, table_att)
    return _parse_ct_comma_format(lines, row_att, col_att, table_att)


def read_tts_cross_tabulation_file(fp: PathLike | str) -> pd.DataFrame:
    """A function to read a TTS Cross Tabulation file downloaded from the DMG Data Retrieval System.

    Args:
        fp (PathLike | str): File path to the TTS Cross Tabulation file.

    Returns:
        pd.DataFrame
    """
    fp = Path(fp).resolve(strict=True)
    with open(fp) as f:
        lines = [line.strip() for line in f if line.strip()]

    return read_tts_cross_tabulation(lines)
