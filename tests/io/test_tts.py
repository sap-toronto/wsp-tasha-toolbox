from __future__ import annotations

import pytest

from wsp_tasha_toolbox.io.tts import (
    _parse_ct_header_attributes,
    read_tts_cross_tabulation,
)

COMMA_NO_TABLE = """
Tue Apr 28 2026 08:50:11 GMT-0400 (Eastern Daylight Time) - Run Time: 601ms

Cross Tabulation Query Form - Household - 2022

Row: Regional municipality of household - region_hhld
Column: Income range of household - income


Filters:
Regional municipality of household - region_hhld In 1

Household 2022
Table:

,$0-$14999,$15000-$39999,$40000-$59999,$60000-$79999,$80000-$99999,$100000-$124999,$125000-$149999,$150000-$199999,$200000 and above,Decline/Don't know
Toronto,44310,122326,119440,124470,119692,126242,79666,102179,142977,189719
"""

COMMA_WITH_TABLE = """
Tue Apr 28 2026 08:48:51 GMT-0400 (Eastern Daylight Time) - Run Time: 680ms

Cross Tabulation Query Form - Household - 2022

Row: Regional municipality of household - region_hhld
Column: Income range of household - income
Table: Type of dwelling unit - dwell_type


Filters:
Regional municipality of household - region_hhld In 1

Household 2022
Table: House

,$0-$14999,$15000-$39999,$40000-$59999,$60000-$79999,$80000-$99999,$100000-$124999,$125000-$149999,$150000-$199999,$200000 and above,Decline/Don't know
Toronto,2867,16889,21419,23669,27885,35213,26060,39247,82731,71198

Household 2022
Table: Apartment

,$0-$14999,$15000-$39999,$40000-$59999,$60000-$79999,$80000-$99999,$100000-$124999,$125000-$149999,$150000-$199999,$200000 and above,Decline/Don't know
Toronto,39817,100675,92070,94036,85111,82995,47601,56188,49738,108774

Household 2022
Table: Townhouse

,$0-$14999,$15000-$39999,$40000-$59999,$60000-$79999,$80000-$99999,$100000-$124999,$125000-$149999,$150000-$199999,$200000 and above,Decline/Don't know
Toronto,1626,4761,5950,6765,6696,8034,6004,6744,10507,9747
"""

COLUMN_NO_TABLE = """
Tue Apr 28 2026 08:49:59 GMT-0400 (Eastern Daylight Time) - Run Time: 622ms

Cross Tabulation Query Form - Household - 2022

Row: Regional municipality of household - region_hhld
Column: Income range of household - income


Filters:
Regional municipality of household - region_hhld In 1

Household 2022
ROW : region_hhld
COLUMN : income
 region_hhld      income      total
           1           1      44310
           1           2     122326
           1           3     119440
           1           4     124470
           1           5     119692
           1           6     126242
           1           7      79666
           1           8     102179
           1           9     142977
           1          99     189719
"""

COLUMN_WITH_TABLE = """
Tue Apr 28 2026 08:49:27 GMT-0400 (Eastern Daylight Time) - Run Time: 773ms

Cross Tabulation Query Form - Household - 2022

Row: Regional municipality of household - region_hhld
Column: Income range of household - income
Table: Type of dwelling unit - dwell_type


Filters:
Regional municipality of household - region_hhld In 1

Household 2022
ROW : region_hhld
COLUMN : income

TABLE    : dwell_type (House)

 region_hhld      income      total
           1           1       2867
           1           2      16889
           1           3      21419
           1           4      23669
           1           5      27885
           1           6      35213
           1           7      26060
           1           8      39247
           1           9      82731
           1          99      71198

TABLE    : dwell_type (Apartment)

 region_hhld      income      total
           1           1      39817
           1           2     100675
           1           3      92070
           1           4      94036
           1           5      85111
           1           6      82995
           1           7      47601
           1           8      56188
           1           9      49738
           1          99     108774

TABLE    : dwell_type (Townhouse)

 region_hhld      income      total
           1           1       1626
           1           2       4761
           1           3       5950
           1           4       6765
           1           5       6696
           1           6       8034
           1           7       6004
           1           8       6744
           1           9      10507
           1          99       9747
"""


def _to_lines(text: str) -> list[str]:
    """Replicate the line-stripping logic used by read_tts_cross_tabulation_file."""
    return [line.rstrip() for line in text.splitlines()]


class TestParseCtHeaderAttributes:
    def test_no_table_attribute(self):
        lines = _to_lines(COMMA_NO_TABLE)
        row_att, col_att, table_att = _parse_ct_header_attributes(lines)
        assert row_att == "region_hhld"
        assert col_att == "income"
        assert table_att is None

    def test_with_table_attribute(self):
        lines = _to_lines(COMMA_WITH_TABLE)
        row_att, col_att, table_att = _parse_ct_header_attributes(lines)
        assert row_att == "region_hhld"
        assert col_att == "income"
        assert table_att == "dwell_type"

    def test_column_format_no_table(self):
        lines = _to_lines(COLUMN_NO_TABLE)
        row_att, col_att, table_att = _parse_ct_header_attributes(lines)
        assert row_att == "region_hhld"
        assert col_att == "income"
        assert table_att is None

    def test_column_format_with_table(self):
        lines = _to_lines(COLUMN_WITH_TABLE)
        row_att, col_att, table_att = _parse_ct_header_attributes(lines)
        assert row_att == "region_hhld"
        assert col_att == "income"
        assert table_att == "dwell_type"

    def test_missing_row_col_raises(self):
        lines = ["Some header", "No row or column info here"]
        with pytest.raises(ValueError, match="row and column attribute names"):
            _parse_ct_header_attributes(lines)


class TestReadTtsCrossTabulation_CommaNoTable:
    @pytest.fixture(autouse=True)
    def result(self):
        self.df = read_tts_cross_tabulation(_to_lines(COMMA_NO_TABLE))

    def test_columns(self):
        assert list(self.df.columns) == ["region_hhld", "income", "total"]

    def test_shape(self):
        assert self.df.shape == (10, 3)

    def test_region_values(self):
        assert (self.df["region_hhld"] == "Toronto").all()

    def test_income_labels(self):
        assert list(self.df["income"]) == [
            "$0-$14999",
            "$15000-$39999",
            "$40000-$59999",
            "$60000-$79999",
            "$80000-$99999",
            "$100000-$124999",
            "$125000-$149999",
            "$150000-$199999",
            "$200000 and above",
            "Decline/Don't know",
        ]

    def test_total_values(self):
        assert list(self.df["total"]) == [44310, 122326, 119440, 124470, 119692, 126242, 79666, 102179, 142977, 189719]


class TestReadTtsCrossTabulation_CommaWithTable:
    @pytest.fixture(autouse=True)
    def result(self):
        self.df = read_tts_cross_tabulation(_to_lines(COMMA_WITH_TABLE))

    def test_columns(self):
        assert list(self.df.columns) == ["region_hhld", "income", "dwell_type", "total"]

    def test_shape(self):
        # 3 dwelling types × 10 income categories
        assert self.df.shape == (30, 4)

    def test_dwelling_types(self):
        assert set(self.df["dwell_type"]) == {"House", "Apartment", "Townhouse"}

    def test_house_totals(self):
        house = self.df[self.df["dwell_type"] == "House"].reset_index(drop=True)
        assert list(house["total"]) == [2867, 16889, 21419, 23669, 27885, 35213, 26060, 39247, 82731, 71198]

    def test_apartment_totals(self):
        apt = self.df[self.df["dwell_type"] == "Apartment"].reset_index(drop=True)
        assert list(apt["total"]) == [39817, 100675, 92070, 94036, 85111, 82995, 47601, 56188, 49738, 108774]

    def test_townhouse_totals(self):
        twn = self.df[self.df["dwell_type"] == "Townhouse"].reset_index(drop=True)
        assert list(twn["total"]) == [1626, 4761, 5950, 6765, 6696, 8034, 6004, 6744, 10507, 9747]


class TestReadTtsCrossTabulation_ColumnNoTable:
    @pytest.fixture(autouse=True)
    def result(self):
        self.df = read_tts_cross_tabulation(_to_lines(COLUMN_NO_TABLE))

    def test_columns(self):
        assert list(self.df.columns) == ["region_hhld", "income", "total"]

    def test_shape(self):
        assert self.df.shape == (10, 3)

    def test_region_values(self):
        assert (self.df["region_hhld"] == 1).all()

    def test_income_codes(self):
        assert list(self.df["income"]) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 99]

    def test_total_values(self):
        assert list(self.df["total"]) == [44310, 122326, 119440, 124470, 119692, 126242, 79666, 102179, 142977, 189719]


class TestReadTtsCrossTabulation_ColumnWithTable:
    @pytest.fixture(autouse=True)
    def result(self):
        self.df = read_tts_cross_tabulation(_to_lines(COLUMN_WITH_TABLE))

    def test_columns(self):
        assert list(self.df.columns) == ["region_hhld", "income", "dwell_type", "total"]

    def test_shape(self):
        # 3 dwelling types × 10 income categories
        assert self.df.shape == (30, 4)

    def test_dwelling_types(self):
        assert set(self.df["dwell_type"]) == {"House", "Apartment", "Townhouse"}

    def test_house_totals(self):
        house = self.df[self.df["dwell_type"] == "House"].reset_index(drop=True)
        assert list(house["total"]) == [2867, 16889, 21419, 23669, 27885, 35213, 26060, 39247, 82731, 71198]

    def test_apartment_totals(self):
        apt = self.df[self.df["dwell_type"] == "Apartment"].reset_index(drop=True)
        assert list(apt["total"]) == [39817, 100675, 92070, 94036, 85111, 82995, 47601, 56188, 49738, 108774]

    def test_townhouse_totals(self):
        twn = self.df[self.df["dwell_type"] == "Townhouse"].reset_index(drop=True)
        assert list(twn["total"]) == [1626, 4761, 5950, 6765, 6696, 8034, 6004, 6744, 10507, 9747]
