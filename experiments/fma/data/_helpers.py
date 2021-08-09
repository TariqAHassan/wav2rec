"""

    FMA Helpers

"""
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

_LIST_GENRE_COLUMNS: Tuple[str, ...] = ("track_genres", "track_genres_all")


def join_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = ["_".join(i) for i in df.columns]
    return df


def sort_list_genres(row: pd.Series) -> pd.Series:
    # Ensure the top genre is always first in the 'list genre'
    # fields (i.e., 'track_genres' and 'track_genres_all').
    def weight(g: str, fallback: str) -> int:
        return -1 if g == row["track_genre_top"] else row[fallback].index(g)

    for c in _LIST_GENRE_COLUMNS:
        if isinstance(row[c], list):
            row[c] = sorted(row[c], key=lambda g: weight(g, fallback=c))

    return row


def column_filter(
    df: pd.DataFrame,
    column: str,
    values: Optional[Union[str, List[str]]] = None,
) -> pd.DataFrame:
    if values is None:
        return df
    elif isinstance(values, str):
        parsed_values = [values.lower()]
    else:
        parsed_values = [s.lower() for s in values]
    return df[df[column].str.lower().isin(parsed_values).values]


class TrackLicenseFilter:
    def filter(self, track_license: pd.Series) -> np.ndarray:
        raise NotImplementedError()

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        sel = self.filter(df["track_license"].str.lower())
        return df[sel]


class PdTrackLicenseFilter(TrackLicenseFilter):
    def filter(self, track_license: pd.Series) -> np.ndarray:
        sel = (
            track_license.astype(str)
            .str.strip()
            .str.lower()
            .str.contains("public domain")
        )
        return sel.values
