"""

    Build Dataset

"""
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from cached_property import cached_property

from experiments.fma.data._helpers import (
    PdTrackLicenseFilter,
    TrackLicenseFilter,
    column_filter,
    join_columns,
    sort_list_genres,
)


class FmaMetadata:
    """FMA Metadata.

    Args:
        path (Path): path to FMA metadata
        genres (str, list, optional): the subset of genres to include.
        set_subset (str, list, optional): the subset of the full dataset that
            ``audio_path`` refers to. If ``None``, will be determined
            automatically.
        license_filter (TrackLicenseFilter): tool for filtering tracks
            based on their license
        min_listens (int): the minimum number of listens a song must
            have in order to be included. Set to ``0`` to disable.

    References:
        * https://github.com/mdeff/fma

    """

    def __init__(
        self,
        path: Path = Path("~/fma_metadata").expanduser(),
        genres: Optional[Union[str, List[str]]] = None,
        set_subset: Optional[Union[str, List[str]]] = None,
        license_filter: TrackLicenseFilter = PdTrackLicenseFilter(),
        min_listens: int = 0,
    ) -> None:
        self.path = path
        self.genres = genres
        self.set_subset = set_subset
        self.license_filter = license_filter
        self.min_listens = min_listens

    @cached_property
    def genre_df(self) -> pd.DataFrame:
        return pd.read_csv(self.path.joinpath("genres.csv"))

    @cached_property
    def genre_id_mapping(self) -> Dict[int, str]:
        return self.genre_df.set_index("genre_id")["title"].to_dict()

    @lru_cache(maxsize=2 ** 16)
    def _decode_genres(self, genres: Optional[str]) -> Optional[List[str]]:
        if not genres or not isinstance(genres, str) or genres == "[]":
            return np.nan
        genres_parsed = genres.strip("[]").split(",")
        return [self.genre_id_mapping[int(i.strip())] for i in genres_parsed]

    @cached_property
    def track_df(self) -> pd.DataFrame:
        return (
            pd.read_csv(
                self.path.joinpath("tracks.csv"),
                header=[0, 1],
                low_memory=False,
            )
            .pipe(join_columns)
            .rename(columns={"Unnamed: 0_level_0_Unnamed: 0_level_1": "track_id"})
            .drop("artist_website", axis=1)
            .iloc[1:]
            .pipe(self.license_filter)
            .pipe(column_filter, column="track_genre_top", values=self.genres)
            .pipe(column_filter, column="set_subset", values=self.set_subset)
            .iloc[lambda f: (f.track_listens >= self.min_listens).values]
            .assign(
                track_id=lambda f: f.track_id.astype(int),
                track_genres=lambda f: f.track_genres.map(self._decode_genres),
                track_genres_all=lambda f: f.track_genres_all.map(self._decode_genres),
            )
            .apply(sort_list_genres, axis=1)
            .reset_index(drop=True)
        )

    @cached_property
    def _track_duration_mapping(self) -> Dict[int, float]:
        return self.track_df.set_index("track_id")["track_duration"].to_dict()

    def get_track_duration(self, track_id: Union[str, int]) -> float:
        return self._track_duration_mapping[int(track_id)]

    @cached_property
    def _track_top_genre_mapping(self) -> Dict[int, float]:
        return self.track_df.set_index("track_id")["track_genre_top"].to_dict()

    def get_track_top_genre(self, track_id: Union[str, int]) -> str:
        return self._track_top_genre_mapping[int(track_id)]

    @cached_property
    def _track_genres_mapping(self) -> Dict[int, float]:
        return self.track_df.set_index("track_id")["track_genres_all"].to_dict()

    def get_track_genres(self, track_id: Union[str, int]) -> List[str]:
        return self._track_genres_mapping[int(track_id)]
