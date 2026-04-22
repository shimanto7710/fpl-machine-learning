from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


FORMATION = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
TARGET_COL = "total_points"
SELECTION_SCORE_COL = "predicted_points"
MAX_PLAYERS_PER_TEAM = 3
TOP_PLAYERS_DEFAULT_LIMIT = 50
FPL_BOOTSTRAP_STATIC_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FPL_FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
HTTP_TIMEOUT_SECONDS = 15
INITIAL_COLS_TO_DROP = [
    "season",
    "penalties_conceded",
    "big_chances_created",
    "big_chances_missed",
    "name",
    "bonus",
    "clean_sheets",
    "creativity",
    "goals_conceded",
    "goals_scored",
    "ict_index",
    "influence",
    "kickoff_time",
    "kickoff_time_formatted",
    "minutes",
    "GW",
    "assists",
    "own_goals",
    "red_cards",
    "round",
    "selected",
    "team_a_score",
    "team_h_score",
    "threat",
    "transfers_balance",
    "transfers_in",
    "penalties_missed",
    "penalties_saved",
    "position",
    "saves",
    "transfers_out",
    "value",
    "yellow_cards",
    "big_chances_created_prev_3_mean",
    "big_chances_created_prev_all_mean",
    "big_chances_missed_prev_3_mean",
    "big_chances_missed_prev_all_mean",
    "assists_prev_5_mean",
    "goals_conceded_prev_5_mean",
    "goals_scored_prev_5_mean",
    "total_points_prev_5_same_opponent_mean",
    "penalties_saved_prev_3_mean",
    "penalties_saved_prev_all_mean",
    "team_a_score_prev_3_mean",
    "team_a_score_prev_all_mean",
    "team_h_score_prev_3_mean",
    "team_h_score_prev_all_mean",
    "opponent_strength_overall_away",
    "opponent_strength_overall_home",
    "self_team_strength_overall_away",
    "self_team_strength_overall_home",
    "goals_scored_prev_all_mean",
    "clean_sheets_prev_all_mean",
    "threat_prev_all_mean",
    "creativity_prev_all_mean",
    "goals_conceded_prev_all_mean",
    "saves_prev_3_mean",
    "assists_prev_all_mean",
    "saves_prev_all_mean",
    "team",
    "opponent_team_name",
    "total_points_prev_3_same_opponent_mean",
    "goals_scored_prev_3_mean",
    "goals_conceded_prev_3_mean",
    "clean_sheets_prev_3_mean",
    "assists_prev_3_mean",
    "player_image_url",
]


@dataclass(frozen=True)
class ArtifactPaths:
    model: Path
    scaler: Path
    metadata: Path


@dataclass
class ModelArtifacts:
    model: XGBRegressor
    scaler: StandardScaler
    feature_columns: list[str]
    raw_df: pd.DataFrame


class FPLPredictionService:
    def __init__(
        self,
        data_path: str | Path = "fpl_final.csv",
        artifacts_dir: str | Path = "artifacts",
    ) -> None:
        self.data_path = Path(data_path)
        self.artifacts_dir = Path(artifacts_dir)
        self.paths = ArtifactPaths(
            model=self.artifacts_dir / "xgb_model.json",
            scaler=self.artifacts_dir / "scaler.joblib",
            metadata=self.artifacts_dir / "metadata.json",
        )
        self.artifacts: ModelArtifacts | None = None

    def load(self) -> None:
        self._validate_artifacts()

        metadata = json.loads(self.paths.metadata.read_text())

        model = XGBRegressor()
        model.load_model(self.paths.model)

        scaler = joblib.load(self.paths.scaler)
        raw_df = pd.read_csv(self.data_path, low_memory=False)

        self.artifacts = ModelArtifacts(
            model=model,
            scaler=scaler,
            feature_columns=metadata["feature_columns"],
            raw_df=raw_df,
        )

    def _validate_artifacts(self) -> None:
        missing = [
            str(path)
            for path in (self.paths.model, self.paths.scaler, self.paths.metadata)
            if not path.exists()
        ]
        if missing:
            missing_str = ", ".join(missing)
            raise FileNotFoundError(
                "Missing model artifacts: "
                f"{missing_str}. Export them from the notebook or run "
                "`python train_artifacts.py` first."
            )

    @staticmethod
    def _ceil_points(value: float) -> int:
        return int(math.ceil(float(value)))

    @staticmethod
    def _normalize_team_name(name: str) -> str:
        return (
            str(name)
            .lower()
            .replace("'", "")
            .replace(".", "")
            .replace("-", " ")
            .replace("&", "and")
            .replace(" ", "")
        )

    def _build_team_strength_map(self, season_df: pd.DataFrame) -> dict[str, float]:
        strengths = (
            season_df.groupby("team", as_index=False)["my_team_strength"]
            .median()
            .set_index("team")["my_team_strength"]
            .to_dict()
        )
        return {str(team): float(value) for team, value in strengths.items()}

    def _build_same_opponent_stats_map(
        self, season_df: pd.DataFrame
    ) -> dict[tuple[str, str], tuple[float, float]]:
        grouped = (
            season_df.sort_values(["name", "opponent_team_name", "GW"])
            .groupby(["name", "opponent_team_name"], as_index=False)
            .tail(3)
            .groupby(["name", "opponent_team_name"], as_index=False)[
                ["assists", "goals_scored"]
            ]
            .mean()
        )
        stats_map: dict[tuple[str, str], tuple[float, float]] = {}
        for _, row in grouped.iterrows():
            key = (str(row["name"]), str(row["opponent_team_name"]))
            stats_map[key] = (float(row["assists"]), float(row["goals_scored"]))
        return stats_map

    def _fetch_bootstrap_static(self) -> dict:
        try:
            bootstrap_resp = requests.get(
                FPL_BOOTSTRAP_STATIC_URL, timeout=HTTP_TIMEOUT_SECONDS
            )
            bootstrap_resp.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError("Failed to fetch team/event data from FPL API.") from exc
        return bootstrap_resp.json()

    def _get_current_csv_teams(self, csv_teams: list[str]) -> set[str]:
        bootstrap = self._fetch_bootstrap_static()
        fpl_normalized_names: set[str] = set()
        for team in bootstrap.get("teams", []):
            team_name = str(team.get("name", ""))
            short_name = str(team.get("short_name", ""))
            if team_name:
                fpl_normalized_names.add(self._normalize_team_name(team_name))
            if short_name:
                fpl_normalized_names.add(self._normalize_team_name(short_name))

        current_csv_teams = {
            team
            for team in csv_teams
            if self._normalize_team_name(team) in fpl_normalized_names
        }
        if not current_csv_teams:
            raise RuntimeError(
                "No matching current-season teams found between CSV and FPL API."
            )
        return current_csv_teams

    def _fetch_next_event_fixtures(self) -> tuple[int, list[dict], dict[int, str]]:
        try:
            fixtures_resp = requests.get(FPL_FIXTURES_URL, timeout=HTTP_TIMEOUT_SECONDS)
            fixtures_resp.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                "Failed to fetch next gameweek fixtures from FPL API."
            ) from exc

        bootstrap = self._fetch_bootstrap_static()
        fixtures = fixtures_resp.json()

        next_event = next(
            (event for event in bootstrap["events"] if event.get("is_next")), None
        )
        if next_event is None:
            raise RuntimeError("Could not find next gameweek from FPL API events.")

        next_event_id = int(next_event["id"])
        team_id_to_name = {
            int(team["id"]): str(team["short_name"]) for team in bootstrap["teams"]
        }
        next_fixtures = [
            fixture
            for fixture in fixtures
            if int(fixture.get("event") or -1) == next_event_id
        ]
        if not next_fixtures:
            raise RuntimeError(
                f"No fixtures found for next gameweek (event id {next_event_id})."
            )
        return next_event_id, next_fixtures, team_id_to_name

    def _build_next_week_player_pool(
        self, latest_players: pd.DataFrame, season_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, int]:
        next_event_id, next_fixtures, team_id_to_name = self._fetch_next_event_fixtures()

        csv_team_map = {
            self._normalize_team_name(team): team
            for team in latest_players["team"].dropna().astype(str).unique().tolist()
        }
        fpl_team_map = {
            self._normalize_team_name(name): name for name in team_id_to_name.values()
        }
        csv_to_fpl_name: dict[str, str] = {}
        for csv_norm, csv_name in csv_team_map.items():
            if csv_norm in fpl_team_map:
                csv_to_fpl_name[csv_name] = fpl_team_map[csv_norm]

        fpl_to_csv_name = {fpl: csv for csv, fpl in csv_to_fpl_name.items()}

        fixtures_by_csv_team: dict[str, list[tuple[str, int]]] = {}
        for fixture in next_fixtures:
            home_id = int(fixture["team_h"])
            away_id = int(fixture["team_a"])
            home_fpl = team_id_to_name.get(home_id)
            away_fpl = team_id_to_name.get(away_id)
            if home_fpl is None or away_fpl is None:
                continue

            home_csv = fpl_to_csv_name.get(home_fpl)
            away_csv = fpl_to_csv_name.get(away_fpl)
            if home_csv is None or away_csv is None:
                continue

            fixtures_by_csv_team.setdefault(home_csv, []).append((away_csv, 1))
            fixtures_by_csv_team.setdefault(away_csv, []).append((home_csv, 0))

        team_strength_map = self._build_team_strength_map(season_df)
        same_opp_stats_map = self._build_same_opponent_stats_map(season_df)

        expanded_rows: list[pd.Series] = []
        for _, row in latest_players.iterrows():
            team = str(row["team"])
            player_name = str(row["name"])
            fixtures_for_team = fixtures_by_csv_team.get(team, [])

            if not fixtures_for_team:
                fallback_row = row.copy()
                fallback_row["fixture_order"] = 0
                fallback_row["base_player_key"] = f"{player_name}::{team}::{row.name}"
                expanded_rows.append(fallback_row)
                continue

            for fixture_order, (opponent_team_name, was_home) in enumerate(
                fixtures_for_team
            ):
                sim_row = row.copy()
                sim_row["opponent_team_name"] = opponent_team_name
                sim_row["was_home"] = int(was_home)
                sim_row["my_team_strength"] = float(
                    team_strength_map.get(team, sim_row.get("my_team_strength", 3.0))
                )
                sim_row["opponent_strength"] = float(
                    team_strength_map.get(
                        opponent_team_name, sim_row.get("opponent_strength", 3.0)
                    )
                )
                assists_mean, goals_mean = same_opp_stats_map.get(
                    (player_name, opponent_team_name), (0.0, 0.0)
                )
                sim_row["assists_prev_3_same_opponent_mean"] = assists_mean
                sim_row["goals_scored_prev_3_same_opponent_mean"] = goals_mean
                sim_row["fixture_order"] = fixture_order
                sim_row["base_player_key"] = f"{player_name}::{team}::{row.name}"
                expanded_rows.append(sim_row)

        if not expanded_rows:
            raise RuntimeError("No player rows available after next-week fixture expansion.")

        expanded_df = pd.DataFrame(expanded_rows)
        return expanded_df, next_event_id

    def _predict_next_week_player_points(
        self,
    ) -> tuple[pd.DataFrame, int, int, int]:
        if self.artifacts is None:
            self.load()

        assert self.artifacts is not None
        raw_df = self.artifacts.raw_df.copy()

        latest_season = int(raw_df["season"].max())
        season_df = raw_df[raw_df["season"] == latest_season].copy()
        csv_teams = season_df["team"].dropna().astype(str).unique().tolist()
        current_csv_teams = self._get_current_csv_teams(csv_teams)
        season_df = season_df[season_df["team"].isin(current_csv_teams)].copy()

        if season_df.empty:
            raise RuntimeError("No player data available for current-season teams.")

        latest_gw = int(season_df["GW"].max())
        latest_players = season_df[season_df["GW"] == latest_gw].copy()
        latest_players["position_group"] = latest_players["position"].replace(
            {"GKP": "GK", "AM": "MID"}
        )

        expanded_players, next_gw = self._build_next_week_player_pool(
            latest_players=latest_players,
            season_df=season_df,
        )

        prediction_features = expanded_players.drop(
            columns=INITIAL_COLS_TO_DROP + [TARGET_COL],
            errors="ignore",
        ).reindex(columns=self.artifacts.feature_columns)

        scaled_features = pd.DataFrame(
            self.artifacts.scaler.transform(prediction_features),
            columns=self.artifacts.feature_columns,
            index=expanded_players.index,
        )

        expanded_players[SELECTION_SCORE_COL] = self.artifacts.model.predict(
            scaled_features
        )

        aggregated_players = (
            expanded_players.sort_values(["base_player_key", "fixture_order"])
            .groupby("base_player_key", as_index=False)
            .agg(
                {
                    "name": "first",
                    "position_group": "first",
                    "team": "first",
                    "opponent_team_name": lambda s: " / ".join(
                        [str(value) for value in s.tolist()]
                    ),
                    SELECTION_SCORE_COL: "sum",
                }
            )
        )

        return aggregated_players, latest_season, latest_gw, int(next_gw)

    def predict_best_xi(self) -> dict:
        aggregated_players, latest_season, _latest_gw, next_gw = (
            self._predict_next_week_player_points()
        )

        candidate_pool = (
            aggregated_players.sort_values(SELECTION_SCORE_COL, ascending=False)
            .groupby("team", group_keys=False)
            .head(MAX_PLAYERS_PER_TEAM)
            .copy()
        )

        best_xi = self._select_optimal_squad(
            players_df=candidate_pool,
            formation=FORMATION,
            max_players_per_team=MAX_PLAYERS_PER_TEAM,
            score_col=SELECTION_SCORE_COL,
        )
        best_xi["formation_slot"] = best_xi["position_group"].map(
            {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
        )
        best_xi = best_xi.sort_values(
            ["formation_slot", SELECTION_SCORE_COL], ascending=[True, False]
        )

        players = []
        for _, row in best_xi.iterrows():
            players.append(
                {
                    "name": str(row["name"]),
                    "position": str(row["position_group"]),
                    "team": str(row["team"]),
                    "opponent_team_name": str(row["opponent_team_name"]),
                    "predicted_points": self._ceil_points(row[SELECTION_SCORE_COL]),
                }
            )

        return {
            "season": int(latest_season),
            "gameweek": int(next_gw),
            "formation": FORMATION,
            "total_predicted_points": sum(
                player["predicted_points"] for player in players
            ),
            "players": players,
        }

    def get_top_players_by_total_points(
        self, limit: int = TOP_PLAYERS_DEFAULT_LIMIT
    ) -> dict:
        if self.artifacts is None:
            self.load()

        assert self.artifacts is not None
        raw_df = self.artifacts.raw_df.copy()

        latest_season = int(raw_df["season"].max())
        season_players = raw_df[raw_df["season"] == latest_season].copy()
        csv_teams = season_players["team"].dropna().astype(str).unique().tolist()
        current_csv_teams = self._get_current_csv_teams(csv_teams)
        season_players = season_players[
            season_players["team"].isin(current_csv_teams)
        ].copy()

        if season_players.empty:
            raise RuntimeError("No player data available for current-season teams.")

        latest_gw = int(season_players["GW"].max())
        season_players["position_group"] = season_players["position"].replace(
            {"GKP": "GK", "AM": "MID"}
        )

        season_totals = season_players.groupby("name", as_index=False)[TARGET_COL].sum()
        latest_player_info = (
            season_players.sort_values(["name", "GW"])
            .groupby("name", as_index=False)
            .tail(1)[["name", "position_group", "team", "opponent_team_name"]]
        )
        ranked_players = (
            season_totals.merge(latest_player_info, on="name", how="left")
            .sort_values([TARGET_COL, "name"], ascending=[False, True])
            .head(limit)
        )

        players = []
        for _, row in ranked_players.iterrows():
            players.append(
                {
                    "name": str(row["name"]),
                    "position": str(row["position_group"]),
                    "team": str(row["team"]),
                    "opponent_team_name": str(row["opponent_team_name"]),
                    "total_points": self._ceil_points(row[TARGET_COL]),
                }
            )

        return {
            "season": int(latest_season),
            "gameweek": int(latest_gw),
            "limit": int(limit),
            "players": players,
        }

    def predict_players_next_week(self, player_names: list[str]) -> dict:
        aggregated_players, latest_season, latest_gw, next_gw = (
            self._predict_next_week_player_points()
        )

        cleaned_names = [name.strip() for name in player_names]
        by_name = aggregated_players.groupby("name", as_index=False).first()
        available_names = set(by_name["name"].astype(str).tolist())

        missing_names = [name for name in cleaned_names if name not in available_names]
        if missing_names:
            missing_str = ", ".join(sorted(set(missing_names)))
            raise ValueError(
                "Some requested players were not found in the latest gameweek data: "
                f"{missing_str}"
            )

        lookup = by_name.set_index("name")
        players = []
        for name in cleaned_names:
            row = lookup.loc[name]
            players.append(
                {
                    "name": name,
                    "position": str(row["position_group"]),
                    "team": str(row["team"]),
                    "opponent_team_name": str(row["opponent_team_name"]),
                    "predicted_points": self._ceil_points(row[SELECTION_SCORE_COL]),
                }
            )

        return {
            "season": int(latest_season),
            "input_gameweek": int(latest_gw),
            "predicted_gameweek": int(next_gw),
            "total_predicted_points": sum(
                player["predicted_points"] for player in players
            ),
            "players": players,
        }

    def _select_optimal_squad(
        self,
        players_df: pd.DataFrame,
        formation: dict[str, int],
        max_players_per_team: int,
        score_col: str,
    ) -> pd.DataFrame:
        total_required = sum(formation.values())
        ranked = players_df.sort_values(score_col, ascending=False).reset_index(
            drop=True
        )
        if ranked.empty:
            raise ValueError("No candidate players found for the latest gameweek.")

        suffix_score_sum = [0.0] * (len(ranked) + 1)
        for i in range(len(ranked) - 1, -1, -1):
            suffix_score_sum[i] = suffix_score_sum[i + 1] + float(
                ranked.at[i, score_col]
            )

        positions = list(formation.keys())
        suffix_pos_counts = {pos: [0] * (len(ranked) + 1) for pos in positions}
        for i in range(len(ranked) - 1, -1, -1):
            current_pos = str(ranked.at[i, "position_group"])
            for pos in positions:
                suffix_pos_counts[pos][i] = suffix_pos_counts[pos][i + 1] + (
                    1 if current_pos == pos else 0
                )

        best_score = float("-inf")
        best_indices: list[int] = []

        def search(
            idx: int,
            remaining: dict[str, int],
            team_counts: dict[str, int],
            chosen_indices: list[int],
            current_score: float,
        ) -> None:
            nonlocal best_score, best_indices

            slots_left = sum(remaining.values())
            if slots_left == 0:
                if current_score > best_score:
                    best_score = current_score
                    best_indices = chosen_indices.copy()
                return

            if idx >= len(ranked):
                return

            if len(ranked) - idx < slots_left:
                return

            for pos, need in remaining.items():
                if suffix_pos_counts[pos][idx] < need:
                    return

            optimistic = current_score + (
                suffix_score_sum[idx] - suffix_score_sum[idx + slots_left]
            )
            if optimistic <= best_score:
                return

            row = ranked.iloc[idx]
            pos = str(row["position_group"])
            team = str(row["team"])
            score = float(row[score_col])

            if (
                remaining.get(pos, 0) > 0
                and team_counts.get(team, 0) < max_players_per_team
            ):
                remaining[pos] -= 1
                team_counts[team] = team_counts.get(team, 0) + 1
                chosen_indices.append(idx)

                search(
                    idx + 1,
                    remaining,
                    team_counts,
                    chosen_indices,
                    current_score + score,
                )

                chosen_indices.pop()
                team_counts[team] -= 1
                if team_counts[team] == 0:
                    team_counts.pop(team)
                remaining[pos] += 1

            search(idx + 1, remaining, team_counts, chosen_indices, current_score)

        search(
            idx=0,
            remaining=formation.copy(),
            team_counts={},
            chosen_indices=[],
            current_score=0.0,
        )

        if len(best_indices) != total_required:
            raise ValueError(
                "Unable to build a valid lineup with current constraints: "
                f"formation={formation}, max_players_per_team={max_players_per_team}."
            )

        return ranked.iloc[best_indices].copy()
