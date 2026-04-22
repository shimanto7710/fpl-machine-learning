from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from service import FPLPredictionService


class PlayerPrediction(BaseModel):
    name: str
    position: str
    team: str
    opponent_team_name: str
    predicted_points: int


class LineupPredictionResponse(BaseModel):
    season: int
    gameweek: int
    formation: dict[str, int]
    total_predicted_points: int
    players: list[PlayerPrediction]


class PlayerTotalPoints(BaseModel):
    name: str
    position: str
    team: str
    opponent_team_name: str
    total_points: int


class TopPlayersTotalPointsResponse(BaseModel):
    season: int
    gameweek: int
    limit: int
    players: list[PlayerTotalPoints]


class PlayerListPredictionRequest(BaseModel):
    players: list[str] = Field(
        min_length=15,
        max_length=15,
        description="Exactly 15 player names from latest gameweek data.",
    )


class PlayerListPredictionResponse(BaseModel):
    season: int
    input_gameweek: int
    predicted_gameweek: int
    total_predicted_points: int
    players: list[PlayerPrediction]


service = FPLPredictionService()


@asynccontextmanager
async def lifespan(_: FastAPI):
    service.load()
    yield


app = FastAPI(
    title="FPL Model API",
    version="0.1.0",
    description="FastAPI wrapper for the FPL best-XI prediction model.",
    lifespan=lifespan,
)


@app.get("/api/v1/predictions/best-xi", response_model=LineupPredictionResponse)
def get_best_xi() -> LineupPredictionResponse:
    return LineupPredictionResponse(**service.predict_best_xi())


@app.get(
    "/api/v1/players/top-total-points",
    response_model=TopPlayersTotalPointsResponse,
)
def get_top_players_total_points(
    limit: int = Query(50, ge=1, le=500),
) -> TopPlayersTotalPointsResponse:
    return TopPlayersTotalPointsResponse(
        **service.get_top_players_by_total_points(limit=limit)
    )


@app.post(
    "/api/v1/predictions/players",
    response_model=PlayerListPredictionResponse,
)
def predict_players_next_week(
    request: PlayerListPredictionRequest,
) -> PlayerListPredictionResponse:
    try:
        result = service.predict_players_next_week(request.players)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PlayerListPredictionResponse(**result)
