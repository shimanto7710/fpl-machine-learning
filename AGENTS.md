# FPL Best XI Prediction System

This is a machine learning service that predicts optimal Fantasy Premier League (FPL) lineups using XGBoost regression on historical player data.

## Build & Run Commands

- **Train model**: `python train_artifacts.py` (generates artifacts/)
- **Run API locally**: `uvicorn main:app --host 0.0.0.0 --port 8000`
- **Docker build/run**: `docker build -t fpl-api . && docker run -p 8000:8000 fpl-api`
- **Notebook exploration**: Execute [fpl_project.ipynb](fpl_project.ipynb) cells for data analysis and model training

## Architecture

- **API Layer**: [main.py](main.py) - FastAPI endpoint `/api/v1/predictions/best-xi`
- **Service Layer**: [service.py](service.py) - Prediction logic and lineup selection
- **Training**: [train_artifacts.py](train_artifacts.py) - ML pipeline for model/scaler generation
- **Artifacts**: [artifacts/](artifacts/) - Pre-trained XGBRegressor, StandardScaler, feature metadata
- **Data**: [fpl_final.csv](fpl_final.csv) - Historical gameweek player statistics

Uses 4-3-3 formation with position normalization (GKP→GK, AM→MID). Predicts for latest gameweek in CSV.

## Conventions

- Constants in CAPS (e.g., `FORMATION`, `TARGET_COL`)
- Full type hints with union syntax
- Dataclasses for config (`ArtifactPaths`) and state (`ModelArtifacts`)
- Feature engineering: rolling means for influence/creativity/threat/goals/assists
- Error handling with descriptive messages
- Async lifespan context for artifact loading

## Potential Pitfalls

- Run training before API deployment (artifacts required)
- Ensure CSV columns match feature metadata
- Position normalization only handles GKP/AM; others assumed valid
- Working directory must contain artifacts/ and fpl_final.csv
- No pinned versions in requirements.txt

## Key Patterns

- [service.py](service.py): Feature processing, model prediction, optimal lineup selection via pandas groupby/sort
- [train_artifacts.py](train_artifacts.py): ML training pipeline with hyperparameter tuning
- [main.py](main.py): Async dependency injection, Pydantic validation
- [fpl_project.ipynb](fpl_project.ipynb): EDA, feature importance, model evaluation

For detailed analysis and visualizations, see [fpl_project.ipynb](fpl_project.ipynb).</content>
<parameter name="filePath">/Users/shimantoa./Documents/Data Cleaning/updated_fpl_project/AGENTS.md