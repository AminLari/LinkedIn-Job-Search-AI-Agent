```markdown
# Lime Micromobility Demand Forecasting Demo

## Project: Real-Time Demand Forecasting for Lime Scooters

### Files
**data/raw/`
- `lime_demand_sample.csv`: Synthetic dataset with historical scooter demand (timestamp, location, rider count, weather conditions, time of day, day of week, event flags)
- `geo_data.json`: Sample geospatial coordinates for Lime hub locations

**models/`
- `forecast_model.pkl`: Trained PyTorch model for time-series demand prediction
- `feature_engineering.py`: Feature extraction pipeline for demand forecasting

**scripts/`
- `train_model.py`: Script to train and evaluate the demand forecasting model
- `deploy_forecast.py`: Simulated deployment script for real-time predictions
- `monitor_model.py`: Basic model monitoring script for drift detection

**notebooks/`
- `demand_analysis.ipynb`: Exploratory analysis of demand patterns
- `feature_importance.ipynb`: Feature importance analysis for the model

### How to Run
1. Install dependencies:
```bash
pip install torch pandas numpy scikit-learn geopandas
```

2. Train the model:
```bash
python scripts/train_model.py --data-path data/raw/lime_demand_sample.csv --output models/forecast_model.pkl
```

3. Simulate real-time predictions:
```bash
python scripts/deploy_forecast.py --model models/forecast_model.pkl --geo-data data/raw/geo_data.json
```

4. Monitor model performance:
```bash
python scripts/monitor_model.py --baseline models/forecast_model.pkl --data data/raw/lime_demand_sample.csv
```

### Key Features
- Time-series forecasting with PyTorch
- Geospatial-aware demand modeling
- Feature engineering for operational relevance
- Basic MLOps components (model versioning, monitoring)
- Scalable architecture for Lime's dynamic deployment needs

### Impact
This demo demonstrates how Lime could optimize scooter positioning by:
- Predicting demand spikes in high-traffic areas
- Dynamically reallocating vehicles based on real-time forecasts
- Reducing idle time and improving rider experience
```