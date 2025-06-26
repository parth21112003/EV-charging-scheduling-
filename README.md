# Smart EV Charging Scheduler

This project implements an AI-based solution for optimizing electric vehicle (EV) charging schedules to prevent grid congestion and manage peak demand efficiently.

## Features

- Time series forecasting using multiple models (Holt-Winters, LSTM)
- Real-time demand monitoring and prediction
- Charging schedule optimization
- Grid load visualization
- Smart threshold-based recommendations

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the main script:
```bash
python ev_charging_scheduler.py
```

## Project Structure

- `ev_charging_scheduler.py`: Main script with the EV charging optimization logic
- `data_generator.py`: Utility to generate sample EV charging demand data
- `models/`: Directory containing different forecasting models
- `utils/`: Helper functions and utilities
- `config.py`: Configuration parameters

## How It Works

1. The system collects historical EV charging demand data
2. Applies time series forecasting to predict future demand
3. Sets dynamic thresholds based on grid capacity
4. Generates optimized charging schedules
5. Provides real-time recommendations for EV charging

## License

MIT License 