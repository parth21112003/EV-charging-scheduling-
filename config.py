# config.py

# ⚡ Grid Configuration 
GRID_CAPACITY = 1000       # Maximum grid capacity in kW
SAFETY_MARGIN = 0.8        # Grid safety threshold (80% of max capacity)

# 📈 Time Series Forecasting Parameters
FORECAST_HORIZON = 24      # Hours to forecast ahead
SEASONALITY = 24           # Daily seasonality (24-hour cycles)
TRAIN_TEST_SPLIT = 0.8

# ⚙️ Optimization Parameters
MAX_CHARGING_POWER = 100   # Max power per vehicle/session (kW)
MIN_CHARGING_POWER = 10    # Min power per vehicle/session (kW)
TIME_SLOTS = 24            # Number of time slots in a day

# 🧠 LSTM Model Parameters
SEQUENCE_LENGTH = 24
LSTM_UNITS = 50
EPOCHS = 50
BATCH_SIZE = 32

# 📊 Visualization
PLOT_STYLE = 'seaborn'
FIGURE_SIZE = (12, 6)

# 🔋 Default charging behavior
DEFAULT_CHARGING_RATE = 50  # kW
