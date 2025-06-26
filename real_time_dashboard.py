import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from flask import Flask, render_template, jsonify
import threading
import time
import os
import pickle
import json
from pre_booking_manager import PreBookingManager
import config
from data_generator import generate_synthetic_data

class RealTimeEVOptimizer:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.data = None
        self.model_hw = None
        self.model_lstm = None
        self.charging_stations = {}
        self.pre_booking_manager = PreBookingManager()
        self.last_update = None
        self.current_forecast = None
        self.current_schedule = None
        self.peak_hours = []
        
        # Model cache paths
        self.model_cache_dir = 'model_cache'
        os.makedirs(self.model_cache_dir, exist_ok=True)
        
    def load_or_train_models(self):
        """Load cached models or train new ones if cache doesn't exist"""
        hw_cache_path = os.path.join(self.model_cache_dir, 'holtwinters_model.pkl')
        lstm_cache_path = os.path.join(self.model_cache_dir, 'lstm_model.h5')
        scaler_cache_path = os.path.join(self.model_cache_dir, 'scaler.pkl')
        
        # Check if cached models exist and are recent (less than 24 hours old)
        cache_valid = False
        if (os.path.exists(hw_cache_path) and os.path.exists(lstm_cache_path) and 
            os.path.exists(scaler_cache_path)):
            cache_age = time.time() - os.path.getmtime(hw_cache_path)
            cache_valid = cache_age < 86400  # 24 hours in seconds
        
        if cache_valid:
            print("Loading cached models...")
            try:
                # Load Holt-Winters model
                with open(hw_cache_path, 'rb') as f:
                    self.model_hw = pickle.load(f)
                
                # Load LSTM model
                self.model_lstm = tf.keras.models.load_model(lstm_cache_path)
                
                # Load scaler
                with open(scaler_cache_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                
                print("Cached models loaded successfully!")
                return True
            except Exception as e:
                print(f"Error loading cached models: {e}")
                print("Will train new models...")
        
        # Train new models
        print("Training new models...")
        self.load_data()
        self.preprocess_data()
        self.train_holtwinters()
        self.train_lstm()
        
        # Cache the models
        try:
            # Save Holt-Winters model
            with open(hw_cache_path, 'wb') as f:
                pickle.dump(self.model_hw, f)
            
            # Save LSTM model
            self.model_lstm.save(lstm_cache_path)
            
            # Save scaler
            with open(scaler_cache_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            print("Models cached successfully!")
        except Exception as e:
            print(f"Error caching models: {e}")
         
        return True

    def load_data(self):
        """Load and prepare data"""
        if not os.path.exists('data.csv'):
            print("Generating synthetic data...")
            generate_synthetic_data()
        
        self.data = pd.read_csv('data.csv')
        
        # Handle timestamp columns more robustly
        timestamp_columns = ['timestamp', 'doneChargingTime', 'disconnectTime']
        for col in timestamp_columns:
            if col in self.data.columns:
                try:
                    self.data[col] = pd.to_datetime(self.data[col], errors='coerce')
                except Exception as e:
                    print(f"Warning: Could not parse {col} column: {e}")
        
        # Drop rows with missing timestamps
        if 'timestamp' in self.data.columns:
            self.data.dropna(subset=["timestamp"], inplace=True)
            if len(self.data) == 0:
                print("Error: No valid timestamp data found. Generating synthetic data.")
                self._generate_synthetic_data()
                return self.data
        
        # Process charging windows if timestamp columns exist
        if all(col in self.data.columns for col in ['timestamp', 'doneChargingTime']):
            self.data["session_start"] = self.data["timestamp"].dt.floor("h")
            self.data["session_end"] = self.data["doneChargingTime"].dt.ceil("h")
            self.data["charging_window_hours"] = (self.data["session_end"] - self.data["session_start"]).dt.total_seconds() / 3600
            self.data = self.data[self.data["charging_window_hours"] > 0]

            def generate_slots(row):
                return [row["session_start"] + timedelta(hours=i) for i in range(int(row["charging_window_hours"]))]

            self.data["hourly_slots"] = self.data.apply(generate_slots, axis=1)
        
        # Set timestamp as index for time series analysis
        if 'timestamp' in self.data.columns:
            self.data.set_index('timestamp', inplace=True)
            self.data.sort_index(inplace=True)
        else:
            # If no timestamp column, create one
            print("No timestamp column found. Creating synthetic timestamps.")
            self._generate_synthetic_data()
            return self.data
        
        return self.data
    
    def _generate_synthetic_data(self):
        """Generate synthetic data when real data is not available"""
        print("Generating synthetic EV charging data...")
        
        # Generate 30 days of hourly data
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        timestamps = pd.date_range(start=start_date, end=end_date, freq='H')
        
        np.random.seed(42)
        hours = len(timestamps)
        
        # Create realistic demand pattern
        time_index = np.arange(hours)
        base_demand = 200  # Base demand in kW
        daily_cycle = 100 * np.sin(2 * np.pi * time_index / 24)  # Daily cycle
        weekly_cycle = 50 * np.sin(2 * np.pi * time_index / (24 * 7))  # Weekly cycle
        noise = np.random.normal(0, 20, hours)  # Random noise
        demand_kw = base_demand + daily_cycle + weekly_cycle + noise
        
        # Create synthetic data
        self.data = pd.DataFrame({
            'demand_kw': demand_kw,
            'sessionID': [f'SESS_{i:06d}' for i in range(hours)],
            'stationID': [f'ST_{i%10:03d}' for i in range(hours)],
            'userID': np.random.randint(1000, 9999, hours)
        }, index=timestamps)
        
        print(f"Generated {len(self.data)} hours of synthetic data")
        return self.data

    def preprocess_data(self):
        """Normalize the data and prepare for modeling"""
        # Only keep numeric columns for resampling
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        if 'demand_kw' not in numeric_columns:
            # If demand_kw is not numeric, try to convert it
            self.data['demand_kw'] = pd.to_numeric(self.data['demand_kw'], errors='coerce')
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        
        # Create a copy with only numeric columns for resampling
        numeric_data = self.data[numeric_columns].copy()
        
        if not self.data.index.freq:
            # Resample only numeric columns
            numeric_data = numeric_data.resample('h').mean().ffill()
            # Update the original data with resampled numeric columns
            for col in numeric_columns:
                self.data[col] = numeric_data[col]
        
        # Ensure demand_kw exists and is numeric
        if 'demand_kw' not in self.data.columns or self.data['demand_kw'].isna().all():
            print("Warning: No valid demand_kw data found. Using synthetic data.")
            self._generate_synthetic_data()
        
        # If after all this, we still don't have enough data, generate synthetic data
        if self.data['demand_kw'].dropna().shape[0] < config.SEASONALITY * 2:
            print("Not enough data for Holt-Winters. Generating synthetic data for 30 days.")
            self._generate_synthetic_data()
        
        # Normalize the demand data
        self.data['demand_normalized'] = self.scaler.fit_transform(
            self.data['demand_kw'].values.reshape(-1, 1)
        )
        return self.data

    def train_holtwinters(self):
        """Train Holt-Winters model"""
        self.model_hw = ExponentialSmoothing(
            self.data['demand_normalized'],
            seasonal_periods=config.SEASONALITY,
            trend='add',
            seasonal='add'
        ).fit()
        return self.model_hw

    def prepare_lstm_data(self, sequence_length=config.SEQUENCE_LENGTH):
        """Prepare data for LSTM model"""
        data = self.data['demand_normalized'].values
        X, y = [], []

        for i in range(len(data) - sequence_length):
            X.append(data[i:(i + sequence_length)])
            y.append(data[i + sequence_length])

        return np.array(X), np.array(y)

    def train_lstm(self):
        """Train LSTM model with reduced epochs for faster training"""
        X, y = self.prepare_lstm_data()
        split = int(len(X) * config.TRAIN_TEST_SPLIT)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        self.model_lstm = tf.keras.Sequential([
            tf.keras.layers.LSTM(config.LSTM_UNITS, input_shape=(config.SEQUENCE_LENGTH, 1)),
            tf.keras.layers.Dense(1)
        ])

        self.model_lstm.compile(optimizer='adam', loss='mse')
        self.model_lstm.fit(
            X_train, y_train,
            epochs=20,  # Reduced from 50 for faster training
            batch_size=config.BATCH_SIZE,
            validation_data=(X_test, y_test),
            verbose=0
        )
        return self.model_lstm

    def forecast_demand(self, horizon=24):
        """Generate forecasts using both models"""
        # Holt-Winters forecast
        hw_forecast = self.model_hw.forecast(horizon)
        hw_forecast = self.scaler.inverse_transform(
            np.array(hw_forecast).reshape(-1, 1)
        ).flatten()

        # LSTM forecast
        last_sequence = self.data['demand_normalized'].values[-config.SEQUENCE_LENGTH:]
        lstm_forecast = []

        for _ in range(horizon):
            next_pred = self.model_lstm.predict(
                last_sequence.reshape(1, config.SEQUENCE_LENGTH, 1),
                verbose=0
            )
            lstm_forecast.append(next_pred[0, 0])
            last_sequence = np.roll(last_sequence, -1)
            last_sequence[-1] = next_pred[0, 0]

        lstm_forecast = self.scaler.inverse_transform(
            np.array(lstm_forecast).reshape(-1, 1)
        ).flatten()

        # Combine forecasts (weighted average favoring LSTM for recent patterns)
        combined_forecast = 0.3 * hw_forecast + 0.7 * lstm_forecast

        return combined_forecast

    def identify_peak_hours(self, forecast):
        """Identify peak hours based on forecast"""
        # Calculate moving average to smooth out noise
        window_size = 3
        smoothed_forecast = pd.Series(forecast).rolling(window=window_size, center=True).mean().ffill().bfill()
        
        # Find peaks (hours with demand above 75th percentile)
        threshold = np.percentile(smoothed_forecast, 75)
        peak_indices = np.where(smoothed_forecast > threshold)[0]
        
        # Group consecutive peak hours
        peak_hours = []
        if len(peak_indices) > 0:
            start_idx = peak_indices[0]
            end_idx = peak_indices[0]
            
            for i in range(1, len(peak_indices)):
                if peak_indices[i] == peak_indices[i-1] + 1:
                    end_idx = peak_indices[i]
                else:
                    peak_hours.append((start_idx, end_idx))
                    start_idx = peak_indices[i]
                    end_idx = peak_indices[i]
            
            peak_hours.append((start_idx, end_idx))
        
        return peak_hours, smoothed_forecast

    def optimize_charging_schedule(self, forecast):
        """Generate optimized charging schedule based on forecasts and peak hours"""
        # Initialize schedule for next 24 hours
        current_time = datetime.now()
        schedule = pd.DataFrame()
        schedule['timestamp'] = pd.date_range(
            start=current_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1),
            periods=len(forecast),
            freq='h'
        )
        
        # Get confirmed bookings for the forecast period
        confirmed_bookings = self.pre_booking_manager.get_confirmed_bookings()
        confirmed_bookings = confirmed_bookings[
            (confirmed_bookings['start_time'] >= schedule['timestamp'].iloc[0]) &
            (confirmed_bookings['end_time'] <= schedule['timestamp'].iloc[-1])
        ]
        
        # Initialize charging power for each hour
        schedule['forecasted_demand'] = forecast
        schedule['available_capacity'] = config.GRID_CAPACITY * config.SAFETY_MARGIN - forecast
        schedule['charging_power'] = 0
        schedule['is_peak_hour'] = False
        
        # Mark peak hours
        peak_hours, _ = self.identify_peak_hours(forecast)
        for start_hour, end_hour in peak_hours:
            for i in range(start_hour, min(end_hour + 1, len(schedule))):
                if i < len(schedule):
                    schedule.loc[i, 'is_peak_hour'] = True
        
        # Allocate charging power to bookings (avoid peak hours when possible)
        for _, booking in confirmed_bookings.iterrows():
            booking_hours = schedule[
                (schedule['timestamp'] >= booking['start_time']) &
                (schedule['timestamp'] < booking['end_time'])
            ]
            
            if len(booking_hours) > 0:
                # Calculate required power per hour
                required_power = booking['required_charge'] / len(booking_hours)
                
                # Prioritize non-peak hours
                non_peak_hours = booking_hours[~booking_hours['is_peak_hour']]
                peak_hours_in_booking = booking_hours[booking_hours['is_peak_hour']]
                
                # Allocate to non-peak hours first
                for idx in non_peak_hours.index:
                    if schedule.loc[idx, 'available_capacity'] > 0:
                        power_to_allocate = min(
                            required_power,
                            schedule.loc[idx, 'available_capacity'],
                            config.MAX_CHARGING_POWER
                        )
                        schedule.loc[idx, 'charging_power'] += power_to_allocate
                        schedule.loc[idx, 'available_capacity'] -= power_to_allocate
                
                # If still need power, allocate to peak hours
                remaining_power = required_power * len(booking_hours) - schedule.loc[booking_hours.index, 'charging_power'].sum()
                if remaining_power > 0:
                    for idx in peak_hours_in_booking.index:
                        if schedule.loc[idx, 'available_capacity'] > 0 and remaining_power > 0:
                            power_to_allocate = min(
                                remaining_power,
                                schedule.loc[idx, 'available_capacity'],
                                config.MAX_CHARGING_POWER
                            )
                            schedule.loc[idx, 'charging_power'] += power_to_allocate
                            schedule.loc[idx, 'available_capacity'] -= power_to_allocate
                            remaining_power -= power_to_allocate
        
        return schedule

    def update_forecast_and_schedule(self):
        """Update forecast and schedule with current data"""
        try:
            # Generate new forecast
            forecast = self.forecast_demand(24)
            
            # Optimize schedule
            schedule = self.optimize_charging_schedule(forecast)
            
            # Identify peak hours
            peak_hours, smoothed_forecast = self.identify_peak_hours(forecast)
            
            # Update current state
            self.current_forecast = forecast
            self.current_schedule = schedule
            self.peak_hours = peak_hours
            self.last_update = datetime.now()
            
            print(f"Forecast updated at {self.last_update.strftime('%Y-%m-%d %H:%M:%S')}")
            return True
        except Exception as e:
            print(f"Error updating forecast: {e}")
            return False

    def get_dashboard_data(self):
        """Get data for the dashboard"""
        if self.current_forecast is None or self.current_schedule is None:
            return None
        
        current_time = datetime.now()
        future_times = [
            current_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=i+1)
            for i in range(len(self.current_forecast))
        ]
        
        # Ensure forecast is a numpy array
        forecast_arr = np.array(self.current_forecast)

        # Calculate grid utilization as average percentage
        grid_utilization_series = (forecast_arr + self.current_schedule['charging_power']) / (config.GRID_CAPACITY * config.SAFETY_MARGIN) * 100
        avg_grid_utilization = float(np.mean(grid_utilization_series))

        # Generate peak hours prediction data
        peak_prediction_data = self.generate_peak_hours_prediction(forecast_arr, future_times)

        # Ensure last_update is a proper datetime object
        last_update_str = None
        if self.last_update is not None:
            if hasattr(self.last_update, 'strftime'):
                last_update_str = self.last_update.strftime('%Y-%m-%d %H:%M:%S')
            else:
                last_update_str = str(self.last_update)

        # Convert peak_hours to list of lists of ints
        peak_hours_serializable = [[int(start), int(end)] for start, end in self.peak_hours]

        # Prepare data for visualization
        dashboard_data = {
            'timestamps': [t.strftime('%Y-%m-%d %H:%M') for t in future_times],
            'forecast': [float(x) for x in forecast_arr.tolist()],
            'charging_power': [float(x) for x in self.current_schedule['charging_power'].tolist()],
            'available_capacity': [float(x) for x in self.current_schedule['available_capacity'].tolist()],
            'is_peak_hour': [bool(x) for x in self.current_schedule['is_peak_hour'].tolist()],
            'peak_hours': peak_hours_serializable,
            'peak_prediction': peak_prediction_data,
            'last_update': last_update_str,
            'summary': {
                'max_forecast': float(np.max(forecast_arr)),
                'min_forecast': float(np.min(forecast_arr)),
                'avg_forecast': float(np.mean(forecast_arr)),
                'peak_hours_count': int(len(self.peak_hours)),
                'total_charging_power': float(self.current_schedule['charging_power'].sum()),
                'grid_utilization': avg_grid_utilization
            }
        }
        
        return dashboard_data

    def generate_peak_hours_prediction(self, forecast, timestamps):
        """Generate detailed peak hours prediction data for visualization"""
        # Calculate moving average for smoothing
        window_size = 3
        smoothed_forecast = pd.Series(forecast).rolling(window=window_size, center=True).mean().ffill().bfill()
        
        # Find peaks using different thresholds
        thresholds = {
            'high': float(np.percentile(smoothed_forecast, 85)),
            'medium': float(np.percentile(smoothed_forecast, 75)),
            'low': float(np.percentile(smoothed_forecast, 65))
        }
        
        peak_data = {
            'timestamps': [t.strftime('%H:%M') for t in timestamps],
            'forecast': [float(x) for x in forecast.tolist()],
            'smoothed_forecast': [float(x) for x in smoothed_forecast.tolist()],
            'thresholds': thresholds,
            'peak_zones': {}
        }
        
        # Identify peak zones for each threshold
        for level, threshold in thresholds.items():
            peak_indices = np.where(smoothed_forecast > threshold)[0]
            peak_zones = []
            
            if len(peak_indices) > 0:
                start_idx = int(peak_indices[0])
                end_idx = int(peak_indices[0])
                
                for i in range(1, len(peak_indices)):
                    if int(peak_indices[i]) == int(peak_indices[i-1]) + 1:
                        end_idx = int(peak_indices[i])
                    else:
                        peak_zones.append({
                            'start_hour': int(start_idx),
                            'end_hour': int(end_idx),
                            'start_time': timestamps[start_idx].strftime('%H:%M'),
                            'end_time': timestamps[end_idx].strftime('%H:%M'),
                            'duration': int(end_idx - start_idx + 1),
                            'max_demand': float(np.max(forecast[start_idx:end_idx+1]))
                        })
                        start_idx = int(peak_indices[i])
                        end_idx = int(peak_indices[i])
                
                peak_zones.append({
                    'start_hour': int(start_idx),
                    'end_hour': int(end_idx),
                    'start_time': timestamps[start_idx].strftime('%H:%M'),
                    'end_time': timestamps[end_idx].strftime('%H:%M'),
                    'duration': int(end_idx - start_idx + 1),
                    'max_demand': float(np.max(forecast[start_idx:end_idx+1]))
                })
            
            peak_data['peak_zones'][level] = peak_zones
        
        return peak_data

# Flask app for real-time dashboard
app = Flask(__name__)
optimizer = RealTimeEVOptimizer()

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/data')
def get_data():
    """API endpoint to get current forecast and schedule data"""
    data = optimizer.get_dashboard_data()
    if data is None:
        return jsonify({'error': 'No data available'}), 404
    return jsonify(data)

@app.route('/api/update')
def update_data():
    """API endpoint to trigger manual update"""
    success = optimizer.update_forecast_and_schedule()
    if success:
        return jsonify({'status': 'success', 'message': 'Data updated successfully'})
    else:
        return jsonify({'status': 'error', 'message': 'Failed to update data'}), 500

def background_updater():
    """Background thread to update data every 15 minutes"""
    while True:
        try:
            optimizer.update_forecast_and_schedule()
            time.sleep(900)  # 15 minutes
        except Exception as e:
            print(f"Background update error: {e}")
            time.sleep(60)  # Wait 1 minute before retrying

# Configure TensorFlow for GPU if available
def configure_gpu():
    """Configure TensorFlow to use GPU if available"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU acceleration enabled: {len(gpus)} GPU(s) found")
            return True
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
            return False
    else:
        print("No GPU found. Using CPU for computation.")
        return False

# Configure GPU at startup
GPU_AVAILABLE = configure_gpu()

if __name__ == '__main__':
    # Initialize models
    print("Initializing EV Charging Optimizer...")
    optimizer.load_or_train_models()
    
    # Load sample pre-bookings
    print("Loading pre-bookings...")
    optimizer.pre_booking_manager.generate_sample_pre_bookings()
    
    # Generate initial forecast
    print("Generating initial forecast...")
    optimizer.update_forecast_and_schedule()
    
    # Start background update thread
    print("Starting background update thread...")
    update_thread = threading.Thread(target=background_updater, daemon=True)
    update_thread.start()
    
    # Start Flask app
    print("Starting real-time dashboard...")
    print("Dashboard will be available at: http://localhost:5000")
    print("The dashboard will automatically update every 15 minutes")
    print("Press Ctrl+C to stop the server")
    
    app.run(debug=False, host='0.0.0.0', port=5000) 