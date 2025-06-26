import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_synthetic_data(days=30, noise_level=0.2):
    """
    Generate synthetic EV charging demand data with daily and weekly patterns.
    
    Args:
        days (int): Number of days to generate data for
        noise_level (float): Amount of random noise to add
        
    Returns:
        pd.DataFrame: DataFrame with timestamp and demand columns
    """
    # Generate hourly timestamps
    timestamps = [datetime.now() - timedelta(days=days) + timedelta(hours=x) 
                 for x in range(days * 24)]
    
    # Base daily pattern (24 hours)
    daily_pattern = np.array([
        0.2, 0.15, 0.1, 0.1, 0.15, 0.3,    # 00:00 - 06:00
        0.5, 0.7, 0.8, 0.7, 0.6, 0.5,      # 06:00 - 12:00
        0.4, 0.4, 0.5, 0.6, 0.8, 0.9,      # 12:00 - 18:00
        1.0, 0.9, 0.7, 0.5, 0.3, 0.2       # 18:00 - 00:00
    ])
    
    # Weekly pattern modifier
    weekly_pattern = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.7, 0.6])  # Mon-Sun
    
    # Generate demand data
    demand = []
    for i, timestamp in enumerate(timestamps):
        day_of_week = timestamp.weekday()
        hour = timestamp.hour
        
        # Combine daily and weekly patterns
        base_demand = daily_pattern[hour] * weekly_pattern[day_of_week]
        
        # Add random noise
        noise = np.random.normal(0, noise_level)
        final_demand = max(0, base_demand + noise)
        
        demand.append(final_demand)
    
    # Scale demand to realistic values (in kW)
    demand = np.array(demand) * 500  # Scale to max ~500 kW
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'demand_kw': demand
    })
    
    return df

def save_sample_data(filename='ev_charging_data.csv', days=30):
    """
    Generate and save sample data to CSV file.
    """
    df = generate_synthetic_data(days=days)
    df.to_csv(filename, index=False)
    print(f"Sample data saved to {filename}")

if __name__ == "__main__":
    save_sample_data() 