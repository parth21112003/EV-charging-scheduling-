import pandas as pd
from datetime import datetime, timedelta
import random

class PreBookingManager:
    def __init__(self):
        self.pre_bookings = pd.DataFrame(columns=[
            'vehicle_id', 'station_id', 'start_time', 'end_time', 
            'required_charge', 'status'
        ])

    def generate_sample_pre_bookings(self, num_bookings=50, days_ahead=7):
        """
        Generate sample pre-bookings for testing
        """
        bookings = []
        current_time = datetime.now()
        
        # Generate random vehicle and station IDs
        vehicle_ids = [f'EV{i:03d}' for i in range(1, 21)]  # 20 vehicles
        station_ids = [f'ST{i:03d}' for i in range(1, 11)]  # 10 stations
        
        for _ in range(num_bookings):
            # Random start time within the next 'days_ahead' days
            start_time = current_time + timedelta(
                days=random.randint(0, days_ahead),
                hours=random.randint(0, 23)
            )
            
            # Random charging duration between 1 and 4 hours
            duration = random.randint(1, 4)
            end_time = start_time + timedelta(hours=duration)
            
            # Random required charge between 20 and 80 kWh
            required_charge = random.randint(20, 80)
            
            booking = {
                'vehicle_id': random.choice(vehicle_ids),
                'station_id': random.choice(station_ids),
                'start_time': start_time,
                'end_time': end_time,
                'required_charge': required_charge,
                'status': 'pending'
            }
            bookings.append(booking)
        
        self.pre_bookings = pd.DataFrame(bookings)
        return self.pre_bookings

    def add_pre_booking(self, vehicle_id, station_id, start_time, end_time, required_charge):
        """
        Add a new pre-booking to the system
        """
        if isinstance(start_time, str):
            start_time = pd.to_datetime(start_time)
        if isinstance(end_time, str):
            end_time = pd.to_datetime(end_time)
            
        new_booking = {
            'vehicle_id': vehicle_id,
            'station_id': station_id,
            'start_time': start_time,
            'end_time': end_time,
            'required_charge': required_charge,
            'status': 'pending'
        }
        
        self.pre_bookings = pd.concat([self.pre_bookings, pd.DataFrame([new_booking])], ignore_index=True)
        return True, "Booking added successfully"

    def load_pre_bookings_from_csv(self, file_path):
        """
        Load pre-bookings from a CSV file
        """
        try:
            self.pre_bookings = pd.read_csv(
                file_path,
                parse_dates=['start_time', 'end_time']
            )
            return True, "Pre-bookings loaded successfully"
        except Exception as e:
            return False, f"Error loading pre-bookings: {str(e)}"

    def save_pre_bookings_to_csv(self, file_path):
        """
        Save current pre-bookings to a CSV file
        """
        try:
            self.pre_bookings.to_csv(file_path, index=False)
            return True, "Pre-bookings saved successfully"
        except Exception as e:
            return False, f"Error saving pre-bookings: {str(e)}"

    def get_confirmed_bookings(self):
        """
        Get all confirmed bookings
        """
        return self.pre_bookings[self.pre_bookings['status'] == 'confirmed']

    def get_pending_bookings(self):
        """
        Get all pending bookings
        """
        return self.pre_bookings[self.pre_bookings['status'] == 'pending']

    def get_bookings_by_station(self, station_id):
        """
        Get all bookings for a specific station
        """
        return self.pre_bookings[self.pre_bookings['station_id'] == station_id]

    def get_bookings_by_vehicle(self, vehicle_id):
        """
        Get all bookings for a specific vehicle
        """
        return self.pre_bookings[self.pre_bookings['vehicle_id'] == vehicle_id]

    def update_booking_status(self, booking_index, new_status):
        """
        Update the status of a specific booking
        """
        if booking_index in self.pre_bookings.index:
            self.pre_bookings.loc[booking_index, 'status'] = new_status
            return True, "Booking status updated successfully"
        else:
            return False, "Booking not found"

    def get_booking_summary(self):
        """
        Get a summary of all bookings
        """
        summary = {
            'total_bookings': len(self.pre_bookings),
            'confirmed_bookings': len(self.get_confirmed_bookings()),
            'pending_bookings': len(self.get_pending_bookings()),
            'unique_vehicles': self.pre_bookings['vehicle_id'].nunique(),
            'unique_stations': self.pre_bookings['station_id'].nunique()
        }
        return summary

if __name__ == "__main__":
    manager = PreBookingManager()
    bookings = manager.generate_sample_pre_bookings()
    print(bookings.head())