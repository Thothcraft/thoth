import json
import random
from datetime import datetime, timedelta
import os

def generate_mock_reading(timestamp):
    return {
        "timestamp": timestamp.isoformat(),
        "imu": {
            "accel": {
                "x": round(random.uniform(-2.0, 2.0), 6),
                "y": round(random.uniform(-2.0, 2.0), 6),
                "z": round(random.uniform(8.0, 10.0), 6)  # Gravity
            },
            "gyro": {
                "x": round(random.uniform(-0.1, 0.1), 6),
                "y": round(random.uniform(-0.1, 0.1), 6),
                "z": round(random.uniform(-0.1, 0.1), 6)
            },
            "mag": {
                "x": round(random.uniform(-100, 100), 6),
                "y": round(random.uniform(-100, 100), 6),
                "z": round(random.uniform(-100, 100), 6)
            }
        },
        "sensor_fusion": {
            "pitch": round(random.uniform(-180, 180), 2),
            "roll": round(random.uniform(-90, 90), 2),
            "yaw": round(random.uniform(0, 360), 2)
        },
        "device": {
            "battery": round(random.uniform(3.6, 4.2), 2),
            "temp": round(random.uniform(20.0, 40.0), 1),
            "rssi": random.randint(-70, -30)
        }
    }

def generate_mock_data(num_readings=1000):
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(seconds=num_readings * 0.1)  # 10Hz data
    
    data = []
    for i in range(num_readings):
        timestamp = start_time + timedelta(seconds=i * 0.1)
        data.append(generate_mock_reading(timestamp))
    
    return data

def save_mock_data():
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Generate and save historical data
    historical_data = generate_mock_data(1000)
    with open('data/sensor_data.json', 'w') as f:
        for item in historical_data:
            f.write(json.dumps(item) + '\n')
    
    # Save current data
    current_data = generate_mock_reading(datetime.utcnow())
    with open('data/current_reading.json', 'w') as f:
        json.dump(current_data, f)
    
    print(f"Generated {len(historical_data)} historical readings and 1 current reading")

if __name__ == "__main__":
    save_mock_data()
