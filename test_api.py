import requests
import json
import time

# Base URL for the API
BASE_URL = "http://localhost:5000/api"

def test_health():
    """Test the health endpoint."""
    print("\n=== Testing health endpoint ===")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_get_current_data():
    """Test getting current sensor data."""
    print("\n=== Testing current data endpoint ===")
    try:
        response = requests.get(f"{BASE_URL}/data/current")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_get_data_history():
    """Test getting historical data."""
    print("\n=== Testing data history endpoint ===")
    try:
        params = {"limit": 5}  # Get last 5 records
        response = requests.get(f"{BASE_URL}/data/history", params=params)
        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"Found {len(data.get('data', []))} records")
        if data.get('data'):
            print(f"Sample record: {json.dumps(data['data'][0], indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_collection_control():
    """Test starting and stopping data collection."""
    print("\n=== Testing collection control endpoints ===")
    
    # Test starting collection
    print("\nStarting collection...")
    try:
        response = requests.post(f"{BASE_URL}/collection/start")
        print(f"Start Collection - Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        if response.status_code != 200:
            return False
    except Exception as e:
        print(f"Error starting collection: {e}")
        return False
    
    # Wait a moment for collection to start
    time.sleep(2)
    
    # Test stopping collection
    print("\nStopping collection...")
    try:
        response = requests.post(f"{BASE_URL}/collection/stop")
        print(f"Stop Collection - Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error stopping collection: {e}")
        return False

if __name__ == "__main__":
    # Run tests
    tests = [
        ("Health Check", test_health),
        ("Current Data", test_get_current_data),
        ("Data History", test_get_data_history),
        ("Collection Control", test_collection_control)
    ]
    
    print("=== Starting Thoth API Tests ===")
    results = {}
    
    for name, test_func in tests:
        print(f"\n{'='*20} {name} {'='*20}")
        success = test_func()
        results[name] = "PASS" if success else "FAIL"
    
    # Print summary
    print("\n=== Test Summary ===")
    for name, result in results.items():
        print(f"{name}: {result}")
    
    # Exit with appropriate status code
    if all(r == "PASS" for r in results.values()):
        print("\nAll tests passed!")
        exit(0)
    else:
        print("\nSome tests failed!")
        exit(1)
