import requests
import time

URL = "http://127.0.0.1:8000/route"
RESET_URL = "http://127.0.0.1:8000/reset-providers"

# Reset providers to full capacity before testing
print("Resetting providers...")
requests.post(RESET_URL)

print("\n--- Simulating Client Traffic ---")
# Send 35 requests rapidly to exhaust provider capacities and test action masking / 429 handling
for i in range(1, 36):
    random_traffic = i % 10 + 1
    payload = {"traffic": random_traffic}
    try:
        response = requests.post(URL, json=payload)
        if response.status_code == 200:
            data = response.json()
            print(f"Request {i:02d} | Traffic: {random_traffic} | Routed to: {data['chosen_provider']} | Status: {data['result']['status_code']} | Latency: {data['result']['latency']}ms | Cost: ₹{data['result']['cost']}")
        elif response.status_code == 429:
            print(f"Request {i:02d} | ❌ Rate Limited (429 Too Many Requests): All provider capacities exhausted.")
        else:
            print(f"Request {i:02d} | Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Request {i:02d} | Connection failed: {e}")
    
    time.sleep(0.1) # Small delay to watch the logs comfortably