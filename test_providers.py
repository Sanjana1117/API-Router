from environment.traffic import generate_traffic


traffic = generate_traffic(20)

for i, requests in enumerate(traffic):
    print(f"Time {i + 1}: {requests} requests")