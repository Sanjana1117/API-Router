import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import torch
import numpy as np
from environment.providers import Provider
from backend.baselines import RoundRobinRouter, CostRouter, SuccessRateRouter
from agent.dqn import DQN

def run_simulation(router_name, num_requests=50):
    # Fresh isolated providers for fair comparison
    providers = [
        Provider(name="A", rate_limit=10, cost=1.0, latency=300, success_rate=0.95),
        Provider(name="B", rate_limit=20, cost=3.0, latency=150, success_rate=0.98),
        Provider(name="C", rate_limit=15, cost=2.0, latency=220, success_rate=0.97),
    ]
    
    if router_name == "RoundRobin":
        router = RoundRobinRouter(providers)
    elif router_name == "CostRouter":
        router = CostRouter(providers)
    elif router_name == "SuccessRouter":
        router = SuccessRateRouter(providers)
    elif router_name == "DQN":
        model_path = os.path.join(os.path.dirname(__file__), "../models/dqn_model.pth")
        policy_net = DQN(13, 3)
        if os.path.exists(model_path):
            policy_net.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        policy_net.eval()

    total_cost = 0.0
    total_latency = 0.0
    success_count = 0
    rate_limit_count = 0
    failure_count = 0

    for i in range(num_requests):
        traffic = np.random.randint(1, 10)
        
        if router_name == "DQN":
            state = [
                float(traffic),
                float(providers[0].remaining_requests),
                float(providers[1].remaining_requests),
                float(providers[2].remaining_requests),
                float(providers[0].latency),
                float(providers[1].latency),
                float(providers[2].latency),
                float(providers[0].cost),
                float(providers[1].cost),
                float(providers[2].cost),
                float(providers[0].success_rate),
                float(providers[1].success_rate),
                float(providers[2].success_rate),
            ]
            valid_actions = [idx for idx, p in enumerate(providers) if p.remaining_requests > 0]
            if not valid_actions:
                rate_limit_count += 1
                continue
            
            with torch.no_grad():
                q_vals = policy_net(torch.FloatTensor(state).unsqueeze(0)).squeeze(0).numpy()
            masked = np.full_like(q_vals, -np.inf)
            for a in valid_actions:
                masked[a] = q_vals[a]
            chosen = int(np.argmax(masked))
            provider = providers[chosen]
            res = provider.handle_request()
        else:
            provider, _ = router.route(traffic)
            if not provider:
                rate_limit_count += 1
                continue
            res = provider.handle_request()

        if res["status_code"] == 429:
            rate_limit_count += 1
        elif res["success"]:
            success_count += 1
            total_cost += res["cost"]
            total_latency += res["latency"]
        else:
            failure_count += 1
            total_cost += res["cost"]
            total_latency += res["latency"]

    processed = success_count + failure_count
    avg_latency = total_latency / processed if processed > 0 else 0
    
    print(f"--- Router: {router_name} ---")
    print(f"Total Requests Tried: {num_requests}")
    print(f"Successful Requests: {success_count}")
    print(f"429 Rate Limit Hits: {rate_limit_count}")
    print(f"Simulated Failures (500): {failure_count}")
    print(f"Total Cost: ₹{total_cost:.2f}")
    print(f"Average Latency: {avg_latency:.2f}ms\n")

if __name__ == "__main__":
    print("Running Comparative Evaluations...\n")
    for r in ["RoundRobin", "CostRouter", "SuccessRouter", "DQN"]:
        run_simulation(r)