import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import matplotlib.pyplot as plt
import torch
import numpy as np
from environment.providers import Provider
from backend.baselines import RoundRobinRouter, CostRouter, SuccessRateRouter
from agent.dqn import DQN

def collect_metrics():
    routers = ["RoundRobin", "CostRouter", "SuccessRouter", "DQN"]
    metrics = {r: {"429": 0, "cost": 0.0, "latency": 0.0, "success": 0} for r in routers}
    
    num_requests = 35
    
    for router_name in routers:
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

        total_lat = 0
        processed = 0

        for _ in range(num_requests):
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
                    metrics[router_name]["429"] += 1
                    continue
                with torch.no_grad():
                    q_vals = policy_net(torch.FloatTensor(state).unsqueeze(0)).squeeze(0).numpy()
                masked = np.full_like(q_vals, -np.inf)
                for a in valid_actions:
                    masked[a] = q_vals[a]
                chosen = int(np.argmax(masked))
                res = providers[chosen].handle_request()
            else:
                provider, _ = router.route(traffic)
                if not provider:
                    metrics[router_name]["429"] += 1
                    continue
                res = provider.handle_request()

            if res["status_code"] == 429:
                metrics[router_name]["429"] += 1
            else:
                metrics[router_name]["success"] += 1
                metrics[router_name]["cost"] += res["cost"]
                total_lat += res["latency"]
                processed += 1

        metrics[router_name]["latency"] = (total_lat / processed) if processed > 0 else 0
        
    return metrics

def plot_results(metrics):
    names = list(metrics.keys())
    rate_limits = [metrics[r]["429"] for r in names]
    costs = [metrics[r]["cost"] for r in names]
    latencies = [metrics[r]["latency"] for r in names]

    os.makedirs(os.path.join(os.path.dirname(__file__), "../reports"), exist_ok=True)

    plt.figure(figsize=(12, 4))

    # 1. 429 Rate Limit Hits
    plt.subplot(1, 3, 1)
    plt.bar(names, rate_limits, color=['gray', 'orange', 'blue', 'green'])
    plt.title("429 Rate Limit Hits (Lower = Better)")
    plt.ylabel("Count")
    plt.xticks(rotation=15)

    # 2. Total Cost
    plt.subplot(1, 3, 2)
    plt.bar(names, costs, color=['gray', 'orange', 'blue', 'green'])
    plt.title("Total Cost (₹)")
    plt.ylabel("Cost in ₹")
    plt.xticks(rotation=15)

    # 3. Average Latency
    plt.subplot(1, 3, 3)
    plt.bar(names, latencies, color=['gray', 'orange', 'blue', 'green'])
    plt.title("Avg Latency (ms) (Lower = Better)")
    plt.ylabel("Latency (ms)")
    plt.xticks(rotation=15)

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), "../reports/performance_comparison.png")
    plt.savefig(output_path)
    print(f"Graphs successfully generated and saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    print("Collecting performance data across routers...")
    data = collect_metrics()
    plot_results(data)