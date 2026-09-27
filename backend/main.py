import sys
import os
import asyncio

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent.dqn import DQN
from environment.providers import providers
from backend.baselines import RoundRobinRouter, CostRouter, SuccessRateRouter

app = FastAPI(title="Autonomous Agentic API Rate-Limiter & Multi-Provider Router")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize baseline routers
round_robin_router = RoundRobinRouter(providers)
cost_router = CostRouter(providers)
success_router = SuccessRateRouter(providers)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/dqn_model.pth")
state_dim = 13
action_dim = 3

policy_net = DQN(state_dim, action_dim)
if os.path.exists(MODEL_PATH):
    policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    policy_net.eval()
else:
    print("Warning: No trained model found! Using random weights.")

class RouteRequest(BaseModel):
    traffic: int = 5

def generate_routing_event():
    """Simulates a live request and uses the DQN model to make a routing decision."""
    traffic = int(np.random.randint(1, 10))
    
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

    valid_actions = [i for i, p in enumerate(providers) if p.remaining_requests > 0]
    
    if not valid_actions:
        for p in providers:
            p.reset()
        valid_actions = [0, 1, 2]

    with torch.no_grad():
        state_tensor = torch.FloatTensor(np.array(state, dtype=np.float32)).unsqueeze(0)
        q_values = policy_net(state_tensor).squeeze(0).numpy()

    masked_q_values = np.full_like(q_values, -np.inf)
    for action in valid_actions:
        masked_q_values[action] = q_values[action]

    chosen_action = int(np.argmax(masked_q_values))
    selected_provider = providers[chosen_action]

    result = selected_provider.handle_request()

    return {
        "traffic": traffic,
        "chosen_provider": selected_provider.name,
        "action_index": chosen_action,
        "status": result["status_code"],
        "latency": result["latency"],
        "cost": result["cost"],
        "q_values": q_values.tolist(),
        "remaining_capacity": {p.name: p.remaining_requests for p in providers}
    }

@app.get("/")
def read_root():
    return {
        "message": "Autonomous Agentic API Rate-Limiter & Multi-Provider Router is live!",
        "docs_url": "/docs",
        "health_check": "/health"
    }

@app.post("/route")
def route_request(req: RouteRequest):
    state = [
        float(req.traffic),
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

    valid_actions = [i for i, p in enumerate(providers) if p.remaining_requests > 0]
    
    if not valid_actions:
        raise HTTPException(status_code=429, detail="All provider rate limits exhausted!")

    with torch.no_grad():
        state_tensor = torch.FloatTensor(np.array(state, dtype=np.float32)).unsqueeze(0)
        q_values = policy_net(state_tensor).squeeze(0).numpy()

    masked_q_values = np.full_like(q_values, -np.inf)
    for action in valid_actions:
        masked_q_values[action] = q_values[action]

    chosen_action = int(np.argmax(masked_q_values))
    selected_provider = providers[chosen_action]

    result = selected_provider.handle_request()

    return {
        "chosen_provider": selected_provider.name,
        "action_index": chosen_action,
        "result": result,
        "q_values": q_values.tolist(),
        "remaining_capacity": {p.name: p.remaining_requests for p in providers}
    }

@app.websocket("/ws/simulate")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            traffic_data = generate_routing_event()
            await websocket.send_json(traffic_data)
            await asyncio.sleep(0.8)
    except WebSocketDisconnect:
        print("Frontend client disconnected from simulation stream.")

@app.post("/route/round-robin")
def route_round_robin(req: RouteRequest):
    provider, _ = round_robin_router.route(req.traffic)
    if not provider:
        raise HTTPException(status_code=429, detail="All provider rate limits exhausted!")
    result = provider.handle_request()
    return {"router": "RoundRobin", "chosen_provider": provider.name, "result": result}

@app.post("/route/cost")
def route_cost(req: RouteRequest):
    provider, _ = cost_router.route(req.traffic)
    if not provider:
        raise HTTPException(status_code=429, detail="All provider rate limits exhausted!")
    result = provider.handle_request()
    return {"router": "CostRouter", "chosen_provider": provider.name, "result": result}

@app.post("/route/success")
def route_success(req: RouteRequest):
    provider, _ = success_router.route(req.traffic)
    if not provider:
        raise HTTPException(status_code=429, detail="All provider rate limits exhausted!")
    result = provider.handle_request()
    return {"router": "SuccessRouter", "chosen_provider": provider.name, "result": result}

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": os.path.exists(MODEL_PATH)}

@app.post("/reset-providers")
def reset_providers():
    for p in providers:
        p.reset()
    return {"message": "All providers reset", "capacity": {p.name: p.remaining_requests for p in providers}}