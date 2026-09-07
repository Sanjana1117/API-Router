import random

class Provider:

    def __init__(self, name, rate_limit, cost, latency, success_rate):
        self.name = name
        self.rate_limit = rate_limit
        self.remaining_requests = rate_limit
        self.cost = cost
        self.latency = latency
        self.success_rate = success_rate

    def handle_request(self):

        # Rate limit exceeded
        if self.remaining_requests <= 0:
            return {
                "success": False,
                "status_code": 429,
                "latency": 0,
                "cost": 0
            }

        # Consume one request
        self.remaining_requests -= 1

        # Simulate success/failure
        success = random.random() < self.success_rate

        if success:
            return {
                "success": True,
                "status_code": 200,
                "latency": self.latency,
                "cost": self.cost
            }

        return {
            "success": False,
            "status_code": 500,
            "latency": self.latency,
            "cost": self.cost
        }

    def reset(self):
        self.remaining_requests = self.rate_limit

provider_a = Provider(
    name="Provider A",
    rate_limit=10,
    cost=1.0,
    latency=300,
    success_rate=0.95
)

provider_b = Provider(
    name="Provider B",
    rate_limit=20,
    cost=3.0,
    latency=150,
    success_rate=0.98
)

provider_c = Provider(
    name="Provider C",
    rate_limit=15,
    cost=2.0,
    latency=220,
    success_rate=0.97
)

providers = [provider_a, provider_b, provider_c]