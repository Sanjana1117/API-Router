class RoundRobinRouter:
    def __init__(self, providers):
        self.providers = providers
        self.index = 0

    def route(self, traffic):
        for _ in range(len(self.providers)):
            provider = self.providers[self.index]
            self.index = (self.index + 1) % len(self.providers)
            if provider.remaining_requests > 0:
                return provider, self.index
        return None, -1


class CostRouter:
    def __init__(self, providers):
        # Sort by cost ascending (cheapest first)
        self.providers = sorted(providers, key=lambda p: p.cost)

    def route(self, traffic):
        for provider in self.providers:
            if provider.remaining_requests > 0:
                return provider, provider.name
        return None, None


class SuccessRateRouter:
    def __init__(self, providers):
        # Sort by success rate descending (highest success first)
        self.providers = sorted(providers, key=lambda p: p.success_rate, reverse=True)

    def route(self, traffic):
        for provider in self.providers:
            if provider.remaining_requests > 0:
                return provider, provider.name
        return None, None