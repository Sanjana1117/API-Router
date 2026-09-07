import random


def generate_traffic(num_requests, surge_probability=0.1):
    traffic = []

    for _ in range(num_requests):

        # Randomly create a traffic surge
        if random.random() < surge_probability:
            requests = random.randint(20, 50)
        else:
            requests = random.randint(1, 10)

        traffic.append(requests)

    return traffic