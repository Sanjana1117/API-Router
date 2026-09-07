from environment.api_env import APIRoutingEnv


env = APIRoutingEnv()

state, info = env.reset()

print("Initial state:")
print(state)

for i in range(10):

    # Random action for now
    action = env.action_space.sample()

    next_state, reward, terminated, truncated, info = env.step(action)

    print("\nRequest:", i + 1)
    print("Action:", action)
    print("Provider:", info)
    print("Reward:", reward)
    print("Next state:", next_state)