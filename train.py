from environment.api_env import APIRoutingEnv
from agent.dqn_agent import DQNAgent
import torch


env = APIRoutingEnv()

agent = DQNAgent(
    state_size=13,
    action_size=3
)

episodes = 100


for episode in range(episodes):

    state, info = env.reset()

    total_reward = 0

    for step in range(100):

        # Agent chooses provider
        action_mask = env.get_action_mask()

        action = agent.choose_action(
            state,
            action_mask
        )

        # Environment processes request
        next_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        # Store experience
        agent.memory.add(
            state,
            action,
            reward,
            next_state,
            done
        )

        # Learn
        agent.train()

        state = next_state

        total_reward += reward

        if done:
            break

    print(
        f"Episode {episode + 1}: "
        f"Reward = {total_reward:.2f}, "
        f"Epsilon = {agent.epsilon:.3f}"
    )

torch.save(
    agent.policy_net.state_dict(),
    "models/dqn_model.pth"
)
    
print("Model saved!")