from agent.dqn_agent import DQNAgent

agent = DQNAgent()

state = [
    8,
    7, 19, 14,
    300, 150, 220,
    1, 3, 2,
    0.95, 0.98, 0.97
]

for i in range(10):

    action = agent.choose_action(state)

    print("Chosen action:", action)