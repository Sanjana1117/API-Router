from agent.dqn_agent import DQNAgent

agent = DQNAgent()

state = [
    8,
    7, 19, 14,
    300, 150, 220,
    1, 3, 2,
    0.95, 0.98, 0.97
]

next_state = [
    7,
    6, 19, 14,
    300, 150, 220,
    1, 3, 2,
    0.95, 0.98, 0.97
]

# Put some experiences into memory
for i in range(40):

    action = agent.choose_action(state)

    agent.memory.add(
        state,
        action,
        10,
        next_state,
        False
    )

print("Memory size:", len(agent.memory))

print("Epsilon before:", agent.epsilon)

agent.train()

print("Epsilon after:", agent.epsilon)