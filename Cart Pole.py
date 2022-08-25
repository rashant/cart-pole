'''TESTING RANDOM ENVIRONMENT WITH OPENAI GYM'''
import gym

env = gym.make("CartPole-v1")  # creating the cart pole environment
states = env.observation_space.shape[0]
actions = env.action_space.n
print(actions)
print(states)

episodes = 10
# for episode in range(episodes):
#     state=env.reset()
#     done=False
#     score=0
#
#     while not done:
#         env.render()
#         action=random.choice([0,1])
#         n_state,reward,done,info=env.step(action)
#         score+=reward
#     print(f"Episode: {episode+1} Score: {score}")

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam


def build_model(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1, states)))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model




'''BUILDING AGENT WITH KERAS-RL'''
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

model = build_model(states, actions)
print(model.summary())
def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dgn = DQNAgent(model=model, memory=memory, policy=policy, nb_actions=actions, nb_steps_warmup=100,
                   target_model_update=1e-2)
    return dgn


dgn = build_agent(model, actions)
dgn.compile(optimizer=Adam(lr=1e-3),metrics=['mae'])
dgn.fit(env, nb_steps=50000, visualize=True, verbose=1)

import numpy as np
'''TESTING THE MODEL'''
scores=dgn.test(env,nb_episodes=100,visualize=True)
print(np.mean(scores.history['episode_reward']))

'''RELOADING THE AGENT FROM MEMORY BY SAVING THE WEIGHTS'''
dgn.save_weights('dgn_weights.h5f',overwrite=True)
#
# '''REUSING THE MODEL'''
# dgn.load_weights('dgn_weights.h5f')