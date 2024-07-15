#from synthetic_workload import generate_synthetic_data
#from rl_agent import SyntheticLambdaEnv
#from train import *
import synthetic_workload
import rl_agent
import train
import numpy as np
import pandas as pd
def evaluate_rl_agent(env, model):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = np.argmax(model.predict(state.reshape([1, state.shape[0]])))
        new_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = new_state
    return total_reward

synthetic_data = synthetic_workload.generate_synthetic_data()

# Split synthetic data into training and validation sets
training_data = synthetic_data.iloc[:800]  # First 80% for training
validation_data = synthetic_data.iloc[800:]  # Last 20% for validation

# Create environments
training_env = rl_agent.SyntheticLambdaEnv(training_data)
validation_env = rl_agent.SyntheticLambdaEnv(validation_data)

# Train RL agent
input_shape = (7,)
output_shape = 3
model = train.create_model(input_shape, output_shape)
train.train_rl_agent(training_env, model)
#train_rl_agent(training_env, model)

# Validate RL agent
validation_reward = evaluate_rl_agent(validation_env, model)
print(f'Total Reward on Validation Set: {validation_reward}')
