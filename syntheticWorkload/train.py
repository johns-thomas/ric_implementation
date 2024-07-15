import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import pandas as pd


def create_model(input_shape, output_shape):
    model = tf.keras.Sequential([
        layers.Dense(24, activation='relu', input_shape=input_shape),
        layers.Dense(24, activation='relu'),
        layers.Dense(output_shape, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

def train_rl_agent(env, model, num_episodes=1000):
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = np.argmax(model.predict(state.reshape([1, state.shape[0]])))
            new_state, reward, done, _ = env.step(action)
            total_reward += reward
            target = reward + 0.95 * np.amax(model.predict(new_state.reshape([1, state.shape[0]])))
            target_f = model.predict(state.reshape([1, state.shape[0]]))
            target_f[0][action] = target
            model.fit(state.reshape([1, state.shape[0]]), target_f, epochs=1, verbose=0)
            state = new_state
        print(f'Episode: {episode+1}, Total Reward: {total_reward}')


