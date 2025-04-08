import ale_py
import gymnasium as gym
import time

def run_random_agent(env_name):
    env = gym.make(env_name, render_mode='human')
    observation, info = env.reset()

    done = False
    while not done:
        env.render()
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            observation, info = env.reset()
            done = True
        time.sleep(0.02)
    env.close()

if __name__ == '__main__':
    run_random_agent('ALE/Breakout-v5')
