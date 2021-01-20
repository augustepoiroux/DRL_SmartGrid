import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# import datetime
# from Env import Env
# from Model import train, save, load
# from Analyze import test

# envTrain = Env("Data/select_train_data_30m_2.csv")
# envTest = Env("Data/select_test_data_30m_2.csv")

# ALGO = "simple"
# HIDDEN_LAYERS = [50, 20]
# NB_EPISODES = 2000
# NB_STEPS = 50
# BATCH_SIZE = 100

# model_name = (
#     f"{ALGO}_"
#     + "-".join(list(map(str, HIDDEN_LAYERS)))
#     + f"nn_{NB_EPISODES}ep_{NB_STEPS}s_{BATCH_SIZE}b"
# )

# print("Training...")
# DQN = train(
#     envTrain,
#     envTest,
#     hidden_layers=HIDDEN_LAYERS,
#     nb_episodes=NB_EPISODES,
#     nb_steps=NB_STEPS,
#     batch_size=BATCH_SIZE,
#     model_name=model_name,
#     algo=ALGO,
#     # save_episode=200,
#     # recup_model=True,
# )
# print("Done")

# test(envTest, nb_step=300, DQN_model=DQN)

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor

from EnvGym import Env
from AnalyzeGym import test

envTrain = Env("Data/select_train_data_30m_2.csv")
check_env(envTrain)

envTest = Env("Data/select_test_data_30m_2.csv", epLength=300)
check_env(envTest)

log_dir = "./logs/"
env = Monitor(envTrain, log_dir)

# model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

model.learn(total_timesteps=200000)

test(envTrain, model=model)
print()
test(envTrain, model=model)
print()
test(envTest, model=model)
print()
test(envTest, model=model)

# import numpy as np

# np.set_printoptions(2, suppress=True)

# obs = env.reset()
# for i in range(10):
#     action = np.random.randint(0, 3)
#     obs, rewards, dones, info = env.step(action)
#     print(f"Reward: {np.float32([rewards])}\tObs: {obs}")
