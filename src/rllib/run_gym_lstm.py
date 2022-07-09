# bootstrap naive RL runs
import gym
from stable_baselines import PPO2
#env = gym.make("gym_cache:cache-guessing-game-v0")
#import wandb
#from wandb import config
#import os
#wandb.login(key='eb6f1b610f9f4bb9f1bd7c8e03173a84ea25a050')
#import tensorflow as tf
#wandb.init(project="my-test-project", entity="mulongluo", config=tf.flags.FLAGS, sync_tensorboard=True)
#os.environ["WANDB_MODE"] = "offline"

#model = PPO2('MlpLstmPolicy', "gym_cache:cache-guessing-game-v0", nminibatches=1, verbose=1, policy_kwargs={"n_lstm":20})
model = PPO2('MlpPolicy', "gym_cache:cache-guessing-game-v0", nminibatches=1, verbose=1)

env = model.get_env()
obs = env.reset()
total_reward = 0

num_correct_arr=[]
num_wrong_arr=[]
num_violation_arr=[]
num_length_violation_arr=[]
num_double_victim_arr=[]
num_no_victim_arr=[]

num_correct =0
num_wrong = 0
num_violation = 0
num_length_violation = 0
num_double_victim = 0
num_no_victim = 0
model.learn(10000)
for i in range(10000):
    print(i)
    # We need to pass the previous state and a mask for recurrent policies
    # to reset lstm state when a new episode begin
    action, state = model.predict(obs)
    obs, reward , done, _ = env.step(action)
    print("wtf")
    iswtf = False
    # Note: with VecEnv, env.reset() is automatically called
    if reward > 10 and action[0][1] == True:
        num_correct += 1
    if reward == -200 and action[0][1] == True: #is guess and wrong
        num_wrong += 1
    if done and reward < -200:
        num_violation += 1
        if reward == -10000:
            num_length_violation += 1
        elif reward == -20000:
            num_double_victim += 1
        else:
            num_no_victim += 1
    #total_reward += reward
    #wandb.log({'reward': reward})
    #wandb.log({'total_reward': total_reward})
    iswtf = True
    print(action)
    print(obs)
    print(reward)
    if done == True:
        if iswtf == False:
          exit()
        obs = env.reset()
        print(obs)

print(num_correct)
print(num_wrong)
print(num_violation)
print(num_length_violation) 
print(num_double_victim)
print(num_no_victim)
print(1.0 * num_correct / (num_correct + num_wrong))
print(1.0 * (num_correct + num_wrong) / (num_violation + num_correct + num_wrong))
num_correct_arr.append(num_correct)
num_wrong_arr.append(num_wrong)
num_violation_arr.append(num_violation)
num_length_violation_arr.append(num_length_violation)
num_double_victim_arr.append(num_double_victim)
num_no_victim_arr.append(num_no_victim)