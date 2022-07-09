# bootstrap naive RL runs
import gym
from stable_baselines3 import PPO, DQN
from stable_baselines3 import A2C, DDPG
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
#env = gym.make("gym_cache:cache-episode-v0")
env = gym.make("gym_cache:cache-guessing-game-v0")
#model = PPO(MlpPolicy, env, verbose=0)
model =PPO("MlpPolicy", env, verbose=1)

###model.learn(total_timesteps=25000)
####model = A2C("MlpPolicy", env, verbose=1)
####model.learn(total_timesteps=25000)
###observation = env.reset()
###total_reward = 0
###num_correct =0
###num_wrong = 0
###for i in range(2000):
###    print(i)
###    #env.render()
###    #action = env.action_space.sample() #my agent
###    action, _state = model.predict(observation, deterministic = False)
###    observation, reward, done, info = env.step(action)
###    if reward > 0:
###        num_correct += 1
###    if reward < 0:
###        num_wrong += 1
###    total_reward += reward
###    print(action)
###    print(reward)
###    if done:
###        observation = env.reset()
###
###print(num_correct)
###print(num_wrong)

num_correct_arr=[]
num_wrong_arr=[]
num_violation_arr=[]
num_length_violation_arr=[]
num_double_victim_arr=[]
num_no_victim_arr=[]
for _ in range(1):
  obs = env.reset()
  model.learn(5000)
  total_reward = 0
  num_correct =0
  num_wrong = 0
  num_violation = 0
  num_length_violation = 0
  num_double_victim = 0
  num_no_victim = 0
  for i in range(10000):
      print(i)
      # We need to pass the previous state and a mask for recurrent policies
      # to reset lstm state when a new episode begin
      action, state = model.predict(obs, deterministic = False)
      obs, reward , done, _ = env.step(action)
      # Note: with VecEnv, env.reset() is automatically called
      if reward > 10 and action[1] == True:
          num_correct += 1
      if reward == -9999 and action[1] == True: #is guess and wrong
          num_wrong += 1
      if done and reward < -9999:
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
      print(action)
      print(obs)
      print(reward)
      if done == True:
          obs = env.reset()
          done = False
          print(obs)
  print(num_correct)
  print(num_wrong)
  print(num_violation)
  print(num_length_violation) 
  print(num_double_victim)
  print(num_no_victim)
  #print(1.0 * num_correct / (num_correct + num_wrong))
  #print(1.0 * (num_correct + num_wrong) / (num_violation + num_correct + num_wrong))
  num_correct_arr.append(num_correct)
  num_wrong_arr.append(num_wrong)
  num_violation_arr.append(num_violation)
  num_length_violation_arr.append(num_length_violation)
  num_double_victim_arr.append(num_double_victim)
  num_no_victim_arr.append(num_no_victim)
