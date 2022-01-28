'''
Author Mulong Luo
Date 2022.1.24
usage: resotre the ray checkpoint to replay the agent and extract the attack pattern
'''

from copy import deepcopy
import gym
from starlette.requests import Request
import requests
import pprint
import ray
from ray import serve
from run_gym_rrllib import * # need this to import the config and PPOtrainer

print(config)
#tune.register_env("cache_guessing_game_env_fix", CacheGuessingGameEnv)#Fix)
#exit(0)

checkpoint_path ='/home/ml2558/ray_results/PPO_cache_guessing_game_env_fix_2022-01-24_21-18-203pft9506/checkpoint_000136/checkpoint-136'
trainer = PPOTrainer(config=config)
trainer.restore(checkpoint_path)


env = CacheGuessingGameEnv(config["env_config"])
obs = env.reset()


for _ in range(1000):
    print(f"-> Sending observation {obs}")
    # Setting explore=False should always return the same action.
    action = trainer.compute_single_action(obs, explore=False)
    print(f"<- Received response {action}")
    obs, reward, done, info = env.step(action)
    if done == True:
        obs = env.reset()

# no cache randomization
# no randomized inference
pattern_buffer = []
for victim_addr in range(env.victim_address_min, env.victim_address_max + 1):
    obs = env.reset(victim_address=victim_addr)
    action_buffer = []
    done = False
    while done == False:
        print(f"-> Sending observation {obs}")
        action = trainer.compute_single_action(obs, explore=False)
        print(f"<- Received response {action}")
        obs, reward, done, info = env.step(action)
        action_buffer.append((action, obs[0]))
    if reward > 0:
        correct = True
    else:
        correct = False
    pattern_buffer.append((victim_addr, action_buffer, correct))
pprint.pprint(pattern_buffer)


def replay_agent():
    # no cache randomization
    # rangomized inference ( 10 times)
    pattern_buffer = []
    num_guess = 0
    num_correct = 0
    for victim_addr in range(env.victim_address_min, env.victim_address_max + 1):
        for repeat in range(5):
            obs = env.reset(victim_address=victim_addr)
            env._randomize_cache("union")
            action_buffer = []
            done = False
            while done == False:
                print(f"-> Sending observation {obs}")
                action = trainer.compute_single_action(obs, explore=False) # randomized inference
                print(f"<- Received response {action}")
                obs, reward, done, info = env.step(action)
                action_buffer.append((action, obs[0]))
            if reward > 0:
                correct = True
                num_correct += 1
            else:
                correct = False
            num_guess += 1
            pattern_buffer.append((victim_addr, action_buffer, correct))
    pprint.pprint(pattern_buffer)
    return 1.0 * num_correct / num_guess, pattern_buffer

replay_agent()
#import pickle
#ickle.loads(pickle.dumps(trainer.get_policy()))

# cache randomization
# no randomized inference





