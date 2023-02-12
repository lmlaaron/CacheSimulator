import hydra
import os
import sys
import torch
import pickle
import numpy as np
from typing import Dict
from tqdm import tqdm
from sklearn.svm import SVC

from rlmeta.core.types import Action
from rlmeta.envs.env import Env
from rlmeta.utils.stats_dict import StatsDict
import rlmeta.utils.nested_utils as nested_utils

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import CacheAttackerDetectorEnv, CacheAttackerDetectorEnvFactory
from model import CachePPOTransformerModel
from agent import PPOAgent, SpecAgent, CycloneAgent, PrimeProbeAgent

LABEL={ 'attacker':1,
        'benign':0,
        }

def unbatch_action(action: Action) -> Action:
    act, info = action
    act.squeeze_(0)
    info = nested_utils.map_nested(lambda x: x.squeeze(0), info)
    return Action(act, info)


def run_loop(env: Env, agents, victim_addr=-1) -> Dict[str, float]:
    episode_length = 0
    episode_return = 0.0
    detector_count = 0.0
    detector_acc = 0.0

    #env.env.opponent_weights = [0.5, 0.5]
    if victim_addr == -1:
        timestep = env.reset()
    else:
        timestep = env.reset(victim_address=victim_addr)
    #print("victim address: ", env.env.victim_address )
    for agent_name, agent in agents.items():
        agent.observe_init(timestep[agent_name])
    while not timestep["__all__"].done:
        # Model server requires a batch_dim, so unsqueeze here for local runs.
        actions = {}
        for agent_name, agent in agents.items():
            timestep[agent_name].observation.unsqueeze_(0)
            #print("attacker obs")
            #print(timestep["attacker"].observation)
            action = agent.act(timestep[agent_name])
            # Unbatch the action.
            if isinstance(action, tuple):
                action = Action(action[0], action[1])
            if not isinstance(action.action, int):
                action = unbatch_action(action)
            actions.update({agent_name:action})
        #print(actions)
        timestep = env.step(actions)

        for agent_name, agent in agents.items():
            agent.observe(actions[agent_name], timestep[agent_name])

        episode_length += 1
        episode_return += timestep['attacker'].reward

        try:
            detector_action = actions['detector'].action.item()
        except:
            detector_action = actions['detector'].action
        if timestep["__all__"].done and detector_action ==1:
            detector_count += 1
        detector_accuracy = detector_count

    metrics = {
        "episode_length": episode_length,
        "episode_return": episode_return,
        "detector_accuracy": detector_accuracy,
    }

    #Cyclone
    return agents['detector'].cyclone_counters, env.env.opponent_agent
    #Same observation   
    #return timestep['detector'].observation.numpy(), env.env.opponent_agent

def collect(cfg, num_samples):
    # load agents and 
    # run environment loop to collect data
    # return 
    env_fac = CacheAttackerDetectorEnvFactory(cfg.env_config)
    env = env_fac(index=0)
    #num_samples = 50

    # Load model
    # Attacker
    cfg.model_config["output_dim"] = env.action_space.n
    attacker_params = torch.load(cfg.attacker_checkpoint)
    attacker_model = CachePPOTransformerModel(**cfg.model_config)
    attacker_model.load_state_dict(attacker_params)
    attacker_model.eval()
    
    #attacker_agent = PPOAgent(attacker_model, deterministic_policy=cfg.deterministic_policy)
    attacker_agent = PrimeProbeAgent(cfg.env_config)
    detector_agent = CycloneAgent(cfg.env_config)
    spec_trace_f = open(os.path.expanduser('~')+'/remix3.txt','r')
    spec_trace = spec_trace_f.read().split('\n')[:1000000]
    trace = []
    for line in spec_trace:
        line = line.split()
        trace.append(line)
    spec_trace = trace
    benign_agent = SpecAgent(cfg.env_config, spec_trace, legacy_trace_format=True)
    agents = {"attacker": attacker_agent, "detector": detector_agent, "benign": benign_agent}
    X, y = [], [] 
    for i in tqdm(range(num_samples)):
        x, label = run_loop(env, agents)
        X.append(x)
        y.append(LABEL[label])
    X = np.array(X)
    #num_samples, m, n = X.shape
    X = X.reshape(num_samples, -1)
    y = np.array(y)
    #print('features:\n',X,'labels\n',y)
    return X, y

def train(cfg):
    # run data collection and 
    # train the svm classifier
    # report accuracy
    X_train, y_train = collect(cfg, num_samples=20000)
    X_test, y_test = collect(cfg, num_samples=100)
    clf = SVC(kernel='rbf', gamma='auto')
    clf.fit(X_train,y_train)
    print("Train Accuracy:", clf.score(X_train,y_train))
    print("Test Accuracy:", clf.score(X_test, y_test))
    
    print("saving the classfier")
    pickle.dump(clf,open('cyclone.pkl','wb'))
    return clf

@hydra.main(config_path="../config", config_name="cyclone")
def main(cfg):
    clf = train(cfg)
    return clf

if __name__=='__main__':
    clf = main()
