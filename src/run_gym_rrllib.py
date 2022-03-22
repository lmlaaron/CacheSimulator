# look at https://github.com/ray-project/ray/blob/ea2bea7e309cd60457aa0e027321be5f10fa0fe5/rllib/examples/custom_env.py#L2
#from CacheSimulator.src.gym_cache.envs.cache_simulator_wrapper import CacheSimulatorWrapper
import gym
import ray
import ray.tune as tune
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import restore_original_dimensions
import torch.nn as nn
import numpy as np
from ray.rllib.models import ModelCatalog
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.sac import SACTrainer
import sys
import copy
import torch
#from ray.rllib.offline import JsonReader
from torch.autograd import Variable

sys.path.append("../src")
from models.dqn_model import DNNEncoder 



# the actual model used by the RLlib
from collections import deque
class TestModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        hidden_dim = 256 
        super(TestModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.a_model = nn.Sequential(
            DNNEncoder(
                input_dim=int(np.product(self.obs_space.shape)),
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
            ),
            nn.Linear(hidden_dim, num_outputs)
        )
        self.v_model = nn.Sequential(
            DNNEncoder(
                input_dim=int(np.product(self.obs_space.shape)),
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
            ),
            nn.Linear(hidden_dim, 1)
        )
        self.past_len = 5
        self.past_models = deque(maxlen=self.past_len)
        self.past_mean_rewards = deque(maxlen=self.past_len)
        self._last_flat_in = None
        #self.reader = JsonReader(self.input_files)
        #self.div_coef = 
        #self.rescale_factor = 
        #self.recent_model = []

    def forward(self, input_dict, state, seq_lens):
        #if obs[-1] > 0.99:
        #    self.recent_model.append((copy.deepcopy(self.a_model), copy.deepcopy(self.v_model)))
        #    if len(self.recent_model) > 5:
        #        self.recent_model.pop()
        obs = input_dict["obs_flat"].float()
        return self._forward(obs, input_dict, state, seq_lens)

    def _forward(self, obs, input_dict, state, seq_lens):
        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        self._output = self.a_model(self._last_flat_in)
        return self._output, state 

    def value_function(self):
        return self.v_model(self._last_flat_in).squeeze(1)

    # modified from https://github.com/ray-project/ray/blob/ac3371a148385027cf034e152d50f528acb853de/rllib/examples/models/custom_loss_model.py
    '''
    let's assume log_prob in diveristy and logits in ray are the same thing
    '''
    def custom_loss(self, policy_loss, loss_inputs):
        #div_loss, div_loss_orig = compute_div_loss(states, log_probs)
        """Calculates a custom loss on top of the given policy_loss(es).
        Args:
            policy_loss (List[TensorType]): The list of already calculated
                policy losses (as many as there are optimizers).
            loss_inputs (TensorStruct): Struct of np.ndarrays holding the
                entire train batch.
        Returns:
            List[TensorType]: The altered list of policy losses. In case the
                custom loss should have its own optimizer, make sure the
                returned list is one larger than the incoming policy_loss list.
                In case you simply want to mix in the custom loss into the
                already calculated policy losses, return a list of altered
                policy losses (as done in this example below).
        """
        # Get the next batch from our input files.
        #batch = self.reader.next()
        batch = loss_inputs
        # Define a secondary loss by building a graph copy with weight sharing.

        #batch["obs"]
        #batch["obs"].float()
        #torch.from_numpy(batch["obs"])
        #torch.from_numpy(batch["obs"]).float()
        #batch["obs"].float().to(policy_loss[0].device)
        #torch.from_numpy(batch["obs"]).float().to(policy_loss[0].device)
        obs = restore_original_dimensions(
            batch["obs"].float().to(policy_loss[0].device),
            #torch.from_numpy(batch["obs"]).float().to(policy_loss[0].device),
            self.obs_space,
            tensorlib="torch")
        logits, states = self.forward({"obs_flat": obs}, [], None)
        #logits, states = self.forward(loss_inputs, [], None)

        # You can also add self-supervised losses easily by referencing tensors
        # created during _build_layers_v2(). For example, an autoencoder-style
        # loss can be added as follows:
        # ae_loss = squared_diff(
        #     loss_inputs["obs"], Decoder(self.fcnet.last_layer))
        print("FYI: You can also use these tensors: {}, ".format(loss_inputs))

        #compute the diverity losss
        # treating logits as log_prob; maybe need correction
        #div_loss, div_loss_orig = self.compute_div_loss(states, logits)
        
        #now compute
        div_metric = nn.KLDivLoss(size_average=False, reduce=False)
        div_loss = 0
        past_ratios = [1.0 for r in range(len(self.past_models))] 
        divs = []
        for idx, past_model in enumerate(self.past_models):
           target_probs, states = past_model.forward({"obs_flat": obs}, [], None)        
           div = div_metric(logits, target_probs).sum(1)
           div = torch.clamp(div, min=-self.div_threshold, max=self.div_threshold)
           div = div.mean(0)
           divs.append(div)
        
        #self.div_loss_metric = div_loss.item()
        #self.policy_loss_metric = np.mean(
        #    [loss.item() for loss in policy_loss])

        divs_sort_idx = np.argsort([d.data[0] for d in divs])
        div_loss_orig = 0
        for idx in divs_sort_idx:
            #if self.use_neg_ratio and self.past_mean_reward_min != self.past_mean_reward_max:
            #    div_loss += float(-past_ratios[idx]) * divs[idx]
            #elif self.use_ratio and self.past_mean_reward_min != self.past_mean_reward_max:
            #    div_loss += (1.0 - float(past_ratios[idx])) * divs[idx]
            #elif self.rel and self.past_mean_reward_min != self.current_perf:
            #    div_loss += float(-past_ratios[idx]) * divs[idx]
            #else:
            div_loss += divs[idx]
            div_loss_orig += divs[idx]

        #if self.use_clip_div_loss:
        #    div_loss = torch.clamp(div_loss / float(len(self.past_models)), min=-self.div_max, max=self.div_max)
        #else:
        div_loss = div_loss / self.past_len# float(len(self.past_models))
        # Add the imitation loss to each already calculated policy loss term.
        # Alternatively (if custom loss has its own optimizer):
        # return policy_loss + [10 * self.imitation_loss]
        #return [loss_ - self.rescale_factor * self.div_coef * div_loss for loss_ in policy_loss]
        return [ loss_ - div_loss for loss_ in policy_loss ]

    def metrics(self):
        return {
            "policy_loss": self.policy_loss_metric,
            "imitation_loss": self.div_loss_metric,
        }

    '''
    Author: Zhang-Wei Hong 
    Email: williamd4112@gapp.nthu.edu.tw>
    Description: source code for paper
    “Diversity-driven exploration strategy for deep reinforcement learning” (NIPS 2018) 
    '''        
    '''
    def compute_div_loss(self, states, log_probs):
        div_metric = nn.KLDivLoss(size_average=False, reduce=False)
        # Div loss
        div_loss = 0
        #if args.use_neg_ratio or args.use_ratio or args.rel:
        #    max_perf = current_max_perf if args.use_history_max else current_perf
        #    min_perf = current_min_perf if args.use_history_max else current_perf
        #    past_mean_reward_max = max(max(past_mean_rewards), max_perf)
        #    past_mean_reward_min = min(min(past_mean_rewards), min_perf)
        #    past_mean_reward_rng = past_mean_reward_max - past_mean_reward_min + 1e-9

        #if args.use_neg_ratio:
        #    past_ratios = [((r - past_mean_reward_min) / past_mean_reward_rng) * 2 - 1 for r in past_mean_rewards]
        #elif args.use_ratio:
        #    past_ratios = [((r - past_mean_reward_min) / past_mean_reward_rng)  for r in past_mean_rewards]
        #elif args.rel:
        #    past_ratios = [((r - current_perf) / past_mean_reward_rng)  for r in past_mean_rewards]
        #else:
        #    past_ratios = [1.0 for r in range(len(past_models))]
        past_ratios = [1.0 for r in range(len(self.past_models))]

        divs = []
        for idx, past_model in enumerate(self.past_models):
            past_model.forward(states,)?????
            _, target_inds = t_probs.max(1)
            target_inds = target_inds.data
            action_size = t_probs.size()
            target_probs = Variable(torch.zeros(action_size[0], action_size[1]).cuda().scatter_(1, target_inds.unsqueeze(1), 1.0), requires_grad=False)

            div = div_metric(log_probs, target_probs).sum(1)
            div = torch.clamp(div, min=-self.div_threshold, max=self.div_threshold)
            div = div.mean(0)
            divs.append(div)

        divs_sort_idx = np.argsort([d.data[0] for d in divs])

        div_loss_orig = 0
        for idx in divs_sort_idx:
            if self.use_neg_ratio and self.past_mean_reward_min != self.past_mean_reward_max:
                div_loss += float(-past_ratios[idx]) * divs[idx]
            elif self.use_ratio and self.past_mean_reward_min != self.past_mean_reward_max:
                div_loss += (1.0 - float(past_ratios[idx])) * divs[idx]
            elif self.rel and self.past_mean_reward_min != self.current_perf:
                div_loss += float(-past_ratios[idx]) * divs[idx]
            else:
                div_loss += divs[idx]
            div_loss_orig += divs[idx]

        if self.use_clip_div_loss:
            div_loss = torch.clamp(div_loss / float(len(self.past_models)), min=-self.div_max, max=self.div_max)
        else:
            div_loss = div_loss / float(len(self.past_models))

        return div_loss, div_loss_orig / float(len(self.past_models))
        

        for idx, past_model in enumerate(self.past_models):
            temperature = args.temp
            actions = Variable(rollouts.actions.view(-1, action_shape), volatile=True)
            _, t_action_log_probs, _, t_log_probs, t_probs = past_model.evaluate_actions(Variable(states.data, volatile=True), actions, temperature=temperature)
            target_log_probs = t_log_probs
            target_log_probs = Variable(target_log_probs.data, requires_grad=False)

            if self.div_soft:
                target_probs = Variable(t_probs.data, requires_grad=False)
            else:
                _, target_inds = t_probs.max(1)
                target_inds = target_inds.data
                action_size = t_probs.size()
                target_probs = Variable(torch.zeros(action_size[0], action_size[1]).cuda().scatter_(1, target_inds.unsqueeze(1), 1.0), requires_grad=False)
            div = div_metric(log_probs, target_probs).sum(1)
            div = torch.clamp(div, min=-self.div_threshold, max=self.div_threshold)
            div = div.mean(0)
            divs.append(div)

        divs_sort_idx = np.argsort([d.data[0] for d in divs])
        if self.knn != 0:
            divs_sort_idx = divs_sort_idx[:self.knn]

        div_loss_orig = 0
        for idx in divs_sort_idx:
            if self.use_neg_ratio and self.past_mean_reward_min != self.past_mean_reward_max:
                div_loss += float(-past_ratios[idx]) * divs[idx]
            elif self.use_ratio and self.past_mean_reward_min != self.past_mean_reward_max:
                div_loss += (1.0 - float(past_ratios[idx])) * divs[idx]
            elif self.rel and self.past_mean_reward_min != self.current_perf:
                div_loss += float(-past_ratios[idx]) * divs[idx]
            else:
                div_loss += divs[idx]
            div_loss_orig += divs[idx]

        if self.use_clip_div_loss:
            div_loss = torch.clamp(div_loss / float(len(self.past_models)), min=-self.div_max, max=self.div_max)
        else:
            div_loss = div_loss / float(len(self.past_models))

        return div_loss, div_loss_orig / float(len(self.past_models))
        '''

ModelCatalog.register_custom_model("test_model", TestModel)
# RLlib does not work with gym registry, must redefine the environment in RLlib
# from cache_guessing_game_env_fix_impl_evict import * # for evict time attack
# from cache_guessing_game_env_fix_impl_flush import * # for flush reload attack
# from cache_guessing_game_env_fix_impl import * # for prime and probe attack
from cache_guessing_game_env_impl import *
# (Re)Start the ray runtime.
if ray.is_initialized():
  ray.shutdown()

'''
CacheSimulatorDiversityWrapper
description:
a gym compatible environment that records legit patterns in the buffer (under some condition)
when step() a guess, first look up whether the pattern is in the register 
if it is then the reward will be very negative
otherwise it will be positive
at the end output everything in the register 
'''
class CacheSimulatorDiversityWrapper(gym.Env):
    def __init__(self, env_config, keep_latency=False):
        self.env_config = env_config
        # two choices for memorize the table
        # 1. keep both the action and the actual latency
        # 2. just keep the action but not the latency
        #      in this case, the model has to be stored
        self.keep_latency = keep_latency
        self._env = CacheGuessingGameEnv(env_config)
        #     in this case the pattern has to be stored
        self.validation_env = CacheGuessingGameEnv(env_config)
        self.pattern_buffer = []
        self.enable_diversity = False
        self.repeat_pattern_reward = -10000
        self.correct_thre = 0.98
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.action_buffer = []
        #self.latency_buffer = []
        self.pattern_init_state = (copy.deepcopy(self._env.l1), self._env.victim_address)
        self.pattern_init_state_buffer = []
    '''
    reset function
    '''
    def reset(self):
        self.action_buffer = []
        self.validation_env.reset()
        rtn = self._env.reset()
        self.pattern_init_state = (copy.deepcopy(self._env.l1), self._env.victim_address)
        print("number of found patterns:" + str(len(self.pattern_buffer)))
        return rtn
    '''
    calculate the guessing correct rate of action_buffer
    '''
    def calc_correct_seq(self, action_buffer):
        correct_guess = 0
        total_guess = 0
        for _ in range(1):
            if self.replay_action_buffer(action_buffer):
                correct_guess += 1 
            total_guess += 1
        rtn = 1.0 * correct_guess / total_guess 
        print('calc_correct_seq ' + rtn)
        return rtn
    '''
    given the action and latency sequence, replay the cache game
    1. for known initial state, the loop runs once
    2. for randomized initial state, the loop runs multiple times
    ''' 
    def replay_action_buffer(self, action_buffer, init_state = None):
        replay_env = self.validation_env 
        while True:
            repeat = False                  # flag for not finding correct pattern
            replay_env.reset()
            if init_state != None:
               l1, victim_address = init_state
               replay_env.verbose = 1
               replay_env.l1 = l1
               replay_env.victim_address = victim_address
            for step in action_buffer:
               action = step[0]
               latency = step[1] 
               print(replay_env.parse_action((action)))
               obs, reward, _, _ = replay_env.step(action)
               print(obs)
               print(latency) 
               if obs[0] != latency:
                   repeat = True
            if repeat == True:
                continue
            else:
                break 
        if reward > 0:
            print("replay buffer correct guess")
            return True         # meaning correct guess
        else:
            return False        # meaning wrong guess
    '''
    chcek whether the sequence in the action_buffer is seen before
    returns true if it is
    returns false and update the pattern buffer 
    '''
    # shall we use trie structure instead ???
    def check_and_save(self):
        
        #return False
        '''
        if self.keep_latency == True:
            if self.action_buffer in self.pattern_buffer:
                return True
            else:
                if self._env.calc_correct_rate() > self.correct_thre:
                    print(self.action_buffer)
                    c = self.calc_correct_seq(self.action_buffer)
                    if c > self.correct_thre:
                        #print("calc_correct_seq")
                        self.pattern_buffer.append(self.action_buffer)
                        self.pattern_init_state_buffer.append(copy.deepcopy(self.pattern_init_state))
                        self.replay_pattern_buffer()
                return False        
        else:
        '''
        if self.action_buffer in self.pattern_buffer:
            return True
        else:
            if self._env.calc_correct_rate() > self.correct_thre:
                if self.keep_latency == True:  # need to calculate the latency of specific pattern
                    print(self.action_buffer)
                    c = self.calc_correct_seq(self.action_buffer)
                    if c > self.correct_thre:
                        #print("calc_correct_seq")
                        self.pattern_buffer.append(self.action_buffer)
                        self.pattern_init_state_buffer.append(copy.deepcopy(self.pattern_init_state))
                        self.replay_pattern_buffer()
                else:                           # just keep the pattern and the model
                    #c = self.calc_correct_seq(self.action_buffer) 
                    #if c > self.correct_thre: 
                    self.pattern_buffer.append(self.action_buffer)
                    print(self.pattern_buffer)
                    self._env.clear_guess_buffer_history()        
            return False        
    '''
    step function
    '''        
    def step(self,action):
        state, reward, done, info = self._env.step(action)
        #state = [state self._env.calc_correct_rate()]
        if self.keep_latency == True:
            latency = state[0]
        else:
            latency = -1
        self.action_buffer.append((action,latency)) #latnecy is part of the attack trace
        #make sure the current existing correct guessing rate is high enough beofre 
        # altering the reward
        if done == False:
            return state, reward, done, info
        else:
            action = self._env.parse_action(action)
            is_guess = action[1]
            if is_guess == True:
                is_exist = self.check_and_save()
                if is_exist == True and self.enable_diversity==True:
                    reward = self.repeat_pattern_reward #-10000
                return state, reward, done, info
            else:
                return state, reward, done, info
    ## pretty print the pattern buffer
    '''
    replay function
    '''
    def replay_pattern_buffer(self):
        print("replay pattern buffer")
        i = 0
        print(self.pattern_buffer)
        print(self.pattern_init_state_buffer)
        for i in range(0, len(self.pattern_buffer)):
            actions = self.pattern_buffer[i]
            self.replay_action_buffer(actions, init_state = self.pattern_init_state_buffer[i])
            print("attack pattern " + str(i))
            i += 1
        if i == 5:
            exit() 

ray.init(include_dashboard=False, ignore_reinit_error=True, num_gpus=1)
tune.register_env("cache_guessing_game_env_fix", CacheGuessingGameEnv)#Fix)
tune.register_env("cache_simulator_diversity_wrapper", CacheSimulatorDiversityWrapper)

# Two ways of training
# method 2b
config = {
    'env': 'cache_guessing_game_env_fix', #'cache_simulator_diversity_wrapper',
    'env_config': {
        'verbose': 1,
        "force_victim_hit": False,
        'flush_inst': False,
        "allow_victim_multi_access": False,
        "attacker_addr_s": 0,
        "attacker_addr_e": 7,#3,
        "victim_addr_s": 0,
        "victim_addr_e": 3,#1,
        "reset_limit": 1,
        "cache_configs": {
                # YAML config file for cache simulaton
            "architecture": {
              "word_size": 1, #bytes
              "block_size": 1, #bytes
              "write_back": True
            },
            "cache_1": {#required
              "blocks": 4,#2, 
              "associativity": 1,#2,  
              "hit_time": 1 #cycles
            },
            "mem": {#required
              "hit_time": 1000 #cycles
            }
        }
    }, 
    #'gamma': 0.9, 
    'num_gpus': 2, 
    'num_workers': 16, 
    'num_envs_per_worker': 2, 
    #'entropy_coeff': 0.001, 
    #'num_sgd_iter': 5, 
    #'vf_loss_coeff': 1e-05, 
    'model': {
        #'custom_model': 'test_model',#'rnn', 
        #'max_seq_len': 20, 
        #'custom_model_config': {
        #    'cell_size': 32
        #   }
    }, 
    'framework': 'torch',
}
#trainer = PPOTrainer(config=config)
#exit(0)
if __name__ == "__main__":
    import signal
    import sys

    #tune.run(PPOTrainer, config=config)#config={"env": 'Freeway-v0', "num_gpus":1})
    from ray.tune.logger import pretty_print
    #trainer = PPOTrainer(config=config)
    trainer = SACTrainer(config=config)

    def signal_handler(sig, frame):
        print('You pressed Ctrl+C!')
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    
    while True:
       # Perform one iteration of training the policy with PPO
       result = trainer.train()
       print(pretty_print(result))

       #if True: #i % 100 == 0:
       #    checkpoint = trainer.save()
       #    print("checkpoint saved at", checkpoint)


    # restore checkpoint
    #trainer2 = PPOTrainer(config=config)
    #trainer2.restore(checkpoint)
