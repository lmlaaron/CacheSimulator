import copy
import logging
import time

import hydra

import torch
import torch.multiprocessing as mp

import rlmeta.envs.gym_wrappers as gym_wrappers
import rlmeta.utils.hydra_utils as hydra_utils
import rlmeta.utils.remote_utils as remote_utils

from rlmeta.agents.agent import AgentFactory
from rlmeta.agents.ppo.ppo_agent import PPOAgent
from rlmeta.core.controller import Phase, Controller, DummyController
from rlmeta.core.maloop import LoopList, MAParallelLoop
from rlmeta.core.model import wrap_downstream_model
from rlmeta.core.replay_buffer import ReplayBuffer, make_remote_replay_buffer
from rlmeta.core.server import Server, ServerList
from rlmeta.core.callbacks import EpisodeCallbacks
from rlmeta.core.types import Action, TimeStep

from cache_env_wrapper import CacheAttackerDetectorEnvFactory
from cache_ppo_transformer_model import CachePPOTransformerModel
# from cache_ppo_transformer_model_pe import CachePPOTransformerModel
from metric_callbacks import MACallbacks

from utils.wandb_logger import WandbLogger, stats_filter

from agents.random_agent import RandomAgent
from agents.benign_agent import BenignAgent
# @hydra.main(config_path="./config", config_name="ppo_lru_8way")
# @hydra.main(config_path="./config", config_name="ppo_2way_2set")
# @hydra.main(config_path="./config", config_name="ppo_4way_4set")
# @hydra.main(config_path="./config", config_name="ppo_8way_8set")
@hydra.main(config_path="./config", config_name="ppo_exp")
# @hydra.main(config_path="./config", config_name="ppo_exp_ceaser")
# @hydra.main(config_path="./config", config_name="ppo_cchunter_baseline")
def main(cfg):
    wandb_logger = WandbLogger(project="cache_attack_detect", config=cfg)
    my_callbacks = MACallbacks()
    logging.info(hydra_utils.config_to_json(cfg))

    env_fac = CacheAttackerDetectorEnvFactory(cfg.env_config)
    env = env_fac(0)
    #### attacker
    cfg.model_config["output_dim"] = env.action_space.n
    train_model = CachePPOTransformerModel(**cfg.model_config).to(
        cfg.train_device)
    optimizer = torch.optim.Adam(train_model.parameters(), lr=cfg.lr)

    infer_model = copy.deepcopy(train_model).to(cfg.infer_device)
    infer_model.eval()

    ctrl = Controller()
    rb = ReplayBuffer(cfg.replay_buffer_size)
    #### detector 
    cfg.model_config["output_dim"] = 2
    train_model_d = CachePPOTransformerModel(**cfg.model_config).to(
        cfg.train_device_d)
    optimizer_d = torch.optim.Adam(train_model_d.parameters(), lr=cfg.lr)

    infer_model_d = copy.deepcopy(train_model_d).to(cfg.infer_device_d)
    infer_model_d.eval()

    ctrl_d = DummyController()
    rb_d = ReplayBuffer(cfg.replay_buffer_size)
    
    #### start server
    m_server = Server(cfg.m_server_name, cfg.m_server_addr)
    r_server = Server(cfg.r_server_name, cfg.r_server_addr)
    c_server = Server(cfg.c_server_name, cfg.c_server_addr)
    m_server.add_service(infer_model)
    r_server.add_service(rb)
    c_server.add_service(ctrl)
    md_server = Server(cfg.md_server_name, cfg.md_server_addr)
    rd_server = Server(cfg.rd_server_name, cfg.rd_server_addr)
    cd_server = Server(cfg.cd_server_name, cfg.cd_server_addr)
    md_server.add_service(infer_model_d)
    rd_server.add_service(rb_d)
    cd_server.add_service(ctrl_d)
    servers = ServerList([m_server, r_server, c_server, md_server, rd_server, cd_server])

    a_model = wrap_downstream_model(train_model, m_server)
    t_model = remote_utils.make_remote(infer_model, m_server)
    e_model = remote_utils.make_remote(infer_model, m_server)

    #### TODO:What does control do?
    a_ctrl = remote_utils.make_remote(ctrl, c_server)
    t_ctrl = remote_utils.make_remote(ctrl, c_server)
    e_ctrl = remote_utils.make_remote(ctrl, c_server)
    a_ctrl_d = remote_utils.make_remote(ctrl_d, cd_server)
    t_ctrl_d = remote_utils.make_remote(ctrl_d, cd_server)
    e_ctrl_d = remote_utils.make_remote(ctrl_d, cd_server)


    a_rb = make_remote_replay_buffer(rb, r_server, prefetch=cfg.prefetch)
    t_rb = make_remote_replay_buffer(rb, r_server)

    agent = PPOAgent(a_model,
                     replay_buffer=a_rb,
                     controller=a_ctrl,
                     optimizer=optimizer,
                     batch_size=cfg.batch_size,
                     learning_starts=cfg.get("learning_starts", None),
                     entropy_coeff=cfg.get("entropy_coeff", 0.01),
                     push_every_n_steps=cfg.push_every_n_steps)
    t_agent_fac = AgentFactory(PPOAgent, t_model, replay_buffer=t_rb)
    e_agent_fac = AgentFactory(PPOAgent, e_model, deterministic_policy=True)
    #### random detector 
    '''
    detector = RandomAgent(2)
    t_d_fac = AgentFactory(RandomAgent, 2)
    e_d_fac = AgentFactory(RandomAgent, 2)
    '''
    #### random benign agent
    benign = BenignAgent(env.action_space.n)
    t_b_fac = AgentFactory(BenignAgent, env.action_space.n)
    e_b_fac = AgentFactory(BenignAgent, env.action_space.n)

    #### detector agent
    a_model_d = wrap_downstream_model(train_model_d, md_server)
    t_model_d = remote_utils.make_remote(infer_model_d, md_server)
    e_model_d = remote_utils.make_remote(infer_model_d, md_server)

    a_rb_d = make_remote_replay_buffer(rb_d, rd_server, prefetch=cfg.prefetch)
    t_rb_d = make_remote_replay_buffer(rb_d, rd_server)

    agent_d = PPOAgent(a_model_d,
                     replay_buffer=a_rb_d,
                     controller=a_ctrl,
                     optimizer=optimizer_d,
                     batch_size=cfg.batch_size,
                     learning_starts=cfg.get("learning_starts", None),
                     entropy_coeff=cfg.get("entropy_coeff", 0.01),
                     push_every_n_steps=cfg.push_every_n_steps)
    t_d_fac = AgentFactory(PPOAgent, t_model_d, replay_buffer=t_rb_d)
    e_d_fac = AgentFactory(PPOAgent, e_model_d, deterministic_policy=True)

    #### create agent list 
    t_ma_fac = {"benign":t_b_fac, "attacker":t_agent_fac, "detector":t_d_fac}
    e_ma_fac = {"benign":e_b_fac, "attacker":e_agent_fac, "detector":e_d_fac}

    t_loop = MAParallelLoop(env_fac,
                          t_ma_fac,
                          t_ctrl, #TODO 
                          running_phase=Phase.TRAIN,
                          should_update=True,
                          num_rollouts=cfg.num_train_rollouts,
                          num_workers=cfg.num_train_workers,
                          seed=cfg.train_seed,
                          episode_callbacks=my_callbacks)
    e_loop = MAParallelLoop(env_fac,
                          e_ma_fac,
                          e_ctrl, #TODO
                          running_phase=Phase.EVAL,
                          should_update=False,
                          num_rollouts=cfg.num_eval_rollouts,
                          num_workers=cfg.num_eval_workers,
                          seed=cfg.eval_seed,
                          episode_callbacks=my_callbacks)
    loops = LoopList([t_loop, e_loop])

    servers.start()
    loops.start()
    agent.connect()
    agent_d.connect()
    a_ctrl.connect()

    start_time = time.perf_counter()
    for epoch in range(cfg.num_epochs):
        a_stats, d_stats = None, None 
        a_ctrl.set_phase(Phase.TRAIN, reset=True)
        if epoch>40:
            d_stats = agent_d.train(cfg.steps_per_epoch) #TODO
        else:
            a_stats = agent.train(cfg.steps_per_epoch)
        #stats = d_stats
        stats = a_stats or d_stats

        cur_time = time.perf_counter() - start_time
        info = f"T Epoch {epoch}"
        if cfg.table_view:
            logging.info("\n\n" + stats.table(info, time=cur_time) + "\n")
        else:
            logging.info(
                stats.json(info, phase="Train", epoch=epoch, time=cur_time))
        if epoch>40:
            train_stats = {"detector":d_stats}
        else:
            train_stats = {"attacker":a_stats}
        time.sleep(1)
        
        a_ctrl.set_phase(Phase.EVAL, limit=cfg.num_eval_episodes, reset=True) #TODO: what is num_episodes doing
        a_stats = agent.eval(cfg.num_eval_episodes)
        d_stats = agent_d.eval(cfg.num_eval_episodes) #TODO
        #stats = d_stats
        stats = a_stats

        cur_time = time.perf_counter() - start_time
        info = f"E Epoch {epoch}"
        if cfg.table_view:
            logging.info("\n\n" + stats.table(info, time=cur_time) + "\n")
        else:
            logging.info(
                stats.json(info, phase="Eval", epoch=epoch, time=cur_time))
        eval_stats = {"attacker":a_stats, "detector":d_stats}
        #eval_stats = {"attacker":a_stats}
        time.sleep(1)
        
        wandb_logger.log(train_stats, eval_stats)

        torch.save(train_model.state_dict(), f"ppo_agent-{epoch}.pth")

    loops.terminate()
    servers.terminate()

def add_prefix(input_dict, prefix=''):
    res = {}
    for k,v in input_dict.items():
        res[prefix+str(k)]=v
    return res

if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()