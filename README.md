RL environment and training script for CacheSimulator
==============
This repo contains wrapper for the CacheSimulator environment based on 
https://github.com/auxiliary/CacheSimulator

For detailed description of CacheSimulator, see the original repo.

The environment is based on openai gym

```
$ pip install gym
```

The trainer is based on RLlib and RLMeta

```
$ pip install rllib
```

To run the training

```
$ cd src/
$ python run_gym_rllib.py
```


### Repo Structure 

```
-configs   # this is the directory for CacheSimulotor configuration
-docs      # documentations
-env_test  # contains testing suit for simulator and replacement policy
-src
 |--config # gym environment configurations
 |--cyclone_data # data for training cyclone svm classifier
 |--fig    # positions for storing the figure
 |--models # customized pytorch models for the RL agent to use
 |--rllib  # scripts for launching RLLib based experiments
 |--rlmeta # scripts for launching RLMeta-basee experiments
 |--setup_scripts # some scripts for setup the environment
-third_party # position for third-party libraries like 
-traces    # places for traces of CacheSimulator
```

