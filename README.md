RL environment and training script for CacheSimulator
==============
This repo contains wrapper for the CacheSimulator environment based on 
https://github.com/auxiliary/CacheSimulator

For detailed description of CacheSimulator, see the original repo.

The environment is based on openai [gym](https://github.com/openai/gym). The details of the environment has been described in our [paper](docs/paper_micro.pdf)

```
$ pip install gym
```

The trainer is based on [RLlib](https://www.ray.io/rllib) or [RLMeta](https://github.com/facebookresearch/rlmeta). The environment can be trained on both RLLib and RLMeta, and we have provided some scripts in ```src/rllib``` and ```src/rlmeta``` correspondingly.

## RLLib Setup and Experiments

Here we show how to launch an experiments in RLlib

First, we assume the ```conda``` is already installed, and we provide a script to install all the depedencies using ```conda```. 

```
$ cd ${GIT_ROOT}/src/rllib
$ bash deploy_conda_rllib.sh
```


To run the training

```
$ cd ${GIT_ROOT}/src/rllib
$ python run_gym_rllib.py
```

To stop the training, just do ```Ctrl+C```, a checkpoint will be saved at default location in

```
~/ray_results
```

To view the training processes in realtime, RLLib provides [tensorboard](https://tensorboard.dev) support. To launch tensorboard

```
$ tensorboard --logdir=~/ray_results/
```

and open the browser, by default, the url is ```localhost:6006```.


To replay the checkpoint, do

```
$ cd ${GIT_ROOT}/src/rllib
$ python replay_checkpoint.py <path_to_the_checkpoint>
```

More documentation can be found at [docs](docs).

## RLMeta Setup and experiments

Please follow setup process on [rlmeta](https://github.com/facebookresearch/rlmeta) for install RLMeta. 

After install rlmeta, you can launch the experiment to train the RL agent

```
$ cd ${GIT_ROOT}/src/rlmeta
$ python train_ppo_transformer.py
```

## Repo Structure 

```
-configs            # this is the directory for CacheSimulotor configuration
-docs               # documentations
-env_test           # contains testing suit for simulator and replacement policy
-src
 |--config          # gym environment configurations
 |--cyclone_data    # data for training cyclone svm classifier
 |--fig             # positions for storing the figure
 |--models          # customized pytorch models for the RL agent to use
 |--rllib           # scripts for launching RLLib based experiments
 |--rlmeta          # scripts for launching RLMeta-basee experiments
 |--setup_scripts   # some scripts for setup the environment
 |--cache.py        # the cache logic implementation
 |--cache_simulator.py              # the interface of the cache simulator
 |--replacement_policy.py           # define the replacement policy for the cache
 |--cache_guessing_game_env_impl.py # the gym implementation of the cache
 |--cchunter_wrapper.py             # the wrapper that implements cchunter attack detector
 |--cyclone_wrapper.py              # the wrapper that implements cyclone attack detector
-third_party        # position for third-party libraries like 
-traces             # places for traces of CacheSimulator
```
## Contact

Please direct any questions to Mulong Luo ```ml2558@cornell.edu```.
