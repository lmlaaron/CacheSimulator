#!/bin/bash
# implement https://sites.google.com/view/diayn/ based diversity

from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from gym import spaces

class SKillCallbacks(DefaultCallbacks):
        def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        # Make sure this episode has just been started (only initial obs
        # logged so far).
        assert episode.length == 0, (
            "ERROR: `on_episode_start()` callback should be called right "
            "after env reset!"
        )
        print("episode {} (env-idx={}) started.".format(episode.episode_id, env_index))
        episode.user_data["pole_angles"] = []
        episode.hist_data["pole_angles"] = []
        
        # select the value z
        policies[DEFAULT_POLICY_ID].sample_z()

class SkillPPOTorchPolicy(PPOTorchPolicy): 
    def __init__(self, observation_space, action_space, config):
        self.discriminator = 
        if "num_skills" in config:
            self.num_skills = config["num_skills"]
        else:
            self.num_skills = 20
        self.z_space = spaces.Discrete(self.num_skills)
        self.z = self.z_space.sample()

        # initialize the discrimininator
        self.discriminator = ModelCatalog.get_model_v2(
            obs_space = self.observation_space, # not augmented obs space
            action_space = self.z_space,
            num_outputs = self.z_space.n,
            model_config = MODEL_DEFAULTS,
            framework = "torch",
            name="z_discriminator",
        )

        # augment the observation space with z
        # assuming original observation space is multidiscrete
        assert( type(observation_space) == spaces.MultiDiscrete)
        self.aug_observation_space = spaces.MultiDiscrete(tuple(list(observation_space.shape()).append(self.num_skills)))
        #initialize with the augmented observation space
        super(CustomPPOTorchPolicy, self).__init__(self.aug_observation_space, action_space, config)

    def sample_z(self):
        self.z = self.z_space.sample()
        return self.z

    def loss(self, model: ModelV2, dist_class: Type[ActionDistribution],
            train_batch: SampleBatch, extern_trigger = True ) -> Union[TensorType, List[TensorType]]:

        # aug_train_batch = (train_batch, zs)
        # augment the observations       
        aug_train_batch = copy.deepcopy(train_batch)
        aug_train_batch["skill"] =[]
        for obs in train_batch[SampleBatch.OBS]:
            aug_obs = obs # augment obs with z
            z_one_hot = np.zeros(self.num_skills)
            z_one_hot[self.z] = 1
            aug_train_batch[SampleBatch.OBS] = np.hstack([obs, z_one_hot])
            aug_train_batch["skill"].append(z_one_hot)

        #model will be adapt to add z
        total_loss = PPOTorchPolicy.loss(self, model, dist_class, aug_train_batch)
        
        # find the disriminiator loss
        dis_logits, _ = self.discriminator.from_batch(train_batch)
        values = self.discriminator.value_function()
        valid_mask = torch.ones_like(values, dtype=torch.bool)
        dist = dist_class(dis_logits, self.discriminator)
        log_probs = dist.logp(aug_train_batch[SampleBatch.OBS])
        
        
        #log_probs = dist.logp(train_batch[SampleBatch.ACTIONS])#.reshape(-1) 


        div_loss = log_z .... - log_pz

        # the discriminator and "model" are in the same position
        # so the discriminator will be retrained simultaneously

        print('total_loss')
        print(total_loss)
        print('div_loss')
        print(div_loss)
        #assert(False)
        ret_loss = total_loss - 0.03 * div_loss
        return ret_loss
        '''
        new_loss = []
        if issubclass(type(total_loss),TensorType):
            return total_loss - compute_div_loss(self, model, dist_class, train_batch)
        else:            
            for loss in total_loss:
                new_loss.append(loss - compute_div_loss(self, model, dist_class, train_batch))
            return new_loss
        '''