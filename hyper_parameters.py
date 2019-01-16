


from dqn_globals import DqnGlobals 


class AgentParameters:
    def __init__(self,
                 state_size,
                 epsilon_start, 
                 epsilon_min, 
                 epsilon_decay_step,
                 delayTraining,
                 update_target_rate,
                 gamma,
                 learning_rate):
        self.state_size = state_size
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay_step = epsilon_decay_step
        self.delayTraining = delayTraining
        self.update_target_rate = update_target_rate
        self.gamma = gamma
        self.learning_rate = learning_rate
        

StandardAgentParameters = AgentParameters(
        state_size = DqnGlobals.STATE_DIMENSIONS,
        epsilon_start = 1.0,
        epsilon_min = 0.05,
        epsilon_decay_step = 0.000002,
        delayTraining = 20000,
        update_target_rate = 10000,
        gamma = 0.99,
        learning_rate = 0.00001
        )

class MiscParameters:
    def __init__(self, createGifEveryXEpisodes):
        self.createGifEveryXEpisodes = createGifEveryXEpisodes


ShortEpisodeParameters = MiscParameters(createGifEveryXEpisodes=500)
LongEpisodeParameters = MiscParameters(createGifEveryXEpisodes=100)


