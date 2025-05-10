class RandomAgent:
    """
    A random agent that ignores the state and samples uniformly.
    """
    def __init__(self, env):
        self.action_space = env.action_space

    def get_action(self, state):
        return self.action_space.sample()
