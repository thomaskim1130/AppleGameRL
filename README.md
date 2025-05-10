# Structure
```
AppleGameRL/
├── AppleGame.py
├── README.md
├── agents
│   ├── __pycache__
│   │   └── random_agent.cpython-311.pyc
│   ├── dqn_agent.py
│   └── random_agent.py
└── scripts
    ├── apple_game_dqn.sh
    ├── apple_game_play.sh
    └── apple_game_random.sh
```

# Adding new agents
```
class AgentTemplate:
    """
    Base template for any RL agent.
    """

    def __init__(self, env, **kwargs):
        """
        Initialize the agent.

        Args:
            env:           A Gymnasium environment instance.
            kwargs:        Additional hyperparameters (e.g., learning rate).
        """
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        # e.g. initialize networks, optimizers, replay buffers here:
        # self.model = ...
        # self.optimizer = ...
        # self.buffer = ...

    def get_action(self, state):
        """
        Select an action given the current state.

        Args:
            state:   The current observation from the environment.

        Returns:
            action:  An action sampled from the policy.
        """
        raise NotImplementedError("get_action must be implemented by subclass")

    def store_transition(self, state, action, reward, next_state, done):
        """
        Store a transition for later learning (optional).

        Args:
            state:       Current state.
            action:      Action taken.
            reward:      Reward received.
            next_state:  Next state observed.
            done:        Whether the episode ended.
        """
        # override if your algorithm uses replay / memory
        pass

    def update(self):
        """
        Update the agent's parameters (optional).

        Typically called after storing a transition.
        """
        # override to implement learning step
        pass

    def load(self, path):
        """
        Load saved parameters from disk (optional).

        Args:
            path:  Path to the saved model file.
        """
        # override to load model weights
        pass

    def save(self, path):
        """
        Save current parameters to disk (optional).

        Args:
            path:  Path where to save the model file.
        """
        # override to save model weights
        pass
```