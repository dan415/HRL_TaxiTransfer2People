<h2>Q-Learning Implementation></h2>

The Q-Learning algorithm is implemented in the `qlearning` module. Although I have specifically created a version
for 1 person and 2 persons, it is, in essence, the same algorithm. This implementation is provided as a baseline for
comparing the performance of the HRL agents.

The policy implemented is decaying Epsilon Greedy, with a fixed epsilon value.


The experiment folder is `qlearning`, and it contains the following files:

* `config.json`: Contains the configuration of the experiment. It is used for tracking the experiment parameters. It saves the following information:
    ```json
    {
        "learning_rate": 0.1,
        "discount_rate": 0.99,
        "decay_rate": 0.001,
        "episodes": 10000,
        "steps": 300,
        "render_training": false,
        "show_plot": true
    }
    ```
  
* `plot.png`: plot of training and test rewards. See Utils section for more information.
* `plot.html`: HTML version of the plot
* `qtable.npy`: Q-table (2 dmensional state-action (500 x n_actions) matrix) for the experiment. Needed for testing the agent. n_actions can be either 13 or 15, depending on the Navigation Task design.
  



<h3>Command Line Arguments</h3>

* --qtable (type: str, default: None): Specifies the name of the Q-table to be used for training or testing the agent. If the Q-table is not provided, the script assumes that training is required.
* --episodes (type: int, default: 10000): Sets the number of episodes for training or testing the agent.
* --steps (type: int, default: 300): Defines the maximum number of steps per episode.
* --discount_rate (type: float, default: 0.99): Sets the discount rate (gamma), which influences the importance of future rewards in the agent's decision-making.
* --decay_rate (type: float, default: 0.001): Sets the decay rate for the epsilon value. It determines how much the epsilon value decreases after each episode, controlling exploration vs exploitation.
* --learning_rate (type: float, default: 0.1): Sets the learning rate (alpha) for the agent. It determines how much the agent updates its Q-values based on new experiences.
* --epsilon (type: float, default: 0.1): Sets the exploration-exploitation factor (epsilon). It controls the likelihood of the agent choosing a random action instead of exploiting its current knowledge.
* --show_plot (action: store_true): When provided, this flag indicates that a plot of training and test rewards should be displayed after the experiment.