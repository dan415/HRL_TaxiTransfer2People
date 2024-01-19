
<h2>Hierarchical Agent for Two Persons</h2>
For the hierarchical agent designed to handle two persons, the transfer learning process involves freezing policies for all tasks except "Root." This enables the agent to learn a policy that determines which action and person choice minimize execution time. The state translation process is integral during state transitions and recursive policy evaluation to ensure the preservation of state integrity.




<h2>Transfer Learning</h2>
The project incorporates transfer learning, starting with training an agent for the Taxi V3.0 environment. To facilitate transfer, knowledge from the "Get" and "Put" tasks is duplicated in the corresponding tables, while the "Root" task's content is overwritten or removed. The transfer involves transitioning from 500 to 10,000 states, necessitating a freeze on policies for all tasks except "Root." The translation of states between hierarchical levels is crucial during both state transitions and policy evaluation to maintain state integrity.



We first need to train an agent for the Taxi V3.0 environment. In order to achieve this,
we provide the maxq experiment implementations. Please note that, as we provide two different options with 
designs of the Navigation Task, the appropiate version of the experiment should be used for its corresponding extension for two persons.


Then, we will have on a experiments folder a subfolder generated with the name of the experiment (it's based on execution datetime).
Inside this folder, we will have the following files:
* `config.json`: Contains the configuration of the experiment. It is used for tracking the experiment parameters. It saves the following information:
    ```json
    {
        "experiment_name": "hrl_transfer2nav_2p",
        "episodes": 10000,
        "steps": 300,
        "alpha": 0.2,
        "gamma": 1,
        "epsilon": 0.004,
        "show_plot": true,
        "render_training": false
    }
    ```

* `plot.png`: plot of training and test rewards. See Utils section for more information.
* `plot.html`: HTML version of the plot
* `ctable.npy`: C-table (3 dimensional state-action-state (500 x n_actions x 500) matrix) for the experiment. Needed for testing the agent. n_actions can be either 13 or 15, depending on the Navigation Task design.
* `vtable.npy`: V-table (2 dmensional state-action (500 x n_actions) matrix) for the experiment. Needed for testing the agent. n_actions can be either 13 or 15, depending on the Navigation Task design.

Then, we need to provide the experiment name to the transfer learning experiment. The experiment name is the name of the folder generated in the previous step. 
The agent, when passing it this low dimensional tables, will automatically expand them to the 10.000 states. This process involves copying
the GET, PUT tasks and adding them as new task (the first two will be used for passenger 1, the second two for the passenger 2). The Root task will be overwritten, and will need to relearned.


This here is the main purpose of the transfer. By doing this, we can have all the hierarchy learned except for the Root Task, and we will only need to train for it. The Root task
will learn how and when to select a passenger and the GET or PUT action. This reduces training time significantly, as oppose to learning the whole hierarchy from scratch.
When training the Root task, the rest of the hierarchy will be frozen, and will not be updated. 


The training will be saved in the same manner as the previous experiment, but with the new experiment name. The new experiment name will be the same as the previous one, but the states dimension will be 10.000.


<h2>Command Line Arguments</h2>

* --experiment (type: str, default: None):
Specifies the name of the experiment. If the experiment name is not provided, the script assumes that training is required, and will train from scrathc. If
the dimension of the states in the provided tables is 500, it will assume that the experiment is for the Taxi V3.0 environment. If the dimension is 10.000, it will assume that the experiment needs transfer learning, and 
will create the tables as described in the previous section, and will train the Root task from scratch. If the dimension is 10.000 a it will assume that the experiment is for the Taxi2P environment, and will not train, but will test the agent, rendering the environment.


* --episodes (type: int, default: 10000):
Sets the number of episodes for training or testing the agent.


* --steps (type: int, default: 300):
Defines the maximum number of steps per episode.


* --alpha (type: float, default: 0.2):
Sets the learning rate (alpha) for the agent. It determines how much the agent updates its Q-values based on new experiences.


* --gamma (type: float, default: 1):
Sets the discount factor (gamma), which influences the importance of future rewards in the agent's decision-making.


* --epsilon (type: float, default: 0.004):
Sets the exploration-exploitation factor (epsilon). It controls the likelihood of the agent choosing a random action instead of exploiting its current knowledge.


* --show_plot (action: store_true):
When provided, this flag indicates that a plot of training and test rewards should be displayed after the experiment.


* --render_training (action: store_true):
When provided, this flag indicates that the training process should be visually rendered, showing the agent's interaction with the environment.