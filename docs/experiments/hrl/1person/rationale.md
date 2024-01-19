
<h2>Hierarchical Agent for Two Persons</h2>
The experiment implemented in this project is based on the Q-Learning algorithm, serving as a performance benchmark for Hierarchical Reinforcement Learning (HRL) agents. The HRL agents are designed to operate in an environment modeled after the Taxi problem, with a focus on hierarchical structures and task representation. The core Markov Decision Process (MDP) is denoted as "Root," overseeing the hierarchy.
The algorithm used is MaxQ-0, which is a hierarchical extension of Q-Learning, with an Epsilon Greeedy policy without decay.

<h2>Task Representation</h2>
The experiment explores the representation of the "Navigate" task, a key component of the HRL framework. Previous works have proposed different parameterizations for the task, including representing the destination or passenger location. The implementation considers two options, with a primary emphasis on the representation where "f" corresponds to the place to navigate.


<h2>Complexity Considerations</h2>
The choice of task representation is influenced by a desire to minimize the cardinality of possible parameters. This is crucial, given that the "Navigate" task must be learned for each parameter, and increasing cardinality implies increased complexity. While an alternative representation is explored for its versatility, the primary focus is on the first option due to its potential for reduced training episodes.


<h2>Policy Implementation</h2>
Across all experiments, an Epsilon Greedy policy with a fixed epsilon value is employed during training. This policy guides the agent's exploration-exploitation trade-off throughout the learning process.


<h2>Termination Conditions</h2>
The termination conditions for tasks vary, with the "Root" task concluding when the episode is either successfully resolved or reaches the maximum number of iterations. The "Get" and "Put" tasks terminate based on the passenger's presence in the taxi. For the "Navigate" task, termination occurs when the taxi's coordinates match the target, and the passenger is either in or out of the taxi, depending on the task variant.


<h2>Command Line Arguments</h2>

* --experiment (type: str, default: None):
Specifies the name of the experiment. If the experiment name doesn't start with "hrl_transfer2nav" or is not provided, the script assumes that training is required.


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
When provided, this flag indicates that the environment should be rendered during training.