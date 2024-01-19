
<h2>Taxi2PEnv</h2>



Taxi2PEnv is an extension of the classic Taxi-v3 environment. This modified environment supports two passengers, each with its own destination. The environment has 6 discrete deterministic actions:


0: move south
1: move north
2: move east
3: move west
4: pickup passenger
5: drop off passenger


<h3>Observation Space</h3>


There are 10.000 discrete states based on the taxi's position, passenger locations, and destinations.
The state space is represented by the tuple (taxi_row, taxi_col, pass_loc1, pass_loc2, dest_idx1, dest_idx2).


The original Taxi-v3 environment has a state space of 500 discrete states, represented by the tuple (taxi_row, taxi_col, pass_loc, dest_idx).
Adding a new passenger implies adding a new destination to the environment, and a new passenger location. The passenger location
can take any of the 5 values (R, G, Y, B, and in taxi), and the destination can take any of the 4 values (R, G, Y, B).
Therefore, we would be incrementing the state space by 20 (5 * 4).


<h3>Action Space</h3>


There are 6 discrete deterministic actions, as described above.
Rewards:


* -1 per step unless other rewards are triggered.
* +20 for delivering both passengers.
* -10 for executing "pickup" and "drop-off" actions illegally. It is considered illegal to pick up a passenger that has already been
dropped off at their destination.
* +10 for delivering the first passenger.


There is no restriction to the initialization of states, except for states where any of the passengers
spawn inside the taxi or at their destination.


<h2>Translation of States</h2>
The translation of states between hierarchical levels is crucial during both state transitions and policy evaluation to maintain state integrity. The translation process is implemented as follows:


Translation of states from the high dimensional space is done by selecting what passenger to translate the state for.
Once the passenger is selected, the state is translated in the same way as Taxi-V3 environment. Encoding functions, as usually, need to be defined by all the state space attributes.


These methods are provided as utils in the package:


### encode_taxi1P
Encodes the state of the environment for a single passenger.

**Parameters:**
- `taxi_row` (int) – Row of the taxi
- `taxi_col` (int) – Column of the taxi
- `pass_loc` (int) – Location of the passenger
- `dest_idx` (int) – Index of the destination

**Returns:**
- `state` (int) – Encoded state

### decode_taxi1P
Decodes the state of the environment for a single passenger.

**Parameters:**
- `state` (int) – Encoded state

**Returns:**
Tuple of:
- `taxi_row` (int) – Row of the taxi
- `taxi_col` (int) – Column of the taxi
- `pass_loc` (int) – Location of the passenger
- `dest_idx` (int) – Index of the destination

### encode_taxi2P
Encodes the state of the environment for two passengers.

**Parameters:**
- `taxi_row` (int) – Row of the taxi
- `taxi_col` (int) – Column of the taxi
- `pass_loc1` (int) – Location of the first passenger
- `pass_loc2` (int) – Location of the second passenger
- `dest_idx1` (int) – Index of the first destination
- `dest_idx2` (int) – Index of the second destination

**Returns:**
- `state` (int) – Encoded state

### decode_taxi2P
Decodes the state of the environment for two passengers.

**Parameters:**
- `state` (int) – Encoded state

**Returns:**
Tuple of:
- `taxi_row` (int) – Row of the taxi
- `taxi_col` (int) – Column of the taxi
- `pass_loc1` (int) – Location of the first passenger
- `pass_loc2` (int) – Location of the second passenger
- `dest_idx1` (int) – Index of the first destination
- `dest_idx2` (int) – Index of the second destination

### translate
Translates the state of the environment from the high dimensional space to the low dimensional space.

**Parameters:**
- `state` (int) – Encoded state
- `passenger` (int) – Passenger to translate the state for (it can be 1 or 2)

**Returns:**
- `state` (int) – Translated encoded state in low dimensional space


Note: The maximum number of steps allowed for this environment is set to 1000.



