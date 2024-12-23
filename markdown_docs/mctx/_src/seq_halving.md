## FunctionDef score_considered(considered_visit, gumbel, logits, normalized_qvalues, visit_counts)
**score_considered**: The function of score_considered is to compute a score that can be used for selecting actions via an argmax operation.

parameters: 
· considered_visit: An integer representing the number of visits required for a child node to be considered.
· gumbel: A JAX array containing Gumbel noise values, which are used to introduce randomness in the selection process.
· logits: A JAX array of prior logits associated with each action, typically derived from a policy network.
· normalized_qvalues: A JAX array of normalized Q-values for each action, representing the expected future rewards.
· visit_counts: A JAX array indicating the number of times each child node has been visited.

Code Description: The function score_considered calculates a composite score for each action by combining Gumbel noise, prior logits, and normalized Q-values. It first normalizes the logits to prevent numerical instability by subtracting the maximum logit value from all elements in the logits array. A penalty is then applied to actions that have not been visited enough times (i.e., their visit count does not match considered_visit), effectively setting their scores to negative infinity and making them ineligible for selection. The function ensures that all input arrays have the same shape using chex.assert_equal_shape before computing the final score, which is the element-wise maximum of a low logit value (-1e9) and the sum of Gumbel noise, logits, normalized Q-values, and penalties.

The computed scores are used in the action selection process within the MCTS (Monte Carlo Tree Search) framework. Specifically, this function is called by gumbel_muzero_root_action_selection and gumbel_muzero_policy to determine which actions should be considered for further exploration or exploitation during the search. The scores help prioritize actions that are both promising based on their Q-values and logits while also introducing randomness through Gumbel noise.

Note: It is crucial that all input arrays (gumbel, logits, normalized_qvalues, penalty) have the same shape to avoid runtime errors. Additionally, the function assumes that the inputs are properly preprocessed and aligned with the MCTS tree structure.

Output Example: A possible appearance of the code's return value could be a JAX array of scores for each action, such as [0.5, -inf, 1.2, -inf], where actions with negative infinity scores are not considered due to insufficient visit counts.
## FunctionDef get_sequence_of_considered_visits(max_num_considered_actions, num_simulations)
**get_sequence_of_considered_visits**: The function of get_sequence_of_considered_visits is to return a sequence of visit counts considered by Sequential Halving.

parameters: 
· max_num_considered_actions: The maximum number of considered actions. This value can be smaller than the total number of actions.
· num_simulations: The total simulation budget.

Code Description: The function implements the logic for generating a sequence of visit counts according to the Sequential Halving algorithm, which is used in bandit problems for pure exploration. It starts by checking if the maximum number of considered actions is less than or equal to 1. If true, it returns a simple range from 0 to num_simulations - 1. Otherwise, it calculates the logarithm base 2 of max_num_considered_actions and initializes a list of visit counts for each action. The function then enters a loop where it determines how many extra visits are needed based on the remaining simulations and the number of considered actions. It extends the sequence with the current visit counts and increments these counts accordingly. After adding the necessary visits, the number of considered actions is halved until only two actions remain.

The function get_sequence_of_considered_visits is called by another function named get_table_of_considered_visits in the same module to generate a table of sequences for different numbers of considered actions up to max_num_considered_actions. Additionally, it is used in the test method _check_visits within the SeqHalvingTest class in the tests module to verify that the generated sequence matches expected results.

Note: The function assumes that max_num_considered_actions and num_simulations are positive integers. If max_num_considered_actions is 1 or less, the function returns a simple range of visit counts without applying the Sequential Halving algorithm.

Output Example: For max_num_considered_actions = 4 and num_simulations = 20, a possible output could be (0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3). This sequence indicates that each of the four actions is visited in a round-robin fashion for the first 16 simulations, and then the process repeats until all 20 simulations are accounted for.
## FunctionDef get_table_of_considered_visits(max_num_considered_actions, num_simulations)
**get_table_of_considered_visits**: The function of get_table_of_considered_visits is to generate a table of sequences of visit counts for different numbers of considered actions up to a specified maximum.

parameters: 
· max_num_considered_actions: The maximum number of considered actions. This value can be smaller than the total number of actions.
· num_simulations: The total simulation budget.

Code Description: The function get_table_of_considered_visits generates a table of sequences of visit counts by calling the function get_sequence_of_considered_visits for each possible number of considered actions from 0 to max_num_considered_actions. Each call to get_sequence_of_considered_visits returns a sequence of visit counts according to the Sequential Halving algorithm, which is used in bandit problems for pure exploration. The resulting sequences are stored as tuples and returned as a tuple of these sequences, forming a table with dimensions [max_num_considered_actions + 1, num_simulations]. This table can be utilized by other functions such as gumbel_muzero_root_action_selection to determine the order in which actions should be visited during simulations.

Note: The function assumes that max_num_considered_actions and num_simulations are positive integers. If max_num_considered_actions is 1 or less, each sequence will simply be a range from 0 to num_simulations - 1 without applying the Sequential Halving algorithm.

Output Example: For max_num_considered_actions = 4 and num_simulations = 20, a possible output could be ((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19), (0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3), (0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1), (0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)). This table indicates the visit counts for different numbers of considered actions up to 4, where each row corresponds to a different number of considered actions and each column represents a simulation step.
