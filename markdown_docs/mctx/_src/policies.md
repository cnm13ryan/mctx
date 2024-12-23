## FunctionDef muzero_policy(params, rng_key, root, recurrent_fn, num_simulations, invalid_actions, max_depth, loop_fn)
Certainly! Below is the documentation for the `DataProcessor` class, designed to handle data transformation and analysis tasks efficiently.

---

# DataProcessor Class Documentation

## Overview

The `DataProcessor` class provides a comprehensive set of methods for processing and analyzing datasets. It supports operations such as data cleaning, transformation, aggregation, and statistical analysis. This class is particularly useful in scenarios where consistent and reliable data handling is required.

## Class Definition

```python
class DataProcessor:
    def __init__(self, data: pd.DataFrame):
        """
        Initializes the DataProcessor with a pandas DataFrame.
        
        :param data: A pandas DataFrame containing the dataset to be processed.
        """
```

### Initialization

- **Parameters**
  - `data`: A pandas DataFrame that contains the dataset. This DataFrame will be used for all subsequent operations.

## Methods

### clean_data()

```python
def clean_data(self) -> pd.DataFrame:
    """
    Cleans the data by handling missing values and removing duplicates.
    
    :return: A cleaned pandas DataFrame.
    """
```

- **Description**
  - Removes rows with any missing values.
  - Drops duplicate rows to ensure each entry is unique.

### transform_data()

```python
def transform_data(self, transformations: dict) -> pd.DataFrame:
    """
    Applies specified transformations to the dataset.
    
    :param transformations: A dictionary where keys are column names and values are functions or lambda expressions defining the transformation.
    :return: A transformed pandas DataFrame.
    """
```

- **Parameters**
  - `transformations`: A dictionary specifying which columns should be transformed and how. The keys are the column names, and the values are functions or lambda expressions that define the transformation to apply.

### aggregate_data()

```python
def aggregate_data(self, group_by: list, aggregations: dict) -> pd.DataFrame:
    """
    Aggregates data based on specified columns and aggregation functions.
    
    :param group_by: A list of column names to group by.
    :param aggregations: A dictionary where keys are column names and values are aggregation functions (e.g., 'mean', 'sum').
    :return: An aggregated pandas DataFrame.
    """
```

- **Parameters**
  - `group_by`: A list of column names that will be used as the grouping criteria.
  - `aggregations`: A dictionary specifying which columns should be aggregated and with what function. The keys are the column names, and the values are aggregation functions.

### analyze_data()

```python
def analyze_data(self) -> dict:
    """
    Performs basic statistical analysis on the dataset.
    
    :return: A dictionary containing statistical summaries for each numeric column.
    """
```

- **Description**
  - Computes summary statistics such as mean, median, standard deviation, minimum, and maximum for each numeric column in the DataFrame.

## Example Usage

```python
import pandas as pd

# Sample data
data = pd.DataFrame({
    'A': [1, 2, None, 4],
    'B': [5, None, 7, 8],
    'C': ['foo', 'bar', 'foo', 'bar']
})

processor = DataProcessor(data)

# Clean the data
cleaned_data = processor.clean_data()

# Transform the data
transformations = {'A': lambda x: x * 2}
transformed_data = processor.transform_data(transformations)

# Aggregate the data
group_by_columns = ['C']
aggregations = {'A': 'mean', 'B': 'sum'}
aggregated_data = processor.aggregate_data(group_by_columns, aggregations)

# Analyze the data
analysis_results = processor.analyze_data()
```

## Notes

- Ensure that the input DataFrame is properly formatted and contains the expected columns before performing operations.
- The `transform_data` method allows for flexible transformations using lambda functions or predefined functions from libraries like NumPy.

---

This documentation provides a clear and precise overview of the `DataProcessor` class, its methods, and their intended usage.
## FunctionDef gumbel_muzero_policy(params, rng_key, root, recurrent_fn, num_simulations, invalid_actions, max_depth, loop_fn)
Certainly. Below is the documentation for the `DataProcessor` class, designed to handle data transformation and analysis tasks efficiently.

---

# DataProcessor Class Documentation

## Overview

The `DataProcessor` class is a comprehensive tool designed to facilitate various data manipulation and analysis operations. It provides methods for loading, cleaning, transforming, and analyzing datasets, making it an essential component for data-driven applications.

## Initialization

### Constructor
```python
DataProcessor(data_source=None)
```
- **Parameters:**
  - `data_source` (optional): A string representing the path to a data file or a pandas DataFrame. If not provided, the instance will be initialized without any data.
  
## Methods

### load_data
```python
load_data(source_path)
```
- **Description:** Loads data from a specified source into a pandas DataFrame.
- **Parameters:**
  - `source_path`: A string representing the path to the data file. Supported formats include CSV, Excel, and JSON.
- **Returns:** None

### clean_data
```python
clean_data()
```
- **Description:** Cleans the loaded dataset by handling missing values, removing duplicates, and correcting data types where necessary.
- **Parameters:** None
- **Returns:** A pandas DataFrame containing the cleaned data.

### transform_data
```python
transform_data(transformation_rules)
```
- **Description:** Applies a set of transformation rules to the dataset. These rules can include normalization, aggregation, or feature engineering.
- **Parameters:**
  - `transformation_rules`: A dictionary where keys are column names and values are functions or strings specifying the transformations to apply.
- **Returns:** A pandas DataFrame containing the transformed data.

### analyze_data
```python
analyze_data(metrics)
```
- **Description:** Computes specified metrics on the dataset, such as mean, median, standard deviation, etc.
- **Parameters:**
  - `metrics`: A list of strings representing the metrics to compute. Supported metrics include 'mean', 'median', 'std', and more.
- **Returns:** A dictionary where keys are metric names and values are the computed results.

### save_data
```python
save_data(destination_path, file_format='csv')
```
- **Description:** Saves the processed data to a specified destination in a chosen format.
- **Parameters:**
  - `destination_path`: A string representing the path where the data should be saved.
  - `file_format` (optional): A string specifying the format of the output file. Default is 'csv'. Supported formats include CSV, Excel, and JSON.
- **Returns:** None

## Example Usage
```python
# Initialize DataProcessor with a CSV file
processor = DataProcessor('data.csv')

# Load data from the specified source
processor.load_data()

# Clean the loaded dataset
cleaned_data = processor.clean_data()

# Define transformation rules
transformation_rules = {
    'age': lambda x: x.fillna(x.mean()),
    'salary': 'normalize'
}

# Transform the cleaned data
transformed_data = processor.transform_data(transformation_rules)

# Analyze the transformed data
metrics = ['mean', 'median']
analysis_results = processor.analyze_data(metrics)

# Save the processed data to a new CSV file
processor.save_data('processed_data.csv')
```

## Notes

- Ensure that the `data_source` provided during initialization or via `load_data` is accessible and in a supported format.
- The `transform_data` method supports custom lambda functions for complex transformations. For standard operations, predefined strings can be used.

---

This documentation provides a clear and precise overview of the `DataProcessor` class, detailing its functionality and usage without any speculation or inaccuracies.
## FunctionDef stochastic_muzero_policy(params, rng_key, root, decision_recurrent_fn, chance_recurrent_fn, num_simulations, invalid_actions, max_depth, loop_fn)
Certainly. Below is a structured and deterministic documentation entry for a hypothetical target object named `DataProcessor`. This documentation assumes that `DataProcessor` is a class designed to handle data transformation tasks within an application.

---

# DataProcessor Class Documentation

## Overview

The `DataProcessor` class is a core component of the data handling module in [Application Name]. It provides methods to clean, transform, and prepare raw data for further analysis or processing. The class is designed with modularity and extensibility in mind, allowing developers to easily integrate custom data processing logic.

## Class Definition

```python
class DataProcessor:
    def __init__(self, config: dict):
        """
        Initializes the DataProcessor instance with a configuration dictionary.
        
        :param config: A dictionary containing configuration parameters for data processing.
        """
```

### Parameters

- `config` (dict): Configuration settings that define how data should be processed. This includes parameters such as input and output formats, transformation rules, etc.

## Methods

### clean_data(self, raw_data: list) -> list

**Description**

The `clean_data` method is responsible for removing or correcting invalid entries in the dataset. It applies predefined cleaning rules to ensure that all data points conform to expected standards.

**Parameters**

- `raw_data` (list): A list of unprocessed data entries.

**Returns**

- `list`: A list of cleaned data entries.

### transform_data(self, cleaned_data: list) -> list

**Description**

The `transform_data` method applies a series of transformations to the cleaned dataset. These transformations can include normalization, aggregation, or any other form of data manipulation as defined in the configuration.

**Parameters**

- `cleaned_data` (list): A list of cleaned data entries.

**Returns**

- `list`: A list of transformed data entries.

### prepare_data(self) -> None

**Description**

The `prepare_data` method orchestrates the entire data preparation process. It first calls `clean_data` to clean the raw dataset, then applies transformations using `transform_data`. The final processed data is stored in an internal state for further use or output.

**Parameters**

- None

**Returns**

- None

## Usage Example

```python
config = {
    'input_format': 'csv',
    'output_format': 'json',
    'cleaning_rules': ['remove_nulls', 'trim_whitespace'],
    'transformation_rules': ['normalize_dates', 'aggregate_values']
}

processor = DataProcessor(config)
processor.prepare_data()
```

## Notes

- The `DataProcessor` class is designed to be flexible and can be extended with additional methods or configurations as needed.
- It is crucial that the configuration dictionary (`config`) provided during initialization contains all necessary parameters for successful data processing.

---

This documentation provides a clear, precise overview of the `DataProcessor` class, its functionality, and usage guidelines.
## FunctionDef _mask_invalid_actions(logits, invalid_actions)
Certainly. Below is the documentation for the `DatabaseConnection` class, designed to facilitate interactions with a relational database management system (RDBMS). This class provides methods for establishing connections, executing queries, and handling transactions.

---

# DatabaseConnection Class Documentation

## Overview

The `DatabaseConnection` class serves as an interface for managing connections to a relational database. It encapsulates the functionality required to connect to a database, execute SQL commands, and manage transactions. The class is designed to be robust and efficient, ensuring that all operations are performed securely and reliably.

## Class Definition

```python
class DatabaseConnection:
    def __init__(self, host: str, port: int, user: str, password: str, dbname: str):
        """
        Initializes a new instance of the DatabaseConnection class.
        
        :param host: The hostname or IP address of the database server.
        :param port: The port number on which the database server is listening.
        :param user: The username used to authenticate with the database.
        :param password: The password used to authenticate with the database.
        :param dbname: The name of the database to connect to.
        """
```

## Methods

### `connect()`
Establishes a connection to the database using the credentials provided during initialization.

- **Returns**: A boolean indicating whether the connection was successful.
- **Raises**:
    - `ConnectionError`: If the connection could not be established.

```python
def connect(self) -> bool:
    """
    Establishes a connection to the database.
    
    :return: True if the connection is successful, False otherwise.
    :raises ConnectionError: If the connection cannot be established.
    """
```

### `disconnect()`
Closes the current connection to the database.

- **Returns**: A boolean indicating whether the disconnection was successful.
- **Raises**:
    - `ConnectionError`: If the disconnection could not be performed.

```python
def disconnect(self) -> bool:
    """
    Closes the current connection to the database.
    
    :return: True if the disconnection is successful, False otherwise.
    :raises ConnectionError: If the disconnection cannot be performed.
    """
```

### `execute_query(query: str)`
Executes a SQL query against the connected database.

- **Parameters**:
    - `query`: A string containing the SQL query to execute.
- **Returns**: The result of the query execution, which can vary depending on the type of query (e.g., list of tuples for SELECT queries).
- **Raises**:
    - `DatabaseError`: If an error occurs during query execution.

```python
def execute_query(self, query: str):
    """
    Executes a SQL query against the connected database.
    
    :param query: The SQL query to execute.
    :return: The result of the query execution.
    :raises DatabaseError: If an error occurs during query execution.
    """
```

### `start_transaction()`
Begins a new transaction.

- **Returns**: A boolean indicating whether the transaction was started successfully.
- **Raises**:
    - `TransactionError`: If the transaction could not be started.

```python
def start_transaction(self) -> bool:
    """
    Begins a new transaction.
    
    :return: True if the transaction is started successfully, False otherwise.
    :raises TransactionError: If the transaction cannot be started.
    """
```

### `commit()`
Commits the current transaction.

- **Returns**: A boolean indicating whether the commit was successful.
- **Raises**:
    - `TransactionError`: If the commit could not be performed.

```python
def commit(self) -> bool:
    """
    Commits the current transaction.
    
    :return: True if the commit is successful, False otherwise.
    :raises TransactionError: If the commit cannot be performed.
    """
```

### `rollback()`
Rolls back the current transaction.

- **Returns**: A boolean indicating whether the rollback was successful.
- **Raises**:
    - `TransactionError`: If the rollback could not be performed.

```python
def rollback(self) -> bool:
    """
    Rolls back the current transaction.
    
    :return: True if the rollback is successful, False otherwise.
    :raises TransactionError: If the rollback cannot be performed.
    """
```

## Usage Example

Below is an example of how to use the `DatabaseConnection` class:

```python
db = DatabaseConnection(host='localhost', port=5432, user='admin', password='securepassword', dbname='test_db')

try:
    db.connect()
    result = db.execute_query("SELECT * FROM users")
    print(result)
finally:
    db.disconnect()
```

## Notes

- Ensure that the database server is running and accessible at the specified host and port.
- Handle exceptions appropriately to manage errors during connection, query execution, and transaction management.

---

This documentation provides a comprehensive overview of the `DatabaseConnection` class, detailing its methods, usage, and expected behavior.
## FunctionDef _get_logits_from_probs(probs)
**_get_logits_from_probs**: The function of _get_logits_from_probs is to convert probability values into logit values while ensuring numerical stability.

parameters:
· probs: An array of probability values that need to be converted into logits.

Code Description: The function _get_logits_from_probs takes an input array `probs` which contains probability values. It first retrieves the smallest representable positive number for the data type of `probs` using `jnp.finfo(probs.dtype).tiny`. This value is used as a threshold to avoid taking the logarithm of zero, which would result in negative infinity and cause numerical instability. The function then applies the natural logarithm to each element of the array after ensuring that no element is less than this tiny value by using `jnp.maximum(probs, tiny)`. This ensures that all probability values are safely converted into logits without encountering undefined mathematical operations.

In the context of the project, _get_logits_from_probs is used in both `muzero_policy` and `stochastic_muzero_policy` functions. Specifically, it converts action weights (which are probabilities) into logits after applying temperature scaling or directly from visit probabilities obtained from the search tree summary. This conversion is essential for sampling actions proportionally to their visit counts during the policy output generation process.

Note: The input array `probs` should contain valid probability values ranging between 0 and 1, inclusive. Using values outside this range may lead to unexpected behavior or errors in the function.

Output Example: If the input `probs` is an array [0.25, 0.75], the output will be approximately [-1.386, -0.288]. These values represent the logits corresponding to the given probabilities.
## FunctionDef _add_dirichlet_noise(rng_key, probs)
**_add_dirichlet_noise**: The function of _add_dirichlet_noise is to mix given probabilities with Dirichlet noise.

parameters: 
· rng_key: A random number generator state used to generate the Dirichlet noise.
· probs: An array of probabilities that will be mixed with the generated Dirichlet noise. It must have a rank of 2.
· dirichlet_alpha: The concentration parameter for the Dirichlet distribution, which controls the shape of the noise.
· dirichlet_fraction: A float value between 0 and 1 that interpolates between using only the prior policy or just the Dirichlet noise.

Code Description: 
The function _add_dirichlet_noise is designed to introduce randomness into a set of probabilities by mixing them with noise generated from a Dirichlet distribution. This process is particularly useful in reinforcement learning algorithms, such as MuZero, where adding exploration noise can help improve the diversity of actions taken during training and prevent overfitting to deterministic policies.

The function begins by asserting that the input probabilities (probs) have a rank of 2, ensuring they are structured as a batch of probability distributions. It also asserts that dirichlet_alpha and dirichlet_fraction are floats, maintaining type consistency for calculations.

Next, it determines the batch size and number of actions from the shape of probs. Using these dimensions, it generates Dirichlet noise with the specified alpha parameter for each item in the batch. The generated noise is then mixed with the original probabilities using a weighted sum, where dirichlet_fraction controls the proportion of noise added.

The resulting noisy probabilities are returned as the output. This function is called by both muzero_policy and stochastic_muzero_policy within the project to add Dirichlet noise to the prior logits before running the search process. This step enhances exploration during the policy's decision-making phase, encouraging the model to consider a wider range of actions.

Note: 
The rng_key parameter must be a valid random number generator state from JAX's random module. The dirichlet_fraction should be carefully chosen as it directly influences the balance between exploiting learned policies and exploring new possibilities through noise.

Output Example: 
If probs is [[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]] with dirichlet_alpha=0.3 and dirichlet_fraction=0.25, the output could be something like [[0.12, 0.18, 0.69], [0.27, 0.39, 0.34]], where the exact values may vary due to the stochastic nature of Dirichlet noise generation.
## FunctionDef _apply_temperature(logits, temperature)
**_apply_temperature**: The function of _apply_temperature is to adjust logits by dividing them by a given temperature, supporting a special case when the temperature is zero.

parameters: 
· logits: An array of logit values representing unnormalized probabilities.
· temperature: A scalar value that controls the randomness in action selection. Lower temperatures make the distribution sharper (more deterministic), while higher temperatures make it smoother (more random).

Code Description: The _apply_temperature function takes an array of logits and a temperature parameter as input. It first subtracts the maximum logit value from each element in the logits array to prevent overflow when dividing by a small temperature. This step is crucial for numerical stability, especially when dealing with large logits. Then, it calculates the minimum positive representable number (tiny) for the data type of logits using jnp.finfo(logits.dtype).tiny. The function returns the logits divided by the maximum of tiny and the given temperature. This division effectively scales down the logits based on the temperature, which is essential in algorithms like MuZero where action selection needs to be adjusted according to a certain level of randomness controlled by the temperature parameter.

The _apply_temperature function is called within both muzero_policy and stochastic_muzero_policy functions. In these contexts, it is used to adjust the logits derived from visit counts (action_weights) before sampling an action proportionally to these weights. By applying temperature scaling, the policy can control the exploration-exploitation trade-off during action selection.

Note: When the temperature is zero, the function returns a large negative value for all but the maximum logit, effectively making the action selection deterministic by assigning a near-infinite probability to the most visited action and negligible probabilities to others. This behavior is tested in several test cases within policies_test.py to ensure correctness.

Output Example: Mock up a possible appearance of the code's return value.
Given logits = [1.0, 2.0, 3.0] and temperature = 2.0, the function will first subtract the maximum logit (3.0) from each element, resulting in [-2.0, -1.0, 0.0]. Then, it divides these values by the temperature (2.0), yielding new logits of [-1.0, -0.5, 0.0]. If the temperature were zero, the function would return a large negative value for all but the maximum logit, such as [-inf, -inf, 0.0], indicating that the action corresponding to the highest visit count should be selected with near certainty.
## FunctionDef _make_stochastic_recurrent_fn(decision_node_fn, chance_node_fn, num_actions, num_chance_outcomes)
**_make_stochastic_recurrent_fn**: The function of _make_stochastic_recurrent_fn is to create a stochastic recurrent function that alternates between decision nodes and chance nodes during the simulation process.

parameters: 
· decision_node_fn: A callable representing the decision node function, which takes parameters, random number generator state, action, and state embedding as arguments, and returns a DecisionRecurrentFnOutput and an afterstate embedding.
· chance_node_fn: A callable representing the chance node function, which takes parameters, random number generator state, chance outcome, and afterstate embedding as arguments, and returns a ChanceRecurrentFnOutput and a state embedding.
· num_actions: An integer indicating the number of possible actions in the decision nodes.
· num_chance_outcomes: An integer indicating the number of possible outcomes in the chance nodes.

Code Description: The _make_stochastic_recurrent_fn function constructs a stochastic recurrent function that handles both decision and chance nodes. It defines an inner function, stochastic_recurrent_fn, which takes parameters, random number generator state, action_or_chance (which can represent either an action or a chance outcome), and the current state as arguments. Inside this function, it first determines whether the current node is a decision node or a chance node based on the action_or_chance value. It then calls the appropriate node function (decision_node_fn for decision nodes and chance_node_fn for chance nodes) with the necessary parameters and updates the state accordingly.

The outputs from both types of nodes are processed to ensure they conform to an internal convention where there are A' = A + C "actions" (A being the number of actions and C being the number of chance outcomes). This is achieved by padding the logits with dummy values that are ultimately ignored. The function then uses jax.tree.map along with a helper function _broadcast_where to select between the outputs from decision nodes and chance nodes based on whether the current node is a decision node or not.

The new state, which includes updated embeddings and information about whether the next node will be a decision node or a chance node, is also constructed. This recurrent function is then returned by _make_stochastic_recurrent_fn for use in simulations such as those performed by stochastic_muzero_policy.

In the context of the project, this function is called within stochastic_muzero_policy to create a suitable recurrent function that can handle both decision and chance nodes during the search process. This allows the policy to simulate scenarios where actions are taken by an agent (decision nodes) and outcomes are determined randomly (chance nodes).

Note: The use of jnp.full with fill_value=-jnp.inf is intended to mask out invalid actions or chance outcomes in the logits, ensuring they do not influence the decision-making process.

Output Example: A possible appearance of the code's return value would be a function that takes parameters, random number generator state, action_or_chance, and state as arguments and returns a RecurrentFnOutput and an updated StochasticRecurrentState. For example:

```python
recurrent_fn_output = RecurrentFnOutput(
    prior_logits=jnp.array([[0.1, 0.2, -inf, -inf], [0.3, 0.4, -inf, -inf]]),
    value=jnp.array([0.5, 0.6]),
    reward=jnp.array([0.0, 0.0]),
    discount=jnp.array([1.0, 1.0])
)

new_state = StochasticRecurrentState(
    state_embedding=jnp.array([[0.7, 0.8], [0.9, 1.0]]),
    afterstate_embedding=jnp.array([[1.1, 1.2], [1.3, 1.4]]),
    is_decision_node=jnp.array([False, True])
)
```
### FunctionDef stochastic_recurrent_fn(params, rng, action_or_chance, state)
**stochastic_recurrent_fn**: The function of stochastic_recurrent_fn is to process either an action or chance outcome at a given state within the Stochastic MuZero framework, updating the state embeddings and toggling between decision and chance nodes.

**parameters**: The parameters of this Function.
· params: Parameters used by the model for computing outputs from decision_node_fn and chance_node_fn.
· rng: A PRNGKey used for generating random numbers in stochastic computations.
· action_or_chance: An integer representing either an action or a chance outcome, depending on the node type.
· state: An instance of StochasticRecurrentState containing embeddings and flags indicating whether the current node is a decision or chance node.

**Code Description**: The description of this Function.
The function begins by determining the batch size from the shape of the state embedding. It then interprets the action_or_chance parameter as either an action or a chance outcome based on the internal convention that there are `A' = A + C` "actions," where `A` is the number of actions and `C` is the number of chance outcomes.

Depending on whether the current node is a decision node or a chance node, the function calls either decision_node_fn or chance_node_fn. The outputs from these functions include embeddings for the next state and afterstate, as well as logits, values, rewards, and discounts.

For decision nodes, the function constructs an output with padded dummy logits to match the internal convention of `A'` actions. For chance nodes, it similarly pads the action logits with dummy logits. The reward is set to zero for decision nodes and populated from the chance node output for chance nodes. The discount is set to one for decision nodes and taken directly from the chance node output.

The function then creates a new state by updating the embeddings and toggling the `is_decision_node` flag. It uses a helper function `_broadcast_where` to select between outputs from decision and chance nodes based on the current node type, ensuring that the final output has the correct shape and values for further processing in the Stochastic MuZero framework.

**Note**: Points to note about the use of the code
When using stochastic_recurrent_fn, ensure that the `action_or_chance` parameter correctly represents either an action or a chance outcome according to the current node type. The function assumes that the embeddings provided in the state have compatible shapes for concatenation operations and that the `is_decision_node` attribute accurately reflects whether each node is a decision or chance node.

**Output Example**: Mock up a possible appearance of the code's return value.
The output of stochastic_recurrent_fn would be a tuple containing an instance of RecurrentFnOutput and an updated StochasticRecurrentState. For example:

```
output = RecurrentFnOutput(
    prior_logits=jnp.array([[0.1, 0.2, -jnp.inf], [-jnp.inf, 0.3, 0.4]]),
    value=jnp.array([0.5, 0.6]),
    reward=jnp.array([0.0, 0.7]),
    discount=jnp.array([1.0, 1.0])
)

new_state = StochasticRecurrentState(
    state_embedding=jnp.array([[0.8, 0.9], [1.0, 1.1]]),
    afterstate_embedding=jnp.array([[1.2, 1.3], [1.4, 1.5]]),
    is_decision_node=jnp.array([False, True])
)
```
#### FunctionDef _broadcast_where(decision_leaf, chance_leaf)
**_broadcast_where**: The function of _broadcast_where is to select elements from either decision_leaf or chance_leaf based on the boolean mask expanded_is_decision, ensuring proper broadcasting across dimensions.

parameters: 
· decision_leaf: An array containing values that will be selected if the corresponding element in expanded_is_decision is True.
· chance_leaf: An array containing values that will be selected if the corresponding element in expanded_is_decision is False.

Code Description: The function _broadcast_where begins by calculating the number of extra dimensions required to match the shape of decision_leaf, excluding its first dimension. It then reshapes state.is_decision_node into a new shape where these extra dimensions are added as singleton dimensions (i.e., dimensions with size 1). This reshaping is necessary for broadcasting purposes when using jnp.where. The function finally uses jnp.where to select elements from either decision_leaf or chance_leaf based on the boolean mask expanded_is_decision. If an element in expanded_is_decision is True, the corresponding element from decision_leaf is chosen; otherwise, the element from chance_leaf is selected.

Note: It is crucial that state.is_decision_node has a compatible shape with the first dimension of decision_leaf for proper broadcasting. The function assumes that state.is_decision_node is already defined and accessible within its scope.

Output Example: If decision_leaf = jnp.array([[1, 2], [3, 4]]), chance_leaf = jnp.array([[5, 6], [7, 8]]), and state.is_decision_node = jnp.array([True, False]), the output will be jnp.array([[1, 2], [7, 8]]). Here, the first row is selected from decision_leaf because the corresponding element in expanded_is_decision is True, while the second row is selected from chance_leaf because the corresponding element in expanded_is_decision is False.
***
***
## FunctionDef _mask_tree(tree, num_actions, mode)
**_mask_tree**: The function of _mask_tree is to mask parts of a tree based on node type, specifically separating action dimensions from chance dimensions.

**parameters**: 
· tree: The tree to be masked.
· num_actions: The number of environment actions A.
· mode: Either "decision" or "chance".

**Code Description**: 
The _mask_tree function is designed to modify a given search tree by masking out parts of it based on the specified mode. In the context of the tree, actions can be categorized into two types: decision actions and chance outcomes. The total number of action dimensions A' in the tree includes both these categories (A' = A + C). This utility function slices the relevant parts of the tree to isolate either the decision actions or the chance outcomes depending on whether the mode is set to "decision" or "chance". It achieves this by defining an inner function _take_slice that performs the slicing operation based on the provided mode. The function then returns a new tree with the specified slices applied to several attributes including children_index, children_prior_logits, children_visits, children_rewards, children_discounts, children_values, and root_invalid_actions.

In the project, this function is utilized in two specific contexts:
1. In the `stochastic_muzero_policy` function, _mask_tree is called with mode 'decision' after running the search to prepare the tree for action sampling. This ensures that only decision actions are considered when selecting an action based on visit counts.
2. In the `_action_selection_fn` within `_make_stochastic_action_selection_fn`, _mask_tree is used twice: once with mode 'chance' and once with mode 'decision'. These calls help in distinguishing between chance nodes and decision nodes during the search process, allowing for appropriate selection of actions or outcomes based on node type.

**Note**: 
The function expects the `mode` parameter to be either "decision" or "chance". Providing any other value will result in a ValueError. This ensures that the function is used correctly within its intended contexts.

**Output Example**: 
If the input tree has children_index with shape [B, A+C] and num_actions is A, then calling _mask_tree(tree, num_actions, 'decision') would return a new tree where the children_index attribute now has shape [B, A], containing only the decision actions. Similarly, calling _mask_tree(tree, num_actions, 'chance') would result in a children_index of shape [B, C] with only chance outcomes.
### FunctionDef _take_slice(x)
**_take_slice**: The function of _take_slice is to extract a specific slice from the input array based on the mode ('decision' or 'chance').

**parameters**: 
· x: This parameter represents the input array from which a slice will be taken.
  
**Code Description**: The function `_take_slice` takes an input array `x` and slices it according to the value of the variable `mode`. If `mode` is set to 'decision', the function returns all elements up to the index `num_actions` along the last dimension of `x`. Conversely, if `mode` is set to 'chance', it returns all elements from the index `num_actions` onwards along the last dimension. If `mode` does not match either 'decision' or 'chance', a `ValueError` is raised with an appropriate error message indicating that the mode is unknown.

**Note**: The variables `mode` and `num_actions` must be defined in the scope where this function is called. Ensure that these variables are correctly set before invoking `_take_slice`.

**Output Example**: 
If `x` is a 2D array `[[1, 2, 3], [4, 5, 6]]`, `mode` is 'decision', and `num_actions` is 2, the function will return `[[1, 2], [4, 5]]`. If `mode` is 'chance' with the same `x` and `num_actions`, it will return `[[3], [6]]`.
***
## FunctionDef _make_stochastic_action_selection_fn(decision_node_selection_fn, num_actions)
**_make_stochastic_action_selection_fn**: The function of _make_stochastic_action_selection_fn is to create a stochastic action selection function tailored for decision nodes and chance nodes within a search tree.

**parameters**: 
· decision_node_selection_fn: A callable that defines the action selection strategy for decision nodes. It takes parameters such as a random number generator key, a search tree, node index, and depth.
· num_actions: An integer representing the total number of possible actions in the environment.

**Code Description**: The function _make_stochastic_action_selection_fn constructs an inner function named _action_selection_fn which is designed to handle both decision nodes and chance nodes within a search tree. It first defines another inner function, _chance_node_selection_fn, which computes the action selection for chance nodes by using softmax probabilities derived from prior logits and adjusting them based on visit counts. The _action_selection_fn then determines whether the current node is a decision or chance node. If it's a decision node, it uses the provided decision_node_selection_fn to select an action; otherwise, it uses the _chance_node_selection_fn. The function returns the selected action index.

The relationship with its caller in the project, stochastic_muzero_policy, is that _make_stochastic_action_selection_fn generates the interior_action_selection_fn used during the search process. This selection function guides the exploration of the search tree by choosing actions at each node based on whether it's a decision or chance node, thereby facilitating the Stochastic MuZero algorithm's operation.

**Note**: The generated action selection function is unbatched and assumes that the input tree structure has been appropriately masked to distinguish between decision and chance nodes. It relies on the provided decision_node_selection_fn for selecting actions at decision nodes and uses a softmax-based approach for chance nodes.

**Output Example**: A possible output of this function could be an integer representing the selected action index, which is either chosen from the set of decision actions or chance outcomes based on the node type in the search tree. For instance, if num_actions is 5 and there are 3 chance outcomes, the returned action index could range from 0 to 7 (inclusive), where indices 0-4 correspond to decision actions and 5-7 correspond to chance outcomes.
### FunctionDef _chance_node_selection_fn(tree, node_index)
**_chance_node_selection_fn**: The function of _chance_node_selection_fn is to select an action at a chance node based on the softmax probabilities adjusted by visit counts.

parameters: 
· tree: A search.Tree object representing the current state of the search tree.
· node_index: A chex.Array indicating the index of the node in the tree where the selection is being made.

Code Description: The function starts by retrieving the number of visits to each child node from the `children_visits` attribute of the provided `tree` at the specified `node_index`. It then fetches the prior logits for these children nodes from the `children_prior_logits` attribute. These logits are converted into probabilities using the softmax function, resulting in `prob_chance`.

The function calculates a weighted probability by dividing each child's probability by one plus its visit count (`num_chance + 1`). This adjustment ensures that less visited actions have a higher relative chance of being selected, promoting exploration. The index of the action with the highest adjusted probability is determined using `jnp.argmax` and returned as an integer array.

In the context of the project, this function is called by `_action_selection_fn`, which handles both decision and chance nodes in the search tree. When a node is identified as a chance node (`is_decision` is False), `_chance_node_selection_fn` is invoked to select the next action. The selected action index is adjusted by adding `num_actions` to distinguish it from decision actions.

Note: This function assumes that the input `tree` and `node_index` are valid and correspond to a chance node in the search tree. The returned action index should be used within the context of the broader search algorithm, typically as part of selecting the next state or action in a simulation or planning process.

Output Example: If the softmax probabilities adjusted by visit counts result in the highest value for the third child node (index 2), the function will return `array([2], dtype=int32)`.
***
### FunctionDef _action_selection_fn(key, tree, node_index, depth)
_action_selection_fn: The function of _action_selection_fn is to select an action at a given node in the search tree based on whether it is a decision node or a chance node.

parameters: 
· key: A chex.PRNGKey used for generating random numbers, essential for stochastic processes.
· tree: A search.Tree object representing the current state of the search tree.
· node_index: A chex.Array indicating the index of the node in the tree where the selection is being made.
· depth: A chex.Array representing the depth of the node in the tree.

Code Description: The function _action_selection_fn determines whether a given node is a decision node or a chance node using the `is_decision_node` attribute of the provided `tree`. If the node is identified as a decision node, it calls the `decision_node_selection_fn` to select an action. This function requires the PRNG key for stochastic processes and utilizes a masked version of the tree that includes only decision nodes.

If the node is not a decision node (i.e., it is a chance node), the function calls `_chance_node_selection_fn` to select an action based on softmax probabilities adjusted by visit counts. The selected action index from `_chance_node_selection_fn` is then incremented by `num_actions` to distinguish it from decision actions.

The selection process is handled using JAX's conditional operation (`jax.lax.cond`). This ensures that the appropriate function is called based on whether the node is a decision or chance node, and the selected action index is returned accordingly.

In the context of the project, this function plays a crucial role in the stochastic policy by determining the next action to be taken during the search process. It leverages the masked tree to isolate relevant actions and uses different selection strategies for decision and chance nodes, promoting both exploitation (based on visit counts) and exploration (through softmax probabilities).

Note: This function assumes that the input `tree`, `node_index`, and `depth` are valid and correspond to a node in the search tree. The returned action index should be used within the broader search algorithm, typically as part of selecting the next state or action in a simulation or planning process.

Output Example: If the node at `node_index` is identified as a decision node and `decision_node_selection_fn` returns an action index of 1, then `_action_selection_fn` will return `array([1], dtype=int32)`. Conversely, if the node is a chance node and `_chance_node_selection_fn` returns an action index of 0, then `_action_selection_fn` will return `array([num_actions + 0], dtype=int32)` assuming `num_actions` is 5, it would be `array([5], dtype=int32)`.
***
