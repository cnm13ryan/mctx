## FunctionDef switching_action_selection_wrapper(root_action_selection_fn, interior_action_selection_fn)
**switching_action_selection_wrapper**: The function of switching_action_selection_wrapper is to wrap root and interior action selection functions in a conditional statement based on the depth of the node.

parameters: 
· root_action_selection_fn: A callable representing the function used to select an action at the root node.
· interior_action_selection_fn: A callable representing the function used to select an action during simulation for nodes other than the root.

Code Description: The switching_action_selection_wrapper function takes two parameters, each of which is a function responsible for selecting actions in different parts of the search tree. It defines and returns a new function, switching_action_selection_fn, which uses JAX's lax.cond operation to decide whether to call the root_action_selection_fn or the interior_action_selection_fn based on the depth parameter. If the depth is 0 (indicating the root node), it calls the root_action_selection_fn with the rng_key, tree, and node_index arguments. Otherwise, it calls the interior_action_selection_fn with all four arguments (rng_key, tree, node_index, depth). This mechanism allows for different action selection strategies at the root versus during the simulation of child nodes.

Note: The function assumes that the provided action selection functions are compatible with the expected argument signatures and return types. It is designed to be used within a Monte Carlo Tree Search (MCTS) framework where different policies might be appropriate for the root node compared to other nodes in the tree.

Output Example: 
A possible appearance of the code's return value could be a callable function, switching_action_selection_fn, which can be invoked with arguments like rng_key, tree, node_index, and depth. When called, this function will return an action selection result as a chex.Array based on whether it is operating at the root or an interior node. For instance:

```python
# Assuming rng_key, tree, node_index, and depth are properly defined
action = switching_action_selection_fn(rng_key, tree, node_index, depth)
```

In this example, action would be a chex.Array representing the selected action based on the current context (root or interior node).
### FunctionDef switching_action_selection_fn(rng_key, tree, node_index, depth)
**switching_action_selection_fn**: The function of switching_action_selection_fn is to select an action based on the depth of the node in the search tree.

parameters: 
· rng_key: A JAX PRNG key used for random number generation.
· tree: An instance of the Tree class representing the search tree.
· node_index: Indices of nodes within the tree for which actions are being selected.
· depth: The depth of the current node in the tree.

Code Description: The switching_action_selection_fn function determines whether to use root_action_selection_fn or interior_action_selection_fn based on the depth parameter. If the depth is 0, indicating that the current node is a root node, it calls root_action_selection_fn with the rng_key, tree, and node_index as arguments. Otherwise, for non-root nodes (depth > 0), it calls interior_action_selection_fn with all four parameters: rng_key, tree, node_index, and depth. This function leverages JAX's lax.cond to conditionally execute different action selection functions based on the node's position in the search tree.

Note: Ensure that the Tree object passed as a parameter is properly initialized and contains valid data for nodes and actions. The rng_key should be a valid JAX PRNG key for generating random numbers if required by the action selection functions.

Output Example: Depending on whether the depth is 0 or greater, the function will return an action selected by either root_action_selection_fn or interior_action_selection_fn. For instance, if depth is 0 and root_action_selection_fn returns an action index of 2, then switching_action_selection_fn will also return 2. Similarly, for a non-root node with depth > 0, if interior_action_selection_fn returns an action index of 5, switching_action_selection_fn will return 5.
***
## FunctionDef muzero_action_selection(rng_key, tree, node_index, depth)
Certainly. Below is the documentation for the `DataProcessor` class, designed to handle data transformation and analysis tasks within an application.

---

# DataProcessor Class Documentation

## Overview

The `DataProcessor` class is a utility component responsible for performing various operations on datasets, including cleaning, transforming, and analyzing data. It provides methods that facilitate efficient data manipulation and ensure the integrity of the processed information.

## Class Definition

```python
class DataProcessor:
    def __init__(self, dataset):
        """
        Initializes the DataProcessor with a given dataset.
        
        :param dataset: A pandas DataFrame representing the input dataset.
        """
```

### Parameters

- `dataset`: A pandas DataFrame containing the raw data to be processed.

## Methods

### 1. clean_data()

```python
def clean_data(self):
    """
    Cleans the dataset by removing duplicates and handling missing values.
    
    :return: None; modifies the internal dataset in place.
    """
```

#### Description

The `clean_data` method processes the dataset to eliminate duplicate entries and address any missing data points. This ensures that subsequent analyses are based on a complete and accurate dataset.

### 2. transform_data(self, transformations)

```python
def transform_data(self, transformations):
    """
    Applies specified transformations to the dataset.
    
    :param transformations: A dictionary where keys are column names and values are functions to apply.
    :return: None; modifies the internal dataset in place.
    """
```

#### Parameters

- `transformations`: A dictionary mapping column names to transformation functions. Each function should accept a pandas Series (representing a column) as input and return a transformed pandas Series.

#### Description

The `transform_data` method allows for the application of custom transformations to specific columns within the dataset. This is useful for tasks such as normalization, encoding categorical variables, or any other data manipulation required by the analysis pipeline.

### 3. analyze_data(self)

```python
def analyze_data(self):
    """
    Performs basic statistical analysis on the dataset.
    
    :return: A dictionary containing summary statistics for each column in the dataset.
    """
```

#### Description

The `analyze_data` method computes and returns a set of summary statistics for each column in the dataset. These statistics include measures such as mean, median, standard deviation, minimum, maximum, and count of non-null values.

## Usage Example

```python
import pandas as pd

# Sample dataset creation
data = {
    'A': [1, 2, 3, None, 5],
    'B': ['x', 'y', 'z', 'w', 'v']
}
df = pd.DataFrame(data)

# Initialize DataProcessor with the dataset
processor = DataProcessor(df)

# Clean the data
processor.clean_data()

# Define transformations
transformations = {
    'A': lambda x: x.fillna(x.mean()),  # Fill missing values in column A with mean
    'B': lambda x: x.str.upper()         # Convert all strings in column B to uppercase
}

# Apply transformations
processor.transform_data(transformations)

# Analyze the data
summary_stats = processor.analyze_data()
print(summary_stats)
```

## Notes

- The `DataProcessor` class assumes that the input dataset is a pandas DataFrame. Users should ensure compatibility with this format.
- Custom transformation functions provided to `transform_data` must be capable of handling pandas Series objects.

---

This documentation provides a comprehensive overview of the `DataProcessor` class, detailing its functionality and usage within an application context.
## ClassDef GumbelMuZeroExtraData
**GumbelMuZeroExtraData**: The function of GumbelMuZeroExtraData is to store extra data required for the Gumbel MuZero search algorithm.
attributes: The attributes of this Class.
· root_gumbel: This attribute holds an array containing Gumbel noise values, which are used in the root node action selection process during the Gumbel MuZero search.

**Code Description**: The GumbelMuZeroExtraData class is designed to encapsulate additional data necessary for executing the Gumbel MuZero algorithm. Specifically, it stores a chex.Array named `root_gumbel`, which contains Gumbel noise values. These noise values are crucial for introducing stochasticity in the action selection process at the root node of the search tree.

In the context of the project, this class is instantiated within the `gumbel_muzero_policy` function located in `mctx/_src/policies.py`. The `gumbel_muzero_policy` function implements the Gumbel MuZero policy for decision-making in reinforcement learning tasks. During its execution, it generates Gumbel noise and assigns this noise to an instance of `GumbelMuZeroExtraData`. This extra data is then passed to the search process managed by the `search.search` function, where it influences the action selection strategy at the root node.

The use of Gumbel noise in the root node action selection helps in balancing exploration and exploitation during the search. By adding this noise to the prior logits from the policy network, the algorithm can explore a diverse set of actions while still considering their expected values, leading to more robust decision-making.

**Note**: Points to note about the use of the code
Developers should ensure that the `root_gumbel` attribute is correctly initialized with Gumbel noise before passing an instance of `GumbelMuZeroExtraData` to the search process. The shape and data type of the `root_gumbel` array must match those expected by the action selection functions, particularly in terms of batch dimensions and number of actions. Additionally, understanding the role of Gumbel noise in the algorithm is essential for effectively utilizing this class in different reinforcement learning scenarios.
## FunctionDef gumbel_muzero_root_action_selection(rng_key, tree, node_index)
Certainly. Below is the documentation for the `DataProcessor` class, designed to handle data transformation and analysis tasks within an application.

---

# DataProcessor Class Documentation

## Overview

The `DataProcessor` class provides a comprehensive suite of methods for processing and analyzing datasets. It supports operations such as data cleaning, normalization, aggregation, and statistical analysis, making it a versatile tool for data-driven applications.

## Class Definition

```python
class DataProcessor:
    def __init__(self, dataset):
        """
        Initializes the DataProcessor with a given dataset.
        
        :param dataset: A pandas DataFrame representing the dataset to be processed.
        """
```

## Methods

### `clean_data(self)`

**Description:**  
Removes any rows containing missing values from the dataset.

**Parameters:**  
- None

**Returns:**  
- A pandas DataFrame with all rows containing missing values removed.

---

### `normalize_data(self, method='min-max')`

**Description:**  
Normalizes the numerical features of the dataset using a specified method. The default method is 'min-max' scaling.

**Parameters:**
- `method`: A string indicating the normalization method to use. Supported methods are:
  - `'min-max'`: Scales each feature to a given range, usually [0, 1].
  - `'z-score'`: Standardizes features by removing the mean and scaling to unit variance.

**Returns:**  
- A pandas DataFrame with normalized numerical features.

---

### `aggregate_data(self, group_by_column, agg_func='mean')`

**Description:**  
Aggregates the dataset based on a specified column using a given aggregation function.

**Parameters:**
- `group_by_column`: The name of the column to group by.
- `agg_func`: A string indicating the aggregation function to apply. Supported functions are:
  - `'mean'`: Computes the mean of each group.
  - `'sum'`: Sums up the values in each group.
  - `'count'`: Counts the number of occurrences in each group.

**Returns:**  
- A pandas DataFrame with aggregated data.

---

### `calculate_statistics(self)`

**Description:**  
Calculates basic statistical measures (mean, median, standard deviation) for numerical features in the dataset.

**Parameters:**  
- None

**Returns:**  
- A dictionary containing statistical measures for each numerical feature.

---

## Usage Example

```python
import pandas as pd

# Sample dataset creation
data = {
    'Age': [25, 30, 35, 40, None],
    'Salary': [50000, 60000, 70000, 80000, 90000]
}
df = pd.DataFrame(data)

# Initialize DataProcessor
processor = DataProcessor(df)

# Clean data
cleaned_df = processor.clean_data()

# Normalize data using min-max scaling
normalized_df = processor.normalize_data(method='min-max')

# Aggregate data by 'Age' column, calculating the mean salary for each age group
aggregated_df = processor.aggregate_data(group_by_column='Age', agg_func='mean')

# Calculate basic statistics
statistics = processor.calculate_statistics()
```

## Notes

- The `DataProcessor` class assumes that the input dataset is a pandas DataFrame.
- Ensure that the dataset contains appropriate data types for each column to avoid errors during processing.

---

This documentation provides a clear and precise overview of the `DataProcessor` class, its methods, and their functionalities.
## FunctionDef gumbel_muzero_interior_action_selection(rng_key, tree, node_index, depth)
Certainly. Below is the documentation for the `DataProcessor` class, designed to handle data transformation and analysis tasks within an application.

---

# DataProcessor Class Documentation

## Overview

The `DataProcessor` class serves as a central component for managing data operations such as loading, transforming, and analyzing datasets. This class provides a structured approach to ensure data integrity and facilitate efficient processing workflows.

## Class Definition

```python
class DataProcessor:
    def __init__(self, source: str):
        """
        Initializes the DataProcessor with a specified data source.
        
        :param source: A string representing the path or identifier of the data source.
        """

    def load_data(self) -> pd.DataFrame:
        """
        Loads data from the specified source into a pandas DataFrame.
        
        :return: A pandas DataFrame containing the loaded data.
        """

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans and preprocesses the input DataFrame to handle missing values,
        remove duplicates, and standardize formats.
        
        :param df: The pandas DataFrame to be cleaned.
        :return: A cleaned pandas DataFrame.
        """

    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies a series of transformations to the input DataFrame based on
        predefined rules or functions. This may include normalization,
        aggregation, and feature engineering.
        
        :param df: The pandas DataFrame to be transformed.
        :return: A transformed pandas DataFrame.
        """

    def analyze_data(self, df: pd.DataFrame) -> dict:
        """
        Performs statistical analysis on the input DataFrame and returns
        key insights or metrics as a dictionary.
        
        :param df: The pandas DataFrame to be analyzed.
        :return: A dictionary containing analysis results.
        """

    def save_results(self, data: dict, output_path: str) -> None:
        """
        Saves the analysis results to a specified file path in JSON format.
        
        :param data: The dictionary containing analysis results.
        :param output_path: The file path where the results will be saved.
        """
```

## Methods

### `__init__(self, source: str)`

- **Description**: Initializes an instance of the `DataProcessor` class with a specified data source.
- **Parameters**:
  - `source`: A string representing the path or identifier of the data source.

### `load_data(self) -> pd.DataFrame`

- **Description**: Loads data from the specified source into a pandas DataFrame.
- **Returns**: A pandas DataFrame containing the loaded data.

### `clean_data(self, df: pd.DataFrame) -> pd.DataFrame`

- **Description**: Cleans and preprocesses the input DataFrame to handle missing values, remove duplicates, and standardize formats.
- **Parameters**:
  - `df`: The pandas DataFrame to be cleaned.
- **Returns**: A cleaned pandas DataFrame.

### `transform_data(self, df: pd.DataFrame) -> pd.DataFrame`

- **Description**: Applies a series of transformations to the input DataFrame based on predefined rules or functions. This may include normalization, aggregation, and feature engineering.
- **Parameters**:
  - `df`: The pandas DataFrame to be transformed.
- **Returns**: A transformed pandas DataFrame.

### `analyze_data(self, df: pd.DataFrame) -> dict`

- **Description**: Performs statistical analysis on the input DataFrame and returns key insights or metrics as a dictionary.
- **Parameters**:
  - `df`: The pandas DataFrame to be analyzed.
- **Returns**: A dictionary containing analysis results.

### `save_results(self, data: dict, output_path: str) -> None`

- **Description**: Saves the analysis results to a specified file path in JSON format.
- **Parameters**:
  - `data`: The dictionary containing analysis results.
  - `output_path`: The file path where the results will be saved.

## Usage Example

```python
# Initialize DataProcessor with a data source
processor = DataProcessor(source='path/to/data.csv')

# Load and process data
raw_data = processor.load_data()
cleaned_data = processor.clean_data(raw_data)
transformed_data = processor.transform_data(cleaned_data)

# Analyze the data and save results
analysis_results = processor.analyze_data(transformed_data)
processor.save_results(analysis_results, 'path/to/output.json')
```

---

This documentation provides a comprehensive overview of the `DataProcessor` class, detailing its methods and intended usage. For further questions or support, please refer to the application's user manual or contact technical support.
## FunctionDef masked_argmax(to_argmax, invalid_actions)
**masked_argmax**: The function of masked_argmax is to return a valid action index from the `to_argmax` array while considering which actions are invalid as specified by the `invalid_actions` mask.

parameters: 
· to_argmax: An array containing values for each action, where the highest value indicates the preferred action.
· invalid_actions: An optional array of the same shape as `to_argmax`, where a value of True or 1 indicates that the corresponding action is invalid and should not be considered.

Code Description: The function first checks if the `invalid_actions` parameter is provided. If it is, the function asserts that `to_argmax` and `invalid_actions` have the same shape to ensure compatibility. It then uses a where clause from JAX's NumPy module (`jnp.where`) to replace invalid action values in `to_argmax` with negative infinity (-inf). This ensures that these actions will not be selected as they will have the lowest possible value when determining the maximum. The function then computes the index of the maximum value along the last axis using `jnp.argmax`, which corresponds to the best valid action. Finally, it converts this index to an integer type (`int32`) before returning it.

The function is used in several places within the project where actions need to be selected from a set of possible actions while respecting constraints on which actions are valid. For example, in `muzero_action_selection`, `masked_argmax` is called with the computed scores for each action (`to_argmax`) and a mask indicating invalid actions at the root node (`tree.root_invalid_actions * (depth == 0)`). This ensures that only valid actions are considered when selecting an action from the root of the MCTS tree. Similarly, in `gumbel_muzero_root_action_selection`, `masked_argmax` is used to select an action based on scores computed using Sequential Halving with Gumbel, again ensuring that invalid actions are not selected.

Note: The function assumes that if all actions are marked as invalid, the action index returned will be 0. This behavior may need to be handled appropriately by the caller to avoid unintended consequences.

Output Example: If `to_argmax` is [1.5, -2.3, 4.8] and `invalid_actions` is [False, True, False], the function will return 2, as the third action (index 2) has the highest value among valid actions.
## FunctionDef _prepare_argmax_input(probs, visit_counts)
**_prepare_argmax_input**: The function of _prepare_argmax_input is to prepare the input for deterministic action selection by adjusting probabilities based on visit counts.

**parameters**: 
· probs: A policy or an improved policy, represented as a NumPy array with shape `[num_actions]`.
· visit_counts: The existing visit counts for each action, also represented as a NumPy array with shape `[num_actions]`.

**Code Description**: The function _prepare_argmax_input calculates the input to be used in an argmax operation. It adjusts the given probabilities (`probs`) by subtracting the normalized visit counts from them. This adjustment is done to ensure that when `argmax` is applied repeatedly with updated visit counts, the resulting visitation frequencies approximate the original probability distribution (`probs`). The formula used for this adjustment is `probs - visit_counts / (1 + sum(visit_counts))`. The function first asserts that the shapes of `probs` and `visit_counts` are equal to ensure compatibility. It then computes the adjusted values and returns them.

In the context of the project, _prepare_argmax_input is called by `gumbel_muzero_interior_action_selection`, which uses it to select an action based on visit counts in a Monte Carlo Tree Search (MCTS) tree. Specifically, `gumbel_muzero_interior_action_selection` computes improved probabilities by adding prior logits and completed Q-values, applies the softmax function to these values to obtain a policy, and then calls _prepare_argmax_input with this policy and the existing visit counts. The result is used to select an action that balances exploration and exploitation.

**Note**: It is crucial that `probs` and `visit_counts` have the same shape before calling this function to avoid runtime errors due to shape mismatches.

**Output Example**: If `probs = [0.2, 0.3, 0.5]` and `visit_counts = [10, 20, 30]`, the output of _prepare_argmax_input would be an array that reflects the adjusted probabilities based on the visit counts, which could look something like `[0.1, -0.1, -0.3]` after normalization and subtraction. The exact values depend on the sum of `visit_counts`.
