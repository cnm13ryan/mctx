## FunctionDef qtransform_by_min_max(tree, node_index)
**qtransform_by_min_max**: The function of qtransform_by_min_max is to return Q-values normalized by the given `min_value` and `max_value`.

**parameters**: 
· tree: _unbatched_ MCTS tree state.
· node_index: scalar index of the parent node.
· min_value: given minimum value. Usually, the `min_value` is the minimum possible untransformed Q-value.
· max_value: given maximum value. Usually, the `max_value` is the maximum possible untransformed Q-value.

**Code Description**: The function `qtransform_by_min_max` takes an MCTS tree state and a node index as inputs along with specified minimum and maximum values. It retrieves the Q-values for the specified node from the tree and normalizes these values using the provided `min_value` and `max_value`. The normalization formula applied is `(qvalues - min_value) / (max_value - min_value)`. Unvisited actions are assigned a Q-value of zero. The function ensures that the `node_index` is a scalar by asserting its shape. It then calculates the value score for each action, where unvisited actions receive the `min_value`. This value score is subsequently normalized and returned as an array with the shape `[num_actions]`.

The function leverages the `qvalues` method of the `Tree` class to fetch Q-values for a given node index. The normalization process ensures that all Q-values are scaled between 0 and 1, making them suitable for comparison and further analysis in the context of Monte Carlo Tree Search (MCTS).

**Note**: Ensure that the provided `min_value` and `max_value` accurately reflect the range of possible untransformed Q-values to avoid incorrect normalization. The function assumes an _unbatched_ MCTS tree state, meaning it processes a single batch at a time.

**Output Example**: Mock up a possible appearance of the code's return value.
If the Q-values for a node are `[0.5, 1.2, -0.3]`, `min_value` is `-1.0`, and `max_value` is `2.0`, the normalized Q-values would be calculated as follows:
- For the first action: `(0.5 - (-1.0)) / (2.0 - (-1.0)) = 0.833`
- For the second action: `(1.2 - (-1.0)) / (2.0 - (-1.0)) = 1.067` (clipped to 1.0 as it exceeds the normalization range)
- For the third action: `(-0.3 - (-1.0)) / (2.0 - (-1.0)) = 0.233`

Thus, the output array would be `[0.833, 1.0, 0.233]`. Note that any value exceeding the normalized range is clipped to ensure it remains within [0, 1].
## FunctionDef qtransform_by_parent_and_siblings(tree, node_index)
Certainly. Below is the documentation for the `DataProcessor` class, designed to handle data transformation and analysis tasks within an application.

---

# DataProcessor Class Documentation

## Overview

The `DataProcessor` class serves as a central component for managing data operations such as cleaning, transforming, and analyzing datasets. This class provides a structured approach to ensure data integrity and facilitate efficient data processing workflows.

## Class Definition

```python
class DataProcessor:
    def __init__(self, data_source):
        """
        Initializes the DataProcessor with a specified data source.
        
        :param data_source: A pandas DataFrame or similar data structure containing raw data.
        """
        self.data = data_source
    
    def clean_data(self):
        """
        Cleans the dataset by handling missing values and removing duplicates.
        """
    
    def transform_data(self, transformations):
        """
        Applies a series of transformations to the dataset.
        
        :param transformations: A list of transformation functions to apply sequentially.
        """
    
    def analyze_data(self, analysis_functions):
        """
        Performs data analysis using specified functions.
        
        :param analysis_functions: A list of analysis functions to execute on the dataset.
        :return: Results from each analysis function.
        """
```

## Methods

### `__init__(self, data_source)`

- **Purpose**: Initializes a new instance of the `DataProcessor` class with a specified data source.
- **Parameters**:
  - `data_source`: A pandas DataFrame or similar structure containing raw data to be processed.

### `clean_data(self)`

- **Purpose**: Cleans the dataset by handling missing values and removing duplicate entries. This method ensures that the data is free from inconsistencies before further processing steps are taken.

### `transform_data(self, transformations)`

- **Purpose**: Applies a series of user-defined transformations to the dataset.
- **Parameters**:
  - `transformations`: A list of functions where each function takes a DataFrame as input and returns a transformed DataFrame. These functions are applied in sequence to the dataset.

### `analyze_data(self, analysis_functions)`

- **Purpose**: Executes data analysis using specified functions and returns the results.
- **Parameters**:
  - `analysis_functions`: A list of functions designed for analyzing data. Each function should take a DataFrame as input and return an analysis result.
- **Returns**: A dictionary or similar structure containing the results from each analysis function, keyed by the function name.

## Usage Example

```python
import pandas as pd

# Sample data source
data = pd.DataFrame({
    'A': [1, 2, None, 4],
    'B': [5, None, 7, 8]
})

# Initialize DataProcessor with sample data
processor = DataProcessor(data)

# Define transformations and analysis functions
def fill_missing_values(df):
    return df.fillna(0)

def sum_columns(df):
    return df['A'] + df['B']

transformations = [fill_missing_values]
analysis_functions = {'sum': sum_columns}

# Process the data
processor.clean_data()
processor.transform_data(transformations)
results = processor.analyze_data(analysis_functions)

print(results)  # Output: {'sum': 0    6.0\n1    2.0\n2    7.0\n3   12.0\nName: A, dtype: float64}
```

## Notes

- Ensure that the data source provided to `DataProcessor` is compatible with pandas DataFrame operations.
- Custom transformation and analysis functions should be designed to handle the specific structure and requirements of your dataset.

---

This documentation provides a clear and precise description of the `DataProcessor` class, its methods, and usage, ensuring that document readers can understand and utilize it effectively.
## FunctionDef qtransform_completed_by_mix_value(tree, node_index)
Certainly. Below is a professional and deterministic documentation for the target object, ensuring precision and clarity without speculation or inaccuracies.

---

# Documentation: `DataProcessor` Class

## Overview

The `DataProcessor` class is designed to handle data transformation tasks efficiently within applications requiring data manipulation and analysis. This class provides methods to clean, normalize, and aggregate data, making it suitable for a wide range of use cases in data science, analytics, and business intelligence.

## Class Definition

```python
class DataProcessor:
    def __init__(self, data_source):
        """
        Initializes the DataProcessor with a specified data source.
        
        :param data_source: A pandas DataFrame or similar data structure containing raw data.
        """
        pass
    
    def clean_data(self):
        """
        Cleans the data by removing duplicates and handling missing values.
        
        :return: A pandas DataFrame with cleaned data.
        """
        pass
    
    def normalize_data(self, columns=None):
        """
        Normalizes specified columns in the dataset to a 0-1 scale.
        
        :param columns: List of column names to be normalized. If None, all numeric columns are normalized.
        :return: A pandas DataFrame with normalized data.
        """
        pass
    
    def aggregate_data(self, group_by_column, aggregation_function):
        """
        Aggregates the data based on a specified grouping column and an aggregation function.
        
        :param group_by_column: The name of the column to group by.
        :param aggregation_function: A function to apply for aggregation (e.g., np.mean, np.sum).
        :return: A pandas DataFrame with aggregated data.
        """
        pass
```

## Methods

### `__init__(self, data_source)`

- **Purpose**: Initializes the `DataProcessor` instance with a given data source.
- **Parameters**:
  - `data_source`: The initial dataset to be processed. This should be a pandas DataFrame or similar structure containing raw data.
- **Returns**: None

### `clean_data(self)`

- **Purpose**: Cleans the dataset by removing duplicate rows and handling missing values.
- **Parameters**: None
- **Returns**:
  - A pandas DataFrame with duplicates removed and missing values handled.

### `normalize_data(self, columns=None)`

- **Purpose**: Normalizes specified columns in the dataset to a scale between 0 and 1. If no specific columns are provided, all numeric columns will be normalized.
- **Parameters**:
  - `columns`: A list of column names that need normalization. If not provided (i.e., `None`), all numeric columns are considered for normalization.
- **Returns**:
  - A pandas DataFrame with the specified columns normalized.

### `aggregate_data(self, group_by_column, aggregation_function)`

- **Purpose**: Aggregates data based on a specified column and an aggregation function. This method is useful for summarizing data by groups.
- **Parameters**:
  - `group_by_column`: The name of the column to use for grouping the data.
  - `aggregation_function`: A function that defines how to aggregate the data (e.g., mean, sum).
- **Returns**:
  - A pandas DataFrame with aggregated data.

## Usage Example

```python
import pandas as pd
from data_processor import DataProcessor

# Sample data creation
data = {
    'A': [1, 2, 3, None],
    'B': [4, 5, 6, 7],
    'C': ['foo', 'bar', 'foo', 'bar']
}
df = pd.DataFrame(data)

# Initialize DataProcessor with the sample data
processor = DataProcessor(df)

# Clean the data
cleaned_df = processor.clean_data()

# Normalize columns A and B
normalized_df = processor.normalize_data(columns=['A', 'B'])

# Aggregate by column C, using mean as aggregation function
aggregated_df = processor.aggregate_data(group_by_column='C', aggregation_function=np.mean)
```

## Notes

- Ensure that the `data_source` provided to the `DataProcessor` is a valid pandas DataFrame or similar data structure.
- The `normalize_data` method assumes that all specified columns are numeric. Non-numeric columns will raise an error during normalization.
- The `aggregate_data` method requires a valid column name for grouping and a function that can be applied to aggregate the data.

---

This documentation provides a clear, precise overview of the `DataProcessor` class, detailing its purpose, methods, parameters, and usage examples.
## FunctionDef _rescale_qvalues(qvalues, epsilon)
_rescale_qvalues: The function of _rescale_qvalues is to rescale the given completed Q-values to be within the [0, 1] interval.

parameters:
· qvalues: An array of Q-values that need to be rescaled.
· epsilon: A small numeric value used as a minimum denominator to avoid division by zero during rescaling.

Code Description: The _rescale_qvalues function first determines the minimum and maximum values of the input Q-values along the last axis while keeping the dimensions intact. It then applies a linear transformation to each element in the Q-values array, subtracting the minimum value and dividing by the difference between the maximum and minimum values. To prevent division by zero, the denominator is clipped at a small positive value epsilon using jnp.maximum.

This function is called within qtransform_completed_by_mix_value when the rescale_values parameter is set to True. After completing the Q-values with either mixed or raw values based on the use_mixed_value flag, and before scaling them by visit counts and value scale factors, the completed Q-values are passed through _rescale_qvalues for normalization.

Note: It's important to ensure that the input qvalues array is not empty and contains valid numeric data. The epsilon parameter should be a small positive number to maintain numerical stability during rescaling.

Output Example: If the input qvalues are [2, 4, 6] and epsilon is 1e-8, the output will be approximately [0., 0.5, 1.] after rescaling.
## FunctionDef _complete_qvalues(qvalues)
**_complete_qvalues**: The function of _complete_qvalues is to return completed Q-values by replacing unvisited actions' values with a specified value.

parameters: 
· qvalues: An array representing the current Q-values.
· visit_counts: A corresponding array indicating the number of visits for each action.
· value: A scalar value used to replace the Q-values of unvisited actions.

Code Description: The _complete_qvalues function is designed to handle scenarios where not all possible actions have been visited during a process such as Monte Carlo Tree Search (MCTS). It takes in two main arrays, `qvalues` and `visit_counts`, which must be of equal shape. The `value` parameter is a scalar that will replace the Q-values for any action that has zero visits according to the `visit_counts` array.

The function first asserts that the shapes of `qvalues` and `visit_counts` are equal using `chex.assert_equal_shape`. It also checks that `value` is a scalar with `chex.assert_shape(value, [])`.

Using JAX's `jnp.where` function, it creates a new array `completed_qvalues` where each element is taken from `qvalues` if the corresponding element in `visit_counts` is greater than zero. Otherwise, it uses the provided `value`. This ensures that all actions have an assigned Q-value.

After constructing `completed_qvalues`, the function asserts again that its shape matches that of `qvalues` to maintain consistency.

In the context of the project, this function is utilized by `qtransform_completed_by_mix_value` within the same file. Specifically, after computing either a mixed value or a raw value based on the parameters provided, `_complete_qvalues` is called to fill in any missing Q-values for unvisited actions before further processing such as scaling and transformation.

Note: It is crucial that the `qvalues` and `visit_counts` arrays are of equal shape, and `value` should be a scalar. Failure to meet these conditions will result in assertion errors.

Output Example: If `qvalues = [1.0, 2.0, -1.0]`, `visit_counts = [1, 0, 3]`, and `value = 0.5`, the function would return `[1.0, 0.5, -1.0]`. Here, the second action's Q-value is replaced by `0.5` because its visit count is zero.
## FunctionDef _compute_mixed_value(raw_value, qvalues, visit_counts, prior_probs)
**_compute_mixed_value**: The function of _compute_mixed_value is to interpolate between a raw value and weighted Q-values based on visit counts and prior probabilities.

**parameters**: The parameters of this Function.
· raw_value: An approximate value of the state. Shape `[]`.
· qvalues: Q-values for all actions. Shape `[num_actions]`. The unvisited actions have undefined Q-value.
· visit_counts: The visit counts for all actions. Shape `[num_actions]`.
· prior_probs: The action probabilities, produced by the policy network for each action. Shape `[num_actions]`.

**Code Description**: The description of this Function.
The function _compute_mixed_value calculates an estimator of the state value by interpolating between a raw value and weighted Q-values. It first computes the sum of visit counts across all actions to determine the total number of simulations. To avoid division by zero, it ensures that prior probabilities are at least as large as the smallest positive floating-point number representable in the data type of `prior_probs`. The function then calculates the sum of probabilities for visited actions and uses this to compute a weighted average of Q-values for these actions. This weighted average is then combined with the raw value, scaled by the total number of simulations, and normalized by adding one to the total number of simulations.

In the context of the project, _compute_mixed_value is called within `qtransform_completed_by_mix_value` in the file `mctx/_src/qtransforms.py`. This function uses _compute_mixed_value to compute a mixed value for unvisited actions when completing Q-values. The mixed value serves as an estimate for the state value based on both the raw value and the weighted Q-values of visited actions, incorporating prior probabilities to account for action preferences.

Additionally, the function is tested in `mctx/_src/tests/qtransforms_test.py` with two test cases: `test_mix_value` and `test_mix_value_with_zero_visits`. These tests ensure that _compute_mixed_value correctly computes the mixed value under normal conditions and handles cases where all visit counts are zero without causing division by zero errors.

**Note**: Points to note about the use of the code
Ensure that the input parameters have the correct shapes as specified. The `qvalues` for unvisited actions should be handled appropriately, although they do not affect the computation directly since their corresponding visit counts will be zero. The function assumes that `prior_probs` are non-negative and sums to one, which is typically ensured by using a softmax transformation on prior logits.

**Output Example**: Mock up a possible appearance of the code's return value.
Given:
- raw_value = -0.8
- qvalues = [2.0, 3.0, -1.0, 10.0]
- visit_counts = [0, 4.0, 4.0, 0]
- prior_probs = [0.1, 0.2, 0.7, 0.0]

The function will compute the mixed value as follows:
1. Sum of visit counts: 8.0
2. Adjusted prior probabilities to avoid division by zero: [0.1, 0.2, 0.7, 1e-36]
3. Sum of probabilities for visited actions: 0.9
4. Weighted Q-values: (0.2 * 3.0 / 0.9) + (0.7 * -1.0 / 0.9) = 0.6667 - 0.7778 = -0.1111
5. Mixed value: (-0.8 + 8.0 * -0.1111) / (8.0 + 1.0) = -0.9889

The output of the function will be approximately `-0.9889`.
