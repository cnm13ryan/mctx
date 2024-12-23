## FunctionDef _get_version
**_get_version**: The function of _get_version is to extract the version number from the `mctx/__init__.py` file.

parameters: This Function does not take any parameters.
Code Description: The `_get_version` function opens and reads the `mctx/__init__.py` file line by line. It searches for a line that starts with `__version__` and contains an equals sign (`=`). Once such a line is found, it extracts the version number from this line by locating the position of the equals sign, then stripping any whitespace or quotation marks around the version string. If a valid version string is extracted, it is returned immediately. If no such line is found after reading all lines in the file, a `ValueError` is raised with an appropriate error message indicating that the `__version__` variable was not defined in the `mctx/__init__.py` file.
Note: This function assumes that the version number is correctly formatted and located on a single line within the `mctx/__init__.py` file. It also expects the version string to be enclosed in either single or double quotes, which are stripped before returning the result.
Output Example: '1.0.3' (This would be the output if the `__version__` variable is set to '1.0.3' in the `mctx/__init__.py` file.)
## FunctionDef _parse_requirements(path)
**_parse_requirements**: The function of _parse_requirements is to read a requirements file from a specified path and return a list of non-empty, non-comment lines.

parameters:
· path: A string representing the relative path to the requirements file from the current directory.

Code Description: 
The function _parse_requirements takes a single argument, `path`, which specifies the location of the requirements file. It opens this file in read mode and iterates through each line. For each line, it checks if the line is not just whitespace (using `isspace()`) and does not start with a hash (`#`), which indicates a comment. If both conditions are met, the line is stripped of any trailing whitespace using `rstrip()` and added to a list. This list, containing only the relevant package specifications from the requirements file, is then returned.

Note: The function assumes that the path provided is relative to `_CURRENT_DIR`, which should be defined elsewhere in the codebase. It also does not handle potential exceptions such as file not found errors or read errors, so it's expected that these are managed by the caller of this function.

Output Example: 
If the requirements file contains the following lines:
```
# Required packages
numpy==1.21.0
pandas>=1.3.0
# Optional package
matplotlib
```

The output of _parse_requirements would be:
['numpy==1.21.0', 'pandas>=1.3.0', 'matplotlib']
