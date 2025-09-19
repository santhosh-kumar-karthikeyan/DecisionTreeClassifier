# Decision Tree CLI Usage Examples

This document shows how to use the Decision Tree CLI for various tasks.

## Installation

First, install the package:

```bash
poetry install
```

## CLI Commands

### 1. Get Dataset Information

Analyze a CSV file to understand its structure:

```bash
poetry run decisiontree info -f your_data.csv
```

Example:

```bash
poetry run decisiontree info -f example_data.csv
```

### 2. Train a Decision Tree Model

Train a model and save it for later use:

```bash
poetry run decisiontree train -f training_data.csv -t target_column -c criterion -o model.json
```

Options:

- `-f, --file`: Path to training CSV file (required)
- `-t, --target`: Name of target column to predict (required)
- `-c, --criterion`: Impurity criterion - 'entropy' or 'gini' (default: gini)
- `-o, --output`: Output file to save trained model (optional)
- `--info/--no-info`: Show dataset information (default: true)
- `--verbose/--quiet`: Verbose output (default: true)

Examples:

```bash
# Train with Gini criterion
poetry run decisiontree train -f iris.csv -t species -c gini -o iris_model.json

# Train with Entropy criterion
poetry run decisiontree train -f data.csv -t class -c entropy -o model.json

# Train without saving model
poetry run decisiontree train -f data.csv -t target
```

### 3. Make Predictions

Use a trained model to make predictions on new data:

```bash
poetry run decisiontree predict -m model.json -f test_data.csv -o predictions.csv
```

Options:

- `-m, --model`: Path to trained model file (required)
- `-f, --test-file`: CSV file with test data (required for batch predictions)
- `-o, --output`: Output CSV file for predictions (optional)
- `--interactive/--batch`: Interactive mode for single predictions (default: batch)

Examples:

```bash
# Batch predictions with output file
poetry run decisiontree predict -m model.json -f test.csv -o results.csv

# Batch predictions without saving
poetry run decisiontree predict -m model.json -f test.csv

# Interactive predictions (planned feature)
poetry run decisiontree predict -m model.json --interactive
```

### 4. Interactive Mode

Launch the full interactive decision tree builder:

```bash
poetry run decisiontree interactive
```

This opens the rich interactive interface where you can:

- Load CSV files with validation
- Select target columns with distribution preview
- Choose impurity criteria
- Build and visualize decision trees
- Make interactive predictions

## Example Workflow

Here's a complete example using the provided sample data:

1. **Examine the dataset:**

```bash
poetry run decisiontree info -f example_data.csv
```

2. **Train a model:**

```bash
poetry run decisiontree train -f example_data.csv -t species -c entropy -o iris_model.json
```

3. **Make predictions:**

```bash
poetry run decisiontree predict -m iris_model.json -f test_data.csv -o predictions.csv
```

4. **Check results:**

```bash
cat predictions.csv
```

## File Formats

### Training Data CSV

- Must have headers
- Can contain both numerical and categorical features
- Target column should be categorical
- Example:

```csv
feature1,feature2,feature3,target
1.2,3.4,cat,class_a
2.1,4.5,dog,class_b
```

### Test Data CSV

- Must have same features as training data (excluding target)
- Same column names and order
- Example:

```csv
feature1,feature2,feature3
1.5,3.2,cat
2.0,4.1,dog
```

### Model File (JSON)

The trained model is saved as JSON containing:

- Tree structure
- Target column name
- Criterion used

### Predictions Output CSV

Contains original test data plus a 'prediction' column:

```csv
feature1,feature2,feature3,prediction
1.5,3.2,cat,class_a
2.0,4.1,dog,class_b
```

## Tips

1. **Data Preparation**: Ensure your CSV files are clean with no missing values in categorical features
2. **Feature Selection**: The algorithm will use all columns except the target for decision making
3. **Model Validation**: Always test your model on separate test data
4. **Criterion Choice**:
   - Use 'gini' for balanced datasets
   - Use 'entropy' for better interpretability
5. **Interactive Mode**: Great for exploration and understanding your data

## Error Handling

The CLI provides helpful error messages for common issues:

- File not found
- Invalid CSV format
- Missing target column
- Prediction failures for unseen values

For prediction failures on unseen values, the model will return the most common class from the training data.
