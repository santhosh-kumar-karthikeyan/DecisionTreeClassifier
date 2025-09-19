#!/usr/bin/env python3
"""
Decision Tree Classifier CLI
============================

Build decision trees from CSV data with detailed metric calculations.
"""

import click
import pandas as pd
from pathlib import Path

from .tree import Tree
from .ImpurityStrategy import GiniIndex, Entropy


def validate_csv_file(ctx, param, value):
    """Validate CSV file exists and is readable"""
    if value is None:
        return value
    
    path = Path(value)
    if not path.exists():
        raise click.BadParameter(f"File not found: {value}")
    
    if not value.lower().endswith('.csv'):
        raise click.BadParameter("File must have .csv extension")
    
    try:
        pd.read_csv(value, nrows=0)
    except Exception as e:
        raise click.BadParameter(f"Error reading CSV file: {e}")
    
    return value


@click.command()
@click.option('--file', '-f', required=True, callback=validate_csv_file,
              help='Path to the CSV file')
@click.option('--target', '-t', required=True,
              help='Name of the target column to predict')
@click.option('--criterion', '-c', type=click.Choice(['gini', 'entropy'], case_sensitive=False),
              default='gini', help='Impurity criterion (default: gini)')
def build_tree(file, target, criterion):
    """Build a decision tree from CSV data showing detailed calculations.
    
    This command loads a CSV dataset, builds a decision tree using the specified
    criterion, and displays intermediate metric calculations and the final tree.
    
    Example:
        decisiontree -f data.csv -t species -c entropy
    """
    
    # Load dataset
    try:
        df = pd.read_csv(file)
        print(f"Loaded dataset: {file}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise click.Abort()
    
    # Validate target column
    if target not in df.columns:
        available_cols = ', '.join(df.columns.tolist())
        print(f"Error: Target column '{target}' not found")
        print(f"Available columns: {available_cols}")
        raise click.Abort()
    
    # Get criterion
    if criterion.lower() == 'entropy':
        criterion_obj = Entropy()
    else:
        criterion_obj = GiniIndex()
    
    # Build tree with verbose output
    tree = Tree(criterion_obj, verbose=True)
    tree.fit(df, target)
    
    # Display the final tree
    tree.display_tree()


if __name__ == '__main__':
    build_tree()