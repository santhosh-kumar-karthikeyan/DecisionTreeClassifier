#!/usr/bin/env python3
"""
Interactive Decision Tree Shell using professional libraries
===========================================================

Uses rich for beautiful terminal UI, questionary for interactive prompts,
and tabulate for data display.
"""

import pandas as pd
import os
from pathlib import Path

# Professional libraries for better UX
from rich.console import Console
from rich.table import Table
from rich.tree import Tree as RichTree
from rich.panel import Panel
from rich.progress import track
from rich.prompt import Prompt, Confirm
from rich import print as rprint
import questionary
from tabulate import tabulate

# Your tree implementation
from .tree import Tree
from .ImpurityStrategy import GiniIndex, Entropy

console = Console()

def print_banner():
    """Print welcome banner using rich"""
    banner = Panel.fit(
        "[bold blue]🌳 Interactive Decision Tree Builder 🌳[/bold blue]\n"
        "Build and test decision trees from CSV data with beautiful visualizations",
        style="blue"
    )
    console.print(banner)

def get_csv_file_path():
    """Get CSV file path using questionary with validation"""
    def validate_csv_path(path):
        if not path:
            return "Please enter a file path"
        
        if not Path(path).exists():
            return f"File not found: {path}"
        
        if not path.lower().endswith('.csv'):
            return "Please provide a CSV file (.csv extension)"
        
        try:
            pd.read_csv(path, nrows=1)
            return True
        except Exception as e:
            return f"Error reading CSV: {e}"
    
    return questionary.path(
        "Enter the path to your CSV file:",
        validate=validate_csv_path,
        only_directories=False
    ).ask()

def display_data_info(df, file_path):
    """Display data information using rich tables"""
    console.print(f"\n✅ [green]Successfully loaded:[/green] {file_path}")
    
    # Basic info panel
    info_panel = Panel(
        f"📊 [bold]Dataset Info[/bold]\n"
        f"Rows: {df.shape[0]:,}\n"
        f"Columns: {df.shape[1]}\n"
        f"Memory usage: {df.memory_usage().sum() / 1024:.1f} KB",
        title="Dataset Overview",
        expand=False
    )
    console.print(info_panel)
    
    # Column info table
    console.print("\n📋 [bold]Column Information[/bold]")
    col_table = Table(show_header=True, header_style="bold magenta")
    col_table.add_column("Column", style="cyan")
    col_table.add_column("Type", style="yellow")
    col_table.add_column("Unique Values", justify="right")
    col_table.add_column("Sample Values", style="dim")
    
    for col in df.columns:
        unique_count = df[col].nunique()
        sample_values = df[col].dropna().unique()[:3]
        sample_str = ", ".join([str(v) for v in sample_values])
        if len(sample_values) == 3 and unique_count > 3:
            sample_str += "..."
            
        col_table.add_row(
            col,
            str(df[col].dtype),
            str(unique_count),
            sample_str
        )
    
    console.print(col_table)
    
    # Data preview
    console.print(f"\n👀 [bold]Data Preview (first 5 rows)[/bold]")
    # Convert to string table for better formatting
    preview_data = df.head().astype(str).to_dict('records')
    preview_table = tabulate(preview_data, headers="keys", tablefmt="grid")
    console.print(f"[dim]{preview_table}[/dim]")

def select_target_column(df):
    """Select target column using questionary"""
    choices = []
    for col in df.columns:
        unique_count = df[col].nunique()
        choices.append({
            'name': f"{col} ({unique_count} unique values)",
            'value': col
        })
    
    target_col = questionary.select(
        "🎯 Select the target column (what you want to predict):",
        choices=choices
    ).ask()
    
    if target_col:
        # Show target distribution
        console.print(f"\n✅ [green]Selected target:[/green] '{target_col}'")
        
        dist_table = Table(title=f"Target Distribution: {target_col}")
        dist_table.add_column("Value", style="cyan")
        dist_table.add_column("Count", justify="right")
        dist_table.add_column("Percentage", justify="right", style="green")
        
        value_counts = df[target_col].value_counts()
        for value, count in value_counts.items():
            percentage = (count / len(df)) * 100
            dist_table.add_row(str(value), str(count), f"{percentage:.1f}%")
        
        console.print(dist_table)
    
    return target_col

def select_criterion():
    """Select impurity criterion using questionary"""
    criterion_choice = questionary.select(
        "🔧 Select the impurity criterion:",
        choices=[
            {'name': 'Entropy (Information Gain)', 'value': 'entropy'},
            {'name': 'Gini Index', 'value': 'gini'}
        ]
    ).ask()
    
    if criterion_choice == 'entropy':
        console.print("✅ [green]Selected:[/green] Entropy")
        return Entropy()
    else:
        console.print("✅ [green]Selected:[/green] Gini Index")
        return GiniIndex()

def build_rich_tree(tree_dict, name="Decision Tree"):
    """Convert decision tree to rich Tree for beautiful display"""
    def add_branches(tree_node, tree_dict):
        if not isinstance(tree_dict, dict):
            # Leaf node
            tree_node.add(f"🍃 [bold green]Predict: {tree_dict}[/bold green]")
            return
        
        for feature, branches in tree_dict.items():
            feature_node = tree_node.add(f"🔍 [bold blue]{feature}[/bold blue]")
            for value, subtree in branches.items():
                value_node = feature_node.add(f"├─ [yellow]if {feature} = '{value}'[/yellow]")
                add_branches(value_node, subtree)
    
    tree = RichTree(f"🌳 [bold magenta]{name}[/bold magenta]")
    add_branches(tree, tree_dict)
    return tree

def get_prediction_input(df, target_col):
    """Get prediction input using questionary"""
    feature_cols = [col for col in df.columns if col != target_col]
    sample = {}
    
    console.print("\n🔮 [bold]Prediction Input[/bold]")
    
    for feature in track(feature_cols, description="Entering feature values..."):
        unique_values = sorted(df[feature].unique())
        
        # Create choices with known values + custom option
        choices = [{'name': str(val), 'value': str(val)} for val in unique_values]
        choices.append({'name': '🖊️  Enter custom value', 'value': '__custom__'})
        
        value = questionary.select(
            f"Select value for '{feature}':",
            choices=choices
        ).ask()
        
        if value == '__custom__':
            value = questionary.text(f"Enter custom value for '{feature}':").ask()
        
        sample[feature] = value
    
    return sample

def prediction_session(tree, df, target_col):
    """Interactive prediction session"""
    console.print(Panel("🔮 Prediction Mode", style="magenta"))
    
    while True:
        try:
            # Get input
            sample = get_prediction_input(df, target_col)
            
            if not sample:  # User cancelled
                break
            
            # Make prediction
            with console.status("[bold green]Making prediction..."):
                prediction = tree.predict(sample)
            
            # Display result
            result_table = Table(title="🎯 Prediction Result", style="green")
            result_table.add_column("Feature", style="cyan")
            result_table.add_column("Value", style="yellow")
            
            for feature, value in sample.items():
                result_table.add_row(feature, str(value))
            
            console.print(result_table)
            console.print(f"\n🔮 [bold green]Predicted {target_col}:[/bold green] [bold]{prediction}[/bold]")
            
        except Exception as e:
            console.print(f"❌ [red]Prediction error:[/red] {e}")
        
        # Ask for another prediction
        if not Confirm.ask("\n❓ Make another prediction?", default=True):
            break

def main():
    """Main interactive shell"""
    print_banner()
    
    try:
        # Step 1: Get CSV file
        file_path = get_csv_file_path()
        if not file_path:
            console.print("👋 [yellow]Goodbye![/yellow]")
            return
        
        # Step 2: Load and display data
        with console.status("[bold blue]Loading data..."):
            df = pd.read_csv(file_path)
        
        display_data_info(df, file_path)
        
        # Step 3: Select target column
        target_col = select_target_column(df)
        if not target_col:
            console.print("👋 [yellow]Goodbye![/yellow]")
            return
        
        # Step 4: Select criterion
        criterion = select_criterion()
        if not criterion:
            console.print("👋 [yellow]Goodbye![/yellow]")
            return
        
        # Step 5: Build tree
        with console.status("[bold blue]Building decision tree..."):
            tree = Tree(criterion)
            tree.fit(df, target_col)
        
        console.print("✅ [green]Decision tree built successfully![/green]")
        
        # Step 6: Display tree
        console.print("\n🌳 [bold]Decision Tree Structure[/bold]")
        rich_tree = build_rich_tree(tree.tree)
        console.print(rich_tree)
        
        # Step 7: Predictions
        if Confirm.ask("\n❓ Make predictions with this tree?", default=True):
            prediction_session(tree, df, target_col)
        
        console.print("\n👋 [bold blue]Thank you for using Decision Tree Builder![/bold blue]")
        
    except KeyboardInterrupt:
        console.print("\n\n👋 [yellow]Goodbye![/yellow]")
    except Exception as e:
        console.print(f"\n❌ [red]Error:[/red] {e}")

if __name__ == "__main__":
    main()
