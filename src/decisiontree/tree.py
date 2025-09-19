from .ImpurityStrategy.Strategy import ImpurityStrategy
import pandas as pd
import numpy as np

class Tree:
    def __init__(self, criterion : ImpurityStrategy, verbose=False) -> None:
        self.criterion = criterion
        self.verbose = verbose
        self.calculations = []  # Store intermediate calculations
        
    def fit(self, df: pd.DataFrame, target: str):
        self.df = df
        self.target = target
        self.calculations = []
        if self.verbose:
            print(f"\nDataset: {df.shape[0]} samples, {df.shape[1]-1} features")
            print(f"Target column: {target}")
            print(f"Classes: {sorted(df[target].unique())}")
            print(f"Criterion: {type(self.criterion).__name__}")
        self.tree = self.build_tree(df, target, depth=0)
                
    def build_tree(self, df: pd.DataFrame, target: str, depth=0):
        #ID 3 alg
        features = [feat for feat in df.columns if feat != target]
        indent = "  " * depth
        
        #If target is pure return the only unique label
        if len(df[target].unique()) == 1:
            result = df[target].iloc[0]
            if self.verbose:
                print(f"{indent}-> Leaf: {result} (pure node)")
            return result
        
        #If there are no more features but target still is impure
        if(len(features) == 0):
            result = df[target].mode().iloc[0]
            if self.verbose:
                print(f"{indent}-> Leaf: {result} (no more features)")
            return result
        
        # Calculate metrics for all features
        if self.verbose:
            current_impurity = self.criterion._get_impurity_measure(df, target)
            print(f"\n{indent}Node at depth {depth}:")
            print(f"{indent}Samples: {len(df)}")
            print(f"{indent}Current {type(self.criterion).__name__}: {current_impurity:.4f}")
            print(f"{indent}Class distribution: {dict(df[target].value_counts())}")
            
        best_feature, best_gain = self.criterion.get_best_feature(df, target)
        
        if self.verbose:
            print(f"{indent}Evaluating features:")
            # Show detailed calculations for all features
            for feature in features:
                calc = self.criterion.get_detailed_calculations(df, feature, target)
                
                # Check if we have detailed calculations or just basic gain
                if 'splits' in calc:
                    print(f"{indent}  Feature: {feature}")
                    
                    # Handle Entropy case
                    if 'total_entropy' in calc:
                        print(f"{indent}    Total entropy: {calc['total_entropy']:.4f}")
                        print(f"{indent}    Splits:")
                        for split in calc['splits']:
                            print(f"{indent}      {feature} = {split['value']}: {split['samples']} samples")
                            print(f"{indent}        Class dist: {split['class_distribution']}")
                            print(f"{indent}        Entropy: {split['entropy']:.4f}")
                            print(f"{indent}        Weight: {split['proportion']:.4f}")
                            print(f"{indent}        Weighted contribution: {split['weighted_contribution']:.4f}")
                        print(f"{indent}    Weighted entropy: {calc['weighted_entropy']:.4f}")
                        print(f"{indent}    Information gain: {calc['information_gain']:.4f}")
                    
                    # Handle Gini case
                    elif 'total_gini' in calc:
                        print(f"{indent}    Total Gini: {calc['total_gini']:.4f}")
                        print(f"{indent}    Splits:")
                        for split in calc['splits']:
                            print(f"{indent}      {feature} = {split['value']}: {split['samples']} samples")
                            print(f"{indent}        Class dist: {split['class_distribution']}")
                            print(f"{indent}        Gini: {split['gini']:.4f}")
                            print(f"{indent}        Weight: {split['proportion']:.4f}")
                            print(f"{indent}        Weighted contribution: {split['weighted_contribution']:.4f}")
                        print(f"{indent}    Weighted Gini: {calc['weighted_gini']:.4f}")
                        print(f"{indent}    Gini gain: {calc['gini_gain']:.4f}")
                else:
                    # Fallback for basic implementations
                    gain = self.criterion._get_splitting_criterion(df, feature, target)
                    print(f"{indent}  {feature}: gain = {gain:.4f}")
                
                print()  # Add spacing between features
            
            print(f"{indent}Best feature: {best_feature} (gain = {best_gain:.4f})")
        
        tree = {best_feature : {}}
        
        for value in df[best_feature].unique():
            subset = df[df[best_feature] == value]
            if self.verbose:
                print(f"{indent}Branch: {best_feature} = {value} ({len(subset)} samples)")
            tree[best_feature][value] = self.build_tree(subset, target, depth + 1)
        
        return tree

    def predict(self, test):
        return self.__prediction_helper(test, self.tree)
    
    def __prediction_helper(self,sample, tree):
        if not isinstance(tree, dict):
            return tree
        
        feature = list(tree.keys())[0]
        feature_value = sample[feature]

        # Handle case where feature value was not seen during training
        if feature_value not in tree[feature]:
            # Return the most common class among all branches
            leaves = []
            self.__collect_leaves(tree[feature], leaves)
            if leaves:
                # Return most common prediction
                from collections import Counter
                most_common = Counter(leaves).most_common(1)
                return most_common[0][0] if most_common else None
            return None

        return self.__prediction_helper(sample, tree[feature][feature_value])
    
    def display_tree(self, tree=None, indent="", feature_name=""):
        """Display the tree structure in text format"""
        if tree is None:
            tree = self.tree
            print("\nDecision Tree Structure:")
            print("=" * 40)
        
        if not isinstance(tree, dict):
            print(f"{indent}-> {tree}")
            return
        
        for feature, branches in tree.items():
            if feature_name:
                print(f"{indent}{feature_name}")
            for value, subtree in branches.items():
                branch_text = f"{indent}├─ {feature} = {value}"
                if isinstance(subtree, dict):
                    print(f"{branch_text}")
                    self.display_tree(subtree, indent + "│  ", "")
                else:
                    print(f"{branch_text} -> {subtree}")

    def __collect_leaves(self, subtree, leaves):
        """Helper method to collect all leaf nodes (predictions) from a subtree"""
        if isinstance(subtree, dict):
            for value in subtree.values():
                self.__collect_leaves(value, leaves)
        else:
            # This is a leaf node
            leaves.append(subtree)


