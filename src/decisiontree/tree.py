from .ImpurityStrategy.Strategy import ImpurityStrategy
import pandas as pd
import numpy as np

class Tree:
    def __init__(self, criterion : ImpurityStrategy) -> None:
        self.criterion = criterion
        
    def fit(self, df: pd.DataFrame, target: str):
        self.df = df
        self.target = target
        self.tree = self.build_tree(df, target)
                
    def build_tree(self, df: pd.DataFrame, target: str):
        #ID 3 alg
        features = [feat for feat in df.columns if feat != target]
        
        #If target is pure return the only unique label
        if len(df[target].unique()) == 1:
            return df[target].iloc[0]
        
        #If there are no more features but target still is impure
        if(len(features) == 0):
            return df[target].mode().iloc[0]
        
        best_feature, _ = self.criterion.get_best_feature(df, target)
        tree = {best_feature : {}}
        
        
        for value in df[best_feature].unique():
            subset = df[df[best_feature] == value]
            tree[best_feature][value] = self.build_tree(subset, target)
        
        return tree

    def predict(self, test):
        return self.__prediction_helper(test, self.tree)
    
    def __prediction_helper(self,sample, tree):
        if not isinstance(tree, dict):
            return tree
        
        feature = list(tree.keys())[0]
        feature_value = sample[feature]

        return self.__prediction_helper(sample, tree[feature][feature_value])


