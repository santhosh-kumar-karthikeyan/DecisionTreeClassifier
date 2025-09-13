from pandas import DataFrame
from .Strategy import ImpurityStrategy
import numpy as np
class GiniIndex(ImpurityStrategy):
    def _get_impurity_measure(self, df: DataFrame, target: str):
        proportions = df[target].value_counts() / len(df)
        return 1 - (np.power(proportions,2).sum())
    
    def _get_splitting_criterion(self, df: DataFrame, curr_feature: str, target: str):
        weighted_gini = 0
        for value in df[curr_feature].unique():
            subset = df[df[curr_feature] == value]
            proportion = len(subset) / len(df)
            weighted_gini += (proportion * self._get_impurity_measure(subset,target))
        return weighted_gini

    def get_best_feature(self, df: DataFrame, target: str):
        features = [f for f in df.columns if f != target]
        weighted_ginis = {}
        for feature in features:
            weighted_ginis[feature] = self._get_splitting_criterion(df,feature,target)
        min_gini = min(weighted_ginis, key=weighted_ginis.get) # pyright: ignore[reportCallIssue, reportArgumentType]
        return min_gini,weighted_ginis[min_gini]