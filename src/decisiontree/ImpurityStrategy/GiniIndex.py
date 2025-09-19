from pandas import DataFrame
from .Strategy import ImpurityStrategy
import numpy as np
class GiniIndex(ImpurityStrategy):
    def _get_impurity_measure(self, df: DataFrame, target: str):
        proportions = df[target].value_counts() / len(df)
        return 1 - (np.power(proportions,2).sum())
    
    def get_detailed_calculations(self, df: DataFrame, feature: str, target: str):
        """Get detailed step-by-step calculations for a feature split"""
        total_gini = self._get_impurity_measure(df, target)
        calculations = {
            'feature': feature,
            'total_gini': total_gini,
            'total_samples': len(df),
            'splits': [],
            'weighted_gini': 0,
            'gini_gain': 0
        }
        
        weighted_gini = 0
        for value in sorted(df[feature].unique()):
            subset = df[df[feature] == value]
            proportion = len(subset) / len(df)
            subset_gini = self._get_impurity_measure(subset, target)
            weighted_contribution = proportion * subset_gini
            weighted_gini += weighted_contribution
            
            # Get class distribution for this subset
            class_dist = dict(subset[target].value_counts())
            
            split_info = {
                'value': value,
                'samples': len(subset),
                'proportion': proportion,
                'gini': subset_gini,
                'weighted_contribution': weighted_contribution,
                'class_distribution': class_dist
            }
            calculations['splits'].append(split_info)
        
        calculations['weighted_gini'] = weighted_gini
        calculations['gini_gain'] = total_gini - weighted_gini
        return calculations
    
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