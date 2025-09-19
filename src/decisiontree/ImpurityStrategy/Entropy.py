from pandas import DataFrame
import numpy as np
from .Strategy import ImpurityStrategy

class Entropy(ImpurityStrategy):
    def _get_impurity_measure(self,df: DataFrame, target: str):
        proportions = df[target].value_counts() / len(df[target])
        # Handle the case where proportion is 0 to avoid log(0)
        # If all values are the same (pure), entropy should be 0
        if len(proportions) == 1:
            return 0.0
        # For cases where there are zeros, replace with very small number
        proportions = proportions.replace(0, 1e-10)
        return (-proportions * np.log2(proportions)).sum()
    
    def get_detailed_calculations(self, df: DataFrame, feature: str, target: str):
        """Get detailed step-by-step calculations for a feature split"""
        total_entropy = self._get_impurity_measure(df, target)
        calculations = {
            'feature': feature,
            'total_entropy': total_entropy,
            'total_samples': len(df),
            'splits': [],
            'weighted_entropy': 0,
            'information_gain': 0
        }
        
        weighted_entropy = 0
        for value in sorted(df[feature].unique()):
            subset = df[df[feature] == value]
            proportion = len(subset) / len(df)
            subset_entropy = self._get_impurity_measure(subset, target)
            weighted_contribution = proportion * subset_entropy
            weighted_entropy += weighted_contribution
            
            # Get class distribution for this subset
            class_dist = dict(subset[target].value_counts())
            
            split_info = {
                'value': value,
                'samples': len(subset),
                'proportion': proportion,
                'entropy': subset_entropy,
                'weighted_contribution': weighted_contribution,
                'class_distribution': class_dist
            }
            calculations['splits'].append(split_info)
        
        calculations['weighted_entropy'] = weighted_entropy
        calculations['information_gain'] = total_entropy - weighted_entropy
        return calculations
    
    def _get_splitting_criterion(self,df: DataFrame, curr_feature: str, target: str):
        total_entropy = self._get_impurity_measure(df,target)
        weighted_entropy = 0
        for value in df[curr_feature].unique():
            subset = df[df[curr_feature] == value]
            proportion = len(subset) / len(df)
            weighted_entropy += (proportion * self._get_impurity_measure(subset, target))
        info_gain = total_entropy - weighted_entropy
        return info_gain
    
    def get_best_feature(self, df: DataFrame, target: str):
        features = [f for f in df.columns if f != target]
        info_gains = {}
        for feature in features:
            info_gains[feature] = self._get_splitting_criterion(df,feature,target)
        max_info = max(info_gains, key = info_gains.get) # pyright: ignore[reportArgumentType, reportCallIssue]
        return max_info, info_gains[max_info]