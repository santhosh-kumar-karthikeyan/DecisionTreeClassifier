from abc import ABC, abstractmethod
import pandas as pd


class ImpurityStrategy(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def _get_impurity_measure(self,df:pd.DataFrame, target: str):
        raise NotImplementedError
    
    @abstractmethod
    def _get_splitting_criterion(self, df: pd.DataFrame, curr_feature: str, target: str):
        raise NotImplementedError
    
    @abstractmethod
    def get_best_feature(self, df: pd.DataFrame, target: str):
        raise NotImplementedError