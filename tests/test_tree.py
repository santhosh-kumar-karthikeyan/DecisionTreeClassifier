import pytest
import pandas as pd
import numpy as np
from decisiontree.ImpurityStrategy.Strategy import ImpurityStrategy
from decisiontree.ImpurityStrategy.Entropy import Entropy
from decisiontree.ImpurityStrategy.GiniIndex import GiniIndex
from decisiontree.tree import Tree

class DummyImpurityStrategy(ImpurityStrategy):
    def _get_impurity_measure(self, df, target):
        return 0.5
    def _get_splitting_criterion(self, df, curr_feature, target):
        return 0.1
    def get_best_feature(self, df, target):
        # Always return the first feature and info gain
        feature = [col for col in df.columns if col != target][0]
        return feature, 0.1

def test_tree_init():
    criterion = DummyImpurityStrategy()
    tree = Tree(criterion)
    assert tree.criterion == criterion

def test_tree_fit():
    df = pd.DataFrame({
        'outlook': ['sunny', 'sunny', 'overcast', 'rainy'],
        'play': ['no', 'no', 'yes', 'yes']
    })
    criterion = DummyImpurityStrategy()
    tree = Tree(criterion)
    tree.fit(df, 'play')
    assert tree.target == 'play'
    assert tree.df.equals(df)
    assert tree.tree is not None

def test_build_tree_pure():
    df = pd.DataFrame({
        'A': [1, 1, 1],
        'target': [0, 0, 0]
    })
    criterion = DummyImpurityStrategy()
    tree = Tree(criterion)
    result = tree.build_tree(df, 'target')
    assert result == 0

def test_build_tree_no_features():
    df = pd.DataFrame({
        'target': [1, 0, 1, 1]
    })
    criterion = DummyImpurityStrategy()
    tree = Tree(criterion)
    result = tree.build_tree(df, 'target')
    assert result == 1  # mode of [1,0,1,1] is 1

def test_tree_with_entropy():
    # Classic tennis dataset scenario
    df = pd.DataFrame({
        'outlook': ['sunny', 'sunny', 'overcast', 'overcast'],
        'humidity': ['high', 'high', 'high', 'normal'],
        'play': ['no', 'no', 'yes', 'yes']
    })
    entropy = Entropy()
    tree = Tree(entropy)
    tree.fit(df, 'play')
    assert tree.tree is not None
    
def test_tree_with_gini():
    df = pd.DataFrame({
        'outlook': ['sunny', 'sunny', 'overcast', 'overcast'],
        'humidity': ['high', 'high', 'high', 'normal'],
        'play': ['no', 'no', 'yes', 'yes']
    })
    gini = GiniIndex()
    tree = Tree(gini)
    tree.fit(df, 'play')
    assert tree.tree is not None

def test_prediction():
    df = pd.DataFrame({
        'outlook': ['sunny', 'rainy', 'overcast'],
        'play': ['no', 'yes', 'yes']
    })
    entropy = Entropy()
    tree = Tree(entropy)
    tree.fit(df, 'play')
    
    # Test prediction
    test_sample = {'outlook': 'sunny'}
    prediction = tree.predict(test_sample)
    assert prediction in ['yes', 'no']
