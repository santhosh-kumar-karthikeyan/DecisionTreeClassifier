import pytest
import pandas as pd
import numpy as np
from decisiontree.ImpurityStrategy.GiniIndex import GiniIndex
from decisiontree.ImpurityStrategy.Strategy import ImpurityStrategy

@pytest.fixture
def gini_instance():
    return GiniIndex()

def test_gini_is_subclass():
    assert issubclass(GiniIndex, ImpurityStrategy)

def test_gini_impurity_measure_pure(gini_instance):
    df = pd.DataFrame({'target': [0, 0, 0, 0]})
    result = gini_instance._get_impurity_measure(df, 'target')
    assert np.isclose(result, 0.0)

def test_gini_impurity_measure_balanced(gini_instance):
    df = pd.DataFrame({'target': [1, 1, 0, 0]})
    result = gini_instance._get_impurity_measure(df, 'target')
    # For balanced binary classification, gini should be 0.5
def test_gini_splitting_criterion(gini_instance):
    df = pd.DataFrame({
        'feature': [1, 1, 0, 0], 
        'target': [1, 1, 0, 0]
    })
    weighted_gini = gini_instance._get_splitting_criterion(df, 'feature', 'target')
    # Perfect split should give 0 weighted gini
    assert np.isclose(weighted_gini, 0.0)

def test_gini_get_best_feature(gini_instance):
    df = pd.DataFrame({
        'feature1': ['A', 'A', 'B', 'B'],
        'feature2': ['X', 'Y', 'X', 'Y'],
        'target': ['yes', 'no', 'no', 'yes']
    })
    best_feature, gini_score = gini_instance.get_best_feature(df, 'target')
    assert best_feature in ['feature1', 'feature2']
    assert 0 <= gini_score <= 1

def test_gini_perfect_split(gini_instance):
    df = pd.DataFrame({
        'feature': ['A', 'A', 'B', 'B'],
        'target': ['yes', 'yes', 'no', 'no']
    })
    best_feature, gini_score = gini_instance.get_best_feature(df, 'target')
    assert best_feature == 'feature'
    assert np.isclose(gini_score, 0.0)  # Perfect split gives 0 weighted gini
