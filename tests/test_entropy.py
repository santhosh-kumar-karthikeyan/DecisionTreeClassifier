import pytest
import pandas as pd
import numpy as np
from decisiontree.ImpurityStrategy.Entropy import Entropy
from decisiontree.ImpurityStrategy.Strategy import ImpurityStrategy

@pytest.fixture
def entropy_instance():
    return Entropy()

def test_entropy_is_subclass():
    assert issubclass(Entropy, ImpurityStrategy)

def test_entropy_impurity_measure_pure(entropy_instance):
    df = pd.DataFrame({'target': [0, 0, 0, 0]})
    result = entropy_instance._get_impurity_measure(df, 'target')
    assert np.isclose(result, 0.0, atol=1e-10)

def test_entropy_impurity_measure_balanced(entropy_instance):
    df = pd.DataFrame({'target': [1, 1, 0, 0]})
    result = entropy_instance._get_impurity_measure(df, 'target')
    # For balanced binary classification, entropy should be 1.0
    assert np.isclose(result, 1.0)

def test_entropy_splitting_criterion(entropy_instance):
    df = pd.DataFrame({
        'feature': [1, 1, 0, 0], 
        'target': [1, 1, 0, 0]
    })
    info_gain = entropy_instance._get_splitting_criterion(df, 'feature', 'target')
    # Perfect split should give maximum information gain (1.0 for this case)
def test_entropy_get_best_feature(entropy_instance):
    df = pd.DataFrame({
        'feature1': ['A', 'A', 'B', 'B'],
        'feature2': ['X', 'Y', 'X', 'Y'],
        'target': ['yes', 'no', 'no', 'yes']
    })
    best_feature, info_gain = entropy_instance.get_best_feature(df, 'target')
    assert best_feature in ['feature1', 'feature2']
    assert info_gain >= 0

def test_entropy_perfect_split(entropy_instance):
    df = pd.DataFrame({
        'feature': ['A', 'A', 'B', 'B'],
        'target': ['yes', 'yes', 'no', 'no']
    })
    best_feature, info_gain = entropy_instance.get_best_feature(df, 'target')
    assert best_feature == 'feature'
    assert np.isclose(info_gain, 1.0)  # Perfect split
