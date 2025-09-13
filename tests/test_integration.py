import pytest
import pandas as pd
import numpy as np
from decisiontree.tree import Tree
from decisiontree.ImpurityStrategy.Entropy import Entropy
from decisiontree.ImpurityStrategy.GiniIndex import GiniIndex

class TestDecisionTreeIntegration:
    """Integration tests for the complete decision tree functionality."""
    
    @pytest.fixture
    def tennis_dataset(self):
        """Classic tennis dataset for testing."""
        return pd.DataFrame({
            'outlook': ['sunny', 'sunny', 'overcast', 'rainy', 'rainy', 'rainy', 
                       'overcast', 'sunny', 'sunny', 'rainy', 'sunny', 'overcast', 
                       'overcast', 'rainy'],
            'temperature': ['hot', 'hot', 'hot', 'mild', 'cool', 'cool', 'cool', 
                           'mild', 'cool', 'mild', 'mild', 'mild', 'hot', 'mild'],
            'humidity': ['high', 'high', 'high', 'high', 'normal', 'normal', 
                        'normal', 'high', 'normal', 'normal', 'normal', 'high', 
                        'normal', 'high'],
            'windy': ['false', 'true', 'false', 'false', 'false', 'true', 'true', 
                     'false', 'false', 'false', 'true', 'true', 'false', 'true'],
            'play': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 
                    'yes', 'yes', 'yes', 'yes', 'no']
        })
    
    @pytest.fixture
    def simple_dataset(self):
        """Simple dataset for basic testing."""
        return pd.DataFrame({
            'feature1': ['A', 'A', 'B', 'B'],
            'feature2': ['X', 'Y', 'X', 'Y'],
            'target': ['yes', 'no', 'no', 'yes']
        })
    
    def test_entropy_tree_training(self, tennis_dataset):
        """Test decision tree training with entropy criterion."""
        entropy = Entropy()
        tree = Tree(entropy)
        tree.fit(tennis_dataset, 'play')
        
        assert tree.tree is not None
        assert isinstance(tree.tree, dict)
        assert tree.target == 'play'
    
    def test_gini_tree_training(self, tennis_dataset):
        """Test decision tree training with Gini criterion."""
        gini = GiniIndex()
        tree = Tree(gini)
        tree.fit(tennis_dataset, 'play')
        
        assert tree.tree is not None
        assert isinstance(tree.tree, dict)
        assert tree.target == 'play'
    
    def test_tree_prediction(self, simple_dataset):
        """Test tree prediction functionality."""
        entropy = Entropy()
        tree = Tree(entropy)
        tree.fit(simple_dataset, 'target')
        
        # Test prediction with known sample
        test_sample = {'feature1': 'A', 'feature2': 'X'}
        prediction = tree.predict(test_sample)
        assert prediction in ['yes', 'no']
    
    def test_tree_with_pure_target(self):
        """Test tree behavior with pure target variable."""
        df = pd.DataFrame({
            'feature': ['A', 'B', 'C'],
            'target': ['yes', 'yes', 'yes']
        })
        entropy = Entropy()
        tree = Tree(entropy)
        tree.fit(df, 'target')
        
        # Tree should just return the pure value
        assert tree.tree == 'yes'
    
    def test_tree_multiple_predictions(self, tennis_dataset):
        """Test multiple predictions on tennis dataset."""
        entropy = Entropy()
        tree = Tree(entropy)
        tree.fit(tennis_dataset, 'play')
        
        test_cases = [
            {'outlook': 'sunny', 'temperature': 'hot', 'humidity': 'high', 'windy': 'false'},
            {'outlook': 'overcast', 'temperature': 'cool', 'humidity': 'normal', 'windy': 'true'},
            {'outlook': 'rainy', 'temperature': 'mild', 'humidity': 'normal', 'windy': 'false'}
        ]
        
        for test_case in test_cases:
            prediction = tree.predict(test_case)
            assert prediction in ['yes', 'no']
    
    def test_entropy_vs_gini_comparison(self, tennis_dataset):
        """Compare results between entropy and Gini criteria."""
        entropy = Entropy()
        gini = GiniIndex()
        
        entropy_tree = Tree(entropy)
        gini_tree = Tree(gini)
        
        entropy_tree.fit(tennis_dataset, 'play')
        gini_tree.fit(tennis_dataset, 'play')
        
        # Both should produce valid trees
        assert entropy_tree.tree is not None
        assert gini_tree.tree is not None
        
        # Test that both can make predictions
        test_sample = {'outlook': 'sunny', 'temperature': 'hot', 'humidity': 'high', 'windy': 'false'}
        entropy_pred = entropy_tree.predict(test_sample)
        gini_pred = gini_tree.predict(test_sample)
        
        assert entropy_pred in ['yes', 'no']
        assert gini_pred in ['yes', 'no']
    
    def test_edge_case_single_feature(self):
        """Test with dataset having only one feature."""
        df = pd.DataFrame({
            'feature': ['A', 'A', 'B', 'B'],
            'target': ['yes', 'yes', 'no', 'no']
        })
        entropy = Entropy()
        tree = Tree(entropy)
        tree.fit(df, 'target')
        
        # Should work with single feature
        prediction = tree.predict({'feature': 'A'})
        assert prediction == 'yes'
        
        prediction = tree.predict({'feature': 'B'})
        assert prediction == 'no'
    
    def test_performance_metrics(self, tennis_dataset):
        """Test basic performance on training data."""
        entropy = Entropy()
        tree = Tree(entropy)
        tree.fit(tennis_dataset, 'play')
        
        # Count correct predictions on training data
        correct = 0
        total = len(tennis_dataset)
        
        for _, row in tennis_dataset.iterrows():
            test_sample = row.drop('play').to_dict()
            prediction = tree.predict(test_sample)
            if prediction == row['play']:
                correct += 1
        
        accuracy = correct / total
        # Should have reasonable accuracy on training data
        assert accuracy >= 0.7  # At least 70% accuracy
