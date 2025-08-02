import json
import random
from typing import Dict, Any, Tuple
from datetime import datetime

class ABTestingFramework:
    """
    A/B testing framework for comparing different prompt versions and configurations.
    """
    
    def __init__(self):
        self.tests = {}
        self.results = {}
        self.active_tests = {}
    
    def create_test(self, test_name: str, prompt_variants: Dict[str, str], 
                   parameters: Dict[str, Any] = None) -> str:
        """
        Create a new A/B test for prompt variants.
        
        Args:
            test_name: Name of the test
            prompt_variants: Dictionary of variant names to prompt content
            parameters: Additional test parameters (sample size, metrics, etc.)
            
        Returns:
            Test ID
        """
        test_id = f"{test_name}_{int(datetime.now().timestamp())}"
        
        self.tests[test_id] = {
            'name': test_name,
            'variants': prompt_variants,
            'parameters': parameters or {},
            'created_at': datetime.now().isoformat(),
            'variant_counts': {variant: 0 for variant in prompt_variants},
            'variant_results': {variant: [] for variant in prompt_variants}
        }
        
        self.active_tests[test_name] = test_id
        
        return test_id
    
    def get_variant_for_test(self, test_name: str) -> Tuple[str, str]:
        """
        Get a random variant for the specified test.
        
        Args:
            test_name: Name of the test
            
        Returns:
            Tuple of (variant_name, prompt_content)
        """
        if test_name not in self.active_tests:
            raise ValueError(f"Test '{test_name}' not found")
        
        test_id = self.active_tests[test_name]
        test = self.tests[test_id]
        
        # Get random variant
        variants = list(test['variants'].keys())
        variant_name = random.choice(variants)
        
        # Update count
        test['variant_counts'][variant_name] += 1
        
        return variant_name, test['variants'][variant_name]
    
    def record_result(self, test_name: str, variant_name: str, 
                     result: Dict[str, Any], user_feedback: float = None):
        """
        Record the result of a test variant.
        
        Args:
            test_name: Name of the test
            variant_name: Name of the variant
            result: Result data (accuracy, response time, etc.)
            user_feedback: Optional user feedback score (0-1)
        """
        if test_name not in self.active_tests:
            raise ValueError(f"Test '{test_name}' not found")
        
        test_id = self.active_tests[test_name]
        test = self.tests[test_id]
        
        if variant_name not in test['variants']:
            raise ValueError(f"Variant '{variant_name}' not found in test '{test_name}'")
        
        result_data = {
            'timestamp': datetime.now().isoformat(),
            'result': result,
            'user_feedback': user_feedback
        }
        
        test['variant_results'][variant_name].append(result_data)
    
    def get_test_results(self, test_name: str) -> Dict[str, Any]:
        """
        Get results for a specific test.
        
        Args:
            test_name: Name of the test
            
        Returns:
            Test results summary
        """
        if test_name not in self.active_tests:
            raise ValueError(f"Test '{test_name}' not found")
        
        test_id = self.active_tests[test_name]
        test = self.tests[test_id]
        
        results = {}
        for variant_name, variant_results in test['variant_results'].items():
            if not variant_results:
                results[variant_name] = {
                    'count': 0,
                    'avg_accuracy': 0,
                    'avg_response_time': 0,
                    'avg_user_feedback': 0
                }
                continue
            
            # Calculate metrics
            total_accuracy = 0
            total_response_time = 0
            total_user_feedback = 0
            user_feedback_count = 0
            
            for result in variant_results:
                if 'accuracy' in result['result']:
                    total_accuracy += result['result']['accuracy']
                if 'response_time' in result['result']:
                    total_response_time += result['result']['response_time']
                if result['user_feedback'] is not None:
                    total_user_feedback += result['user_feedback']
                    user_feedback_count += 1
            
            count = len(variant_results)
            results[variant_name] = {
                'count': count,
                'avg_accuracy': total_accuracy / count if count > 0 else 0,
                'avg_response_time': total_response_time / count if count > 0 else 0,
                'avg_user_feedback': total_user_feedback / user_feedback_count if user_feedback_count > 0 else 0
            }
        
        return results
    
    def get_winner(self, test_name: str, metric: str = 'avg_accuracy') -> str:
        """
        Determine the winning variant based on a specific metric.
        
        Args:
            test_name: Name of the test
            metric: Metric to use for comparison
            
        Returns:
            Name of the winning variant
        """
        results = self.get_test_results(test_name)
        
        if not results:
            return None
        
        # Find variant with highest metric value
        winner = max(results.items(), key=lambda x: x[1].get(metric, 0))
        return winner[0]
    
    def end_test(self, test_name: str) -> Dict[str, Any]:
        """
        End a test and return final results.
        
        Args:
            test_name: Name of the test
            
        Returns:
            Final test results
        """
        if test_name not in self.active_tests:
            raise ValueError(f"Test '{test_name}' not found")
        
        results = self.get_test_results(test_name)
        
        # Remove from active tests
        self.active_tests.pop(test_name)
        
        return results
    
    def save_results(self, filepath: str):
        """
        Save test results to a JSON file.
        
        Args:
            filepath: Path to save results
        """
        data = {
            'tests': self.tests,
            'active_tests': self.active_tests
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_results(self, filepath: str):
        """
        Load test results from a JSON file.
        
        Args:
            filepath: Path to load results from
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.tests = data.get('tests', {})
        self.active_tests = data.get('active_tests', {})

# Global instance
ab_testing_framework = ABTestingFramework()