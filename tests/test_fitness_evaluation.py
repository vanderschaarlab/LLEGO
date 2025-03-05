from typing import Dict

import numpy as np
import pandas as pd
import pytest

from llego.custom.fitness_evaluation import FitnessEvaluation
from llego.custom.generic_tree import GenericTree
from llego.custom.parsing_to_string import parse_dict_to_string
from llego.operators.individual import Individual


@pytest.fixture
def sample_data():
    """Fixture to create sample data for testing."""
    X_train = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "feature2": [5, 4, 3, 2, 1]})
    y_train = np.array([0, 0, 0, 1, 1])

    X_val = pd.DataFrame({"feature1": [2, 4, 6], "feature2": [8, 6, 4]})
    y_val = np.array([0, 1, 1])

    X_test = pd.DataFrame({"feature1": [1, 5], "feature2": [9, 7]})
    y_test = np.array([0, 1])

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


@pytest.fixture
def regression_data():
    """Fixture to create regression data for testing."""
    X_train = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "feature2": [5, 4, 3, 2, 1]})
    y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    X_val = pd.DataFrame({"feature1": [2, 4, 6], "feature2": [8, 6, 4]})
    y_val = np.array([2.0, 4.0, 6.0])

    X_test = pd.DataFrame({"feature1": [1, 5], "feature2": [9, 7]})
    y_test = np.array([1.0, 5.0])

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


@pytest.fixture
def classification_tree_dict():
    """Fixture to create a simple classification decision tree dictionary."""
    return {"feature1": {"< 3.5": {"value": 0}, ">= 3.5": {"value": 1}}}


@pytest.fixture
def regression_tree_dict():
    """Fixture to create a simple regression decision tree dictionary."""
    return {"feature1": {"< 3.5": {"value": 2.0}, ">= 3.5": {"value": 4.0}}}


@pytest.fixture
def create_classification_individual(classification_tree_dict):
    """Fixture to create a classification individual."""
    llm_format = parse_dict_to_string(classification_tree_dict)
    individual = Individual(
        machine_readable_format=classification_tree_dict, llm_readable_format=llm_format
    )
    # Add a dummy functional signature so we don't get an error when setting it
    individual.functional_signature = np.array([0, 0, 0, 1, 1])
    return individual


@pytest.fixture
def create_regression_individual(regression_tree_dict):
    """Fixture to create a regression individual."""
    llm_format = parse_dict_to_string(regression_tree_dict)
    individual = Individual(
        machine_readable_format=regression_tree_dict, llm_readable_format=llm_format
    )
    # Add a dummy functional signature so we don't get an error when setting it
    individual.functional_signature = np.array([2.0, 2.0, 2.0, 4.0, 4.0])
    return individual


@pytest.fixture
def multiple_classification_individuals():
    """Fixture to create multiple classification individuals with different thresholds."""
    individuals = []
    for threshold in [2.5, 3.5, 4.5]:
        tree_dict = {
            "feature1": {
                f"< {threshold}": {"value": 0},
                f">= {threshold}": {"value": 1},
            }
        }
        llm_format = parse_dict_to_string(tree_dict)
        individual = Individual(
            machine_readable_format=tree_dict, llm_readable_format=llm_format
        )
        # Add a dummy functional signature
        individual.functional_signature = np.array([0, 0, 0, 1, 1])
        individuals.append(individual)
    return individuals


@pytest.fixture
def multiple_regression_individuals():
    """Fixture to create multiple regression individuals with different thresholds."""
    individuals = []
    for threshold in [2.5, 3.5, 4.5]:
        tree_dict = {
            "feature1": {
                f"< {threshold}": {"value": 2.0},
                f">= {threshold}": {"value": 4.0},
            }
        }
        llm_format = parse_dict_to_string(tree_dict)
        individual = Individual(
            machine_readable_format=tree_dict, llm_readable_format=llm_format
        )
        # Add a dummy functional signature
        individual.functional_signature = np.array([2.0, 2.0, 2.0, 4.0, 4.0])
        individuals.append(individual)
    return individuals


class TestFitnessEvaluation:

    def test_init_valid_inputs(self, sample_data):
        """Test initialization with valid inputs."""
        fitness_eval = FitnessEvaluation(
            data=sample_data,
            task_type="classification",
            fitness_metric="accuracy",
            complexity_metric="depth",
            lower_is_better=False,
        )

        assert fitness_eval.task_type == "classification"
        assert fitness_eval.fitness_name == "accuracy"
        assert fitness_eval.complexity_name == "depth"
        assert not fitness_eval.lower_is_better

    def test_init_invalid_data_type(self):
        """Test initialization with invalid data types."""
        invalid_data = {
            "X_train": np.array([1, 2, 3]),  # Not a DataFrame
            "y_train": np.array([0, 1, 0]),
            "X_val": pd.DataFrame({"feature": [4, 5]}),
            "y_val": np.array([1, 0]),
            "X_test": pd.DataFrame({"feature": [6]}),
            "y_test": np.array([1]),
        }

        with pytest.raises(AssertionError):
            FitnessEvaluation(
                data=invalid_data,
                task_type="classification",
                fitness_metric="accuracy",
                complexity_metric="depth",
                lower_is_better=False,
            )

    def test_init_invalid_target_type(self, sample_data):
        """Test initialization with invalid target types."""
        invalid_data = sample_data.copy()
        invalid_data["y_train"] = pd.Series([0, 1, 0, 1, 0])  # Not a numpy array

        with pytest.raises(AssertionError):
            FitnessEvaluation(
                data=invalid_data,
                task_type="classification",
                fitness_metric="accuracy",
                complexity_metric="depth",
                lower_is_better=False,
            )

    def test_init_invalid_fitness_metric(self, sample_data):
        """Test initialization with invalid fitness metric."""
        with pytest.raises(AssertionError):
            FitnessEvaluation(
                data=sample_data,
                task_type="classification",
                fitness_metric="invalid_metric",  # Invalid metric
                complexity_metric="depth",
                lower_is_better=False,
            )

    def test_init_invalid_complexity_metric(self, sample_data):
        """Test initialization with invalid complexity metric."""
        with pytest.raises(AssertionError):
            FitnessEvaluation(
                data=sample_data,
                task_type="classification",
                fitness_metric="accuracy",
                complexity_metric="invalid_metric",  # Invalid metric
                lower_is_better=False,
            )

    def test_init_invalid_lower_is_better_type(self, sample_data):
        """Test initialization with invalid lower_is_better type."""
        with pytest.raises(AssertionError):
            FitnessEvaluation(
                data=sample_data,
                task_type="classification",
                fitness_metric="accuracy",
                complexity_metric="depth",
                lower_is_better="False",  # Not a boolean
            )

    def test_calculate_depth(self, sample_data):
        """Test the _calculate_depth method."""
        fitness_eval = FitnessEvaluation(
            data=sample_data,
            task_type="classification",
            fitness_metric="accuracy",
            complexity_metric="depth",
            lower_is_better=False,
        )

        # Create a GenericTree with a known depth
        tree = GenericTree(task="classification")
        # Construct a simple tree that should have depth 1
        tree_dict = {"feature1": {"< 3": {"value": 0}, ">= 3": {"value": 1}}}
        tree.construct_tree(tree_dict)

        depth = fitness_eval._calculate_depth(tree)
        assert depth == 1  # The depth should be 1 for this simple tree

    def test_evaluate_fitness_classification(
        self, sample_data, create_classification_individual
    ):
        """Test evaluate_fitness method for classification task."""
        # Create fitness evaluator
        fitness_eval = FitnessEvaluation(
            data=sample_data,
            task_type="classification",
            fitness_metric="accuracy",
            complexity_metric="depth",
            lower_is_better=False,
        )

        # Evaluate fitness
        fitness_eval.evaluate_fitness(
            [create_classification_individual], fit_tree=False, verbose=True
        )

        # Check that fitness was updated with all expected keys
        assert "depth" in create_classification_individual.fitness
        assert "accuracy_train" in create_classification_individual.fitness
        assert "accuracy_val" in create_classification_individual.fitness
        assert "accuracy_test" in create_classification_individual.fitness

        # Verify the depth is as expected (1 for this simple tree)
        assert create_classification_individual.fitness["depth"] == 1

        # Check that functional signature was updated
        assert create_classification_individual.functional_signature is not None

        # Check machine_readable_format and llm_readable_format are set
        assert create_classification_individual.machine_readable_format is not None
        assert isinstance(create_classification_individual.llm_readable_format, str)

    def test_evaluate_fitness_regression(
        self, regression_data, create_regression_individual
    ):
        """Test evaluate_fitness method for regression task."""
        # Create fitness evaluator for regression
        fitness_eval = FitnessEvaluation(
            data=regression_data,
            task_type="regression",
            fitness_metric="mse",
            complexity_metric="depth",
            lower_is_better=True,  # For MSE, lower is better
        )

        # Evaluate fitness
        fitness_eval.evaluate_fitness(
            [create_regression_individual], fit_tree=False, verbose=True
        )

        # Check that fitness was updated with all expected keys
        assert "depth" in create_regression_individual.fitness
        assert "mse_train" in create_regression_individual.fitness
        assert "mse_val" in create_regression_individual.fitness
        assert "mse_test" in create_regression_individual.fitness

        # Verify the depth is as expected (1 for this simple tree)
        assert create_regression_individual.fitness["depth"] == 1

        # Check that functional signature was updated
        assert create_regression_individual.functional_signature is not None

        # Check machine_readable_format and llm_readable_format are set
        assert create_regression_individual.machine_readable_format is not None
        assert isinstance(create_regression_individual.llm_readable_format, str)

    def test_evaluate_fitness_with_fit_tree(
        self, sample_data, create_classification_individual
    ):
        """Test evaluate_fitness with fit_tree=True, which should raise an assertion error."""
        fitness_eval = FitnessEvaluation(
            data=sample_data,
            task_type="classification",
            fitness_metric="accuracy",
            complexity_metric="depth",
            lower_is_better=False,
        )

        with pytest.raises(AssertionError):
            fitness_eval.evaluate_fitness(
                [create_classification_individual], fit_tree=True
            )

    def test_balanced_accuracy_metric(
        self, sample_data, create_classification_individual
    ):
        """Test using balanced_accuracy as a fitness metric."""
        # Create fitness evaluator with balanced_accuracy
        fitness_eval = FitnessEvaluation(
            data=sample_data,
            task_type="classification",
            fitness_metric="balanced_accuracy",
            complexity_metric="depth",
            lower_is_better=False,
        )

        # Evaluate fitness
        fitness_eval.evaluate_fitness(
            [create_classification_individual], fit_tree=False, verbose=True
        )

        # Check that fitness was calculated using balanced_accuracy
        assert "balanced_accuracy_train" in create_classification_individual.fitness
        assert "balanced_accuracy_val" in create_classification_individual.fitness
        assert "balanced_accuracy_test" in create_classification_individual.fitness

    def test_multiple_individuals_higher_better(
        self, sample_data, multiple_classification_individuals
    ):
        """Test evaluating multiple individuals when higher fitness is better."""
        # Create fitness evaluator with higher is better
        fitness_eval = FitnessEvaluation(
            data=sample_data,
            task_type="classification",
            fitness_metric="accuracy",
            complexity_metric="depth",
            lower_is_better=False,  # Higher is better
        )

        # Evaluate fitness
        fitness_eval.evaluate_fitness(
            multiple_classification_individuals, fit_tree=False, verbose=True
        )

        # Check all individuals have fitness values
        for ind in multiple_classification_individuals:
            assert "accuracy_train" in ind.fitness
            assert "depth" in ind.fitness

    def test_multiple_individuals_lower_better(
        self, regression_data, multiple_regression_individuals
    ):
        """Test evaluating multiple individuals when lower fitness is better."""
        # Create fitness evaluator with lower is better
        fitness_eval = FitnessEvaluation(
            data=regression_data,
            task_type="regression",
            fitness_metric="mse",
            complexity_metric="depth",
            lower_is_better=True,  # Lower is better
        )

        # Evaluate fitness
        fitness_eval.evaluate_fitness(
            multiple_regression_individuals, fit_tree=False, verbose=True
        )

        # Check all individuals have fitness values
        for ind in multiple_regression_individuals:
            assert "mse_train" in ind.fitness
            assert "depth" in ind.fitness
