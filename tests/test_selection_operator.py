from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from llego.operators.filter_operator import Filter
from llego.operators.individual import Individual
from llego.operators.selection_operator import SelectionOperator


@pytest.fixture
def create_individual():

    def _create_individual(fitness_dict):
        individual = Individual()
        individual.fitness = fitness_dict
        individual.functional_signature = np.array([0, 1, 0, 1])
        individual.machine_readable_format = {"test": "model"}
        individual.llm_readable_format = "test model"
        return individual

    return _create_individual


@pytest.fixture
def mock_filter():
    filter_mock = MagicMock(spec=Filter)

    def mock_filter_population(population):
        return population

    filter_mock.filter_population.side_effect = mock_filter_population
    return filter_mock


@pytest.fixture
def sample_population(create_individual):
    """Create a sample population with 10 individuals with varying fitness."""
    population = []

    for i in range(10):
        fitness = {
            "accuracy": 0.6 + i * 0.03,  # 0.6 to 0.87
            "depth": 10 - i,  # 10 to 1
        }
        population.append(create_individual(fitness))

    return population


class TestSelectionOperator:

    def test_init_valid(self, mock_filter):
        """Test initialization with valid parameters."""
        # Test with filter
        selection_op = SelectionOperator(
            filter=mock_filter, sorting_key="accuracy", lower_is_better=False
        )
        assert selection_op.filter == mock_filter
        assert selection_op.sorting_key == "accuracy"
        assert selection_op.lower_is_better is False

    @patch("llego.operators.selection_operator.logger")
    def test_select_with_filter_higher_is_better(
        self, mock_logger, sample_population, mock_filter
    ):
        """Test selection with filter when higher fitness is better."""
        pop_size = 4
        selection_op = SelectionOperator(
            filter=mock_filter,
            sorting_key="accuracy",
            lower_is_better=False,  # Higher accuracy is better
        )

        result = selection_op.select(sample_population, pop_size)

        # Check result size
        assert len(result) == pop_size

        accuracies = [ind.fitness["accuracy"] for ind in result]
        assert all(
            acc >= 0.69 for acc in accuracies
        ), f"Got accuracies {accuracies}"  # Individuals with indices 3-4 have accuracy >= 0.69

    @patch("llego.operators.selection_operator.logger")
    def test_select_without_filter(self, mock_logger, sample_population):
        """Test selection without filter."""
        pop_size = 5
        selection_op = SelectionOperator(
            filter=None,
            sorting_key="accuracy",
            lower_is_better=False,  # Higher accuracy is better
        )

        result = selection_op.select(sample_population, pop_size)

        # Check result size
        assert len(result) == pop_size

        # Without filter, expect individuals with highest accuracy (indices 9, 8, 7, 6, 5)
        accuracies = [ind.fitness["accuracy"] for ind in result]
        assert all(
            acc >= 0.75 for acc in accuracies
        )  # Individuals with indices 5-9 have accuracy >= 0.75

    @patch("llego.operators.selection_operator.logger")
    def test_select_filtered_population_smaller_than_popsize(
        self, mock_logger, sample_population, mock_filter
    ):
        """Test selection when filtered population is smaller than required population size."""
        # Create a filter that returns only 2 individuals
        strict_filter = MagicMock(spec=Filter)
        strict_filter.filter_population.return_value = sample_population[:2]

        pop_size = 6
        selection_op = SelectionOperator(
            filter=strict_filter, sorting_key="accuracy", lower_is_better=False
        )

        result = selection_op.select(sample_population, pop_size)

        # Check result size
        assert len(result) == pop_size

        # The 2 individuals should be repeated to reach the required size
        unique_individuals = set(id(ind) for ind in result)
        assert len(unique_individuals) <= 2

    @patch("llego.operators.selection_operator.logger")
    def test_select_equal_fitness_values(self, mock_logger, create_individual):
        """Test selection when all individuals have the same fitness."""
        # Create a population where all individuals have identical fitness
        population = [
            create_individual({"accuracy": 0.8, "depth": 5}) for _ in range(10)
        ]

        pop_size = 5
        selection_op = SelectionOperator(
            filter=None, sorting_key="accuracy", lower_is_better=False
        )

        result = selection_op.select(population, pop_size)

        # Check result size
        assert len(result) == pop_size

        # All individuals should have the same fitness
        accuracies = [ind.fitness["accuracy"] for ind in result]
        assert all(acc == 0.8 for acc in accuracies)

    @patch("llego.operators.selection_operator.logger")
    def test_select_missing_fitness_key(self, mock_logger, create_individual):
        """Test selection with a missing fitness key."""
        # Create individuals with different fitness keys
        population = [
            create_individual({"accuracy": 0.8}),
            create_individual({"performance": 0.7}),  # Missing 'accuracy' key
            create_individual({"accuracy": 0.9}),
        ]

        pop_size = 2
        selection_op = SelectionOperator(
            filter=None, sorting_key="accuracy", lower_is_better=False
        )

        with pytest.raises(KeyError):
            selection_op.select(population, pop_size)
