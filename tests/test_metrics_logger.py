from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from llego.operators.filter_operator import Filter
from llego.operators.individual import Individual
from llego.operators.metrics_logger import MetricsLogger


@pytest.fixture
def mock_filter():
    """Create a mock Filter operator."""
    filter_mock = MagicMock(spec=Filter)

    # Configure the filter to return half the population
    def mock_filter_population(population):
        return population[: len(population) // 2]

    filter_mock.filter_population.side_effect = mock_filter_population
    return filter_mock


@pytest.fixture
def create_individual():

    def _create_individual(fitness_dict, functional_signature):
        individual = Individual()
        individual.fitness = fitness_dict
        individual.functional_signature = functional_signature
        return individual

    return _create_individual


@pytest.fixture
def sample_population(create_individual):
    """Create a sample population of individuals for testing."""
    population = []

    for i in range(10):
        fitness = {
            "accuracy_train": 0.7 + i * 0.02,
            "accuracy_val": 0.65 + i * 0.02,
            "accuracy_test": 0.6 + i * 0.02,
            "depth": i + 1,
        }

        func_sig = np.zeros(20)
        func_sig[i : i + 10] = 1

        population.append(create_individual(fitness, func_sig))

    return population


class TestMetricsLogger:

    def test_init(self, mock_filter):
        """Test initialization of MetricsLogger."""
        logger = MetricsLogger(log_wandb=False, filter=mock_filter)
        assert logger.log_wandb is False
        assert logger.filter_operator == mock_filter
        assert logger.PREFIX == ["population", "crossover", "mutation"]
        assert logger.population_across_iterations == {}

    def test_compute_fitness_statistics(self, sample_population):
        """Test computation of fitness statistics."""
        logger = MetricsLogger(log_wandb=False, filter=MagicMock())

        stats = logger.compute_fitness_statistics(sample_population)

        # Check if all expected keys are present
        expected_keys = [
            "accuracy_train/mean",
            "accuracy_train/median",
            "accuracy_train/min",
            "accuracy_train/max",
            "accuracy_val/mean",
            "accuracy_val/median",
            "accuracy_val/min",
            "accuracy_val/max",
            "accuracy_test/mean",
            "accuracy_test/median",
            "accuracy_test/min",
            "accuracy_test/max",
            "depth/mean",
            "depth/median",
            "depth/min",
            "depth/max",
        ]

        for key in expected_keys:
            assert key in stats

        # Check if the statistics are calculated correctly
        assert stats["accuracy_train/min"] == 0.7
        assert stats["accuracy_train/max"] == 0.7 + 0.02 * 9
        assert stats["depth/min"] == 1
        assert stats["depth/max"] == 10

    def test_compute_diversity(self, sample_population):
        """Test computation of diversity metrics."""
        logger = MetricsLogger(log_wandb=False, filter=MagicMock())

        diversity = logger.compute_diversity(sample_population)

        # Check if all expected keys are present
        assert "mean_l1_distance" in diversity
        assert "median_l1_distance" in diversity

        # Ensure the diversity values are positive
        assert diversity["mean_l1_distance"] > 0
        assert diversity["median_l1_distance"] > 0

    def test_compute_uniqueness(self, sample_population, mock_filter):
        """Test computation of uniqueness metric."""
        logger = MetricsLogger(log_wandb=False, filter=mock_filter)

        uniqueness = logger.compute_uniqueness(sample_population)

        # Check if uniqueness key is present
        assert "uniqueness" in uniqueness

        # Since our mock filter returns half the population, uniqueness should be 0.5
        assert uniqueness["uniqueness"] == 0.5

    def test_log_population(self, sample_population, mock_filter):
        """Test logging of population metrics."""
        logger = MetricsLogger(log_wandb=False, filter=mock_filter)

        # Test with valid prefix
        logs = logger.log_population(sample_population, step=0, prefix="population")

        # Check if the population is stored
        assert 0 in logger.population_across_iterations
        assert "population" in logger.population_across_iterations[0]
        assert len(logger.population_across_iterations[0]["population"]) == len(
            sample_population
        )

        # Check if all expected metrics are logged with the correct prefix
        for key in logs:
            assert key.startswith("population/")

        # Ensure diversity and uniqueness metrics are included
        assert "population/mean_l1_distance" in logs
        assert "population/uniqueness" in logs

    def test_log_population_invalid_prefix(self, sample_population, mock_filter):
        """Test logging with invalid prefix."""
        logger = MetricsLogger(log_wandb=False, filter=mock_filter)

        # Test with invalid prefix
        with pytest.raises(AssertionError) as excinfo:
            logger.log_population(sample_population, step=0, prefix="invalid_prefix")

        assert "prefix should be one of" in str(excinfo.value)

    @patch("wandb.log")
    def test_log_population_with_wandb(
        self, mock_wandb_log, sample_population, mock_filter
    ):
        """Test logging of population metrics with wandb enabled."""
        logger = MetricsLogger(log_wandb=True, filter=mock_filter)

        logs = logger.log_population(sample_population, step=1, prefix="population")

        # Check if wandb.log was called with the correct arguments
        mock_wandb_log.assert_called_once()
        call_args = mock_wandb_log.call_args[0][0]
        assert isinstance(call_args, dict)
        assert mock_wandb_log.call_args[1]["step"] == 1

        # Check if the same logs are returned and passed to wandb
        for key in logs:
            assert key in call_args

    def test_get_population_across_iterations(self, sample_population, mock_filter):
        """Test retrieval of population across iterations."""
        logger = MetricsLogger(log_wandb=False, filter=mock_filter)

        # Log population for multiple iterations
        logger.log_population(sample_population, step=0, prefix="population")
        logger.log_population(
            sample_population[:5], step=1, prefix="population"
        )  # Smaller population

        # Get population across iterations
        populations = logger.get_population_across_iterations(num_iterations=1)

        # Check if all iterations are present
        assert 0 in populations
        assert 1 in populations

        # Check if the populations have the correct size
        assert len(populations[0]["population"]) == len(sample_population)
        assert len(populations[1]["population"]) == 5

    def test_get_population_missing_iterations(self, sample_population, mock_filter):
        """Test retrieval of population with missing iterations."""
        logger = MetricsLogger(log_wandb=False, filter=mock_filter)

        # Log population for only iteration 0
        logger.log_population(sample_population, step=0, prefix="population")

        # Try to get population for iterations 0 and 1
        with pytest.raises(AssertionError):
            logger.get_population_across_iterations(num_iterations=1)

    def test_multiple_prefixes(self, sample_population, mock_filter):
        """Test logging with different prefixes."""
        logger = MetricsLogger(log_wandb=False, filter=mock_filter)

        # Log population with different prefixes
        logger.log_population(sample_population, step=0, prefix="population")
        logger.log_population(sample_population[:8], step=0, prefix="crossover")
        logger.log_population(sample_population[:6], step=0, prefix="mutation")

        # Check if all prefixes are stored for the same iteration
        assert "population" in logger.population_across_iterations[0]
        assert "crossover" in logger.population_across_iterations[0]
        assert "mutation" in logger.population_across_iterations[0]

        # Check if the populations have the correct size
        assert len(logger.population_across_iterations[0]["population"]) == len(
            sample_population
        )
        assert len(logger.population_across_iterations[0]["crossover"]) == 8
        assert len(logger.population_across_iterations[0]["mutation"]) == 6
