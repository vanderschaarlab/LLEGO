import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import openml
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.utils.class_weight import compute_class_weight

from utils.private_data_loader import load_cutract, load_maggic, load_seer

warnings.simplefilter(action="ignore", category=FutureWarning)

import logging

logger = logging.getLogger(__name__)

DATASET_IDS = {
    "credit-g": [31, "classification"],
    "diabetes": [37, "classification"],
    "compas": [42192, "classification"],
    "heart-statlog": [53, "classification"],
    "liver": [1480, "classification"],
    "breast": [15, "classification"],
    "vehicle": [994, "classification"],
    "cholesterol": [204, "regression"],
    "wine": [287, "regression"],
    "wage": [534, "regression"],
    "abalone": [44956, "regression"],
    "cars": [44994, "regression"],
}


def get_raw_data(
    dataset_name: str,
) -> Tuple[pd.DataFrame, pd.Series, List[bool], List[str], str, str]:
    # if openml dataset
    if dataset_name in DATASET_IDS:
        data_id = DATASET_IDS[dataset_name][0]
        assert isinstance(data_id, int), "data_id must be an integer"
        dataset = openml.datasets.get_dataset(data_id)
        X, y, categorical_mask, attribute_names = dataset.get_data(
            target=dataset.default_target_attribute
        )
        target_name = dataset.default_target_attribute
        task_type = DATASET_IDS[dataset_name][1]

    # private dataset - cutract
    elif dataset_name == "cutract":
        X, y, categorical_mask, attribute_names, target_name = load_cutract()
        task_type = "classification"
    # private dataset - seer
    elif dataset_name == "seer":
        X, y, categorical_mask, attribute_names, target_name = load_seer()
        task_type = "classification"
    elif dataset_name == "maggic":
        X, y, categorical_mask, attribute_names, target_name = load_maggic()
        task_type = "classification"
    else:
        raise ValueError(f"Dataset {dataset_name} not found")

    # assert X is pandas dataframe and Y is pandas series
    assert isinstance(X, pd.DataFrame), "X must be a pandas DataFrame"
    assert isinstance(y, pd.Series), "y must be a pandas Series"
    assert (
        isinstance(categorical_mask, list) or categorical_mask is None
    ), "categorical_mask must be a list"
    assert isinstance(attribute_names, list), "attribute_names must be a list"
    assert isinstance(target_name, str), "target_name must be a string"
    assert isinstance(task_type, str), "task_type must be a string"
    return X, y, categorical_mask, attribute_names, target_name, task_type


def get_data(
    dataset_name: str,
    dataset_details: dict,
    max_samples: Optional[int] = None,
    include_task_semantics: bool = True,
):

    X, y, categorical_mask, attribute_names, target_name, task_type = get_raw_data(
        dataset_name
    )
    # task_type = DATASET_IDS[dataset_name][1]

    if max_samples is not None:
        idx = np.random.choice(X.shape[0], max_samples, replace=False)
        X = X[idx]
        y = y[idx]

    n_y_unique = len(set(y))
    if task_type == "classification":
        assert (
            n_y_unique == 2
        ), f"Target variable must be binary, found {n_y_unique} unique values"
    else:
        assert (
            n_y_unique > 2
        ), f"Target variable cannot be binary, found {n_y_unique} unique values"

    # check if has attribute categorical_mask
    if "categorical_mask" in dataset_details:
        logger.info(
            f"[Data Loader] Dataset: {dataset_name}, using custom categorical mask..."
        )
        categorical_mask = dataset_details["categorical_mask"]
        categorical_mask = list(categorical_mask)
        # convert categorical_mask to boolean array
        categorical_mask = [bool(item) for item in categorical_mask]

    if "attribute_names" in dataset_details:
        logger.info(
            f"[Data Loader] Dataset: {dataset_name}, using custom attribute names..."
        )
        attribute_names = dataset_details["attribute_names"]
        attribute_names = list(attribute_names)
        X.columns = attribute_names

    if include_task_semantics:
        task_description = dataset_details["task_description"]
    else:
        task_description = "The task is to generate interpretable and high-performing decision trees given a set of attributes"
        attribute_names = [f"X_{i}" for i in range(X.shape[1])]
        target_name = "y"
        X.columns = attribute_names

    target_type = "binary" if task_type == "classification" else "continuous"

    meta_data = {
        "task_description": task_description,
        "n_attributes": X.shape[1],
        "n_numerical": X.shape[1] - sum(categorical_mask),
        "n_categorical": sum(categorical_mask),
        "target_name": target_name,
        "attribute_names": attribute_names,
        "categorical_mask": categorical_mask,
        "target_type": target_type,
        "task_type": task_type,
    }
    return X, y, meta_data


def sort_categories_by_response(
    X_train: pd.DataFrame, y_train: np.ndarray, cat_columns: List
) -> list:
    """
    Sort categories by average response value
    """
    list_sorted_categories = []
    temp_combined = X_train.copy()
    temp_combined["target"] = y_train

    for col in cat_columns:
        averages = temp_combined.groupby(col)["target"].mean().reset_index()
        sorted_categories = averages.sort_values("target")[col].tolist()
        # convert each element to str
        sorted_categories = [str(cat) for cat in sorted_categories]
        list_sorted_categories.append(sorted_categories)

    return list_sorted_categories


def get_label_information(y_train: np.ndarray, task_type: str) -> str:
    """
    Get label information. For classification, return label distribution, for regression return mean and std.
    """
    if task_type == "classification":
        unique, counts = np.unique(y_train, return_counts=True)
        total = y_train.size
        percentages = [
            f"{label}: {(count / total * 100):.2f}%"
            for label, count in zip(unique, counts)
        ]
        label_information = "the label distribution is [" + ", ".join(percentages) + "]"
    else:
        mean = np.mean(y_train)
        std = np.std(y_train)
        if mean >= -1e-3 or mean <= 1e-3:
            mean = 0.0
        label_information = (
            f"the mean of the label is {mean:.2f} and the std is {std:.2f}"
        )
    return label_information


def get_feature_semantics(X_train: pd.DataFrame, cat_columns: List) -> str:
    feature_semantics = []
    for i, column in enumerate(X_train.columns):
        feature_type = "int" if column in cat_columns else "float"
        min_val, max_val = (X_train[column].min(), X_train[column].max())
        feature_name = column
        if feature_type == "float":
            feature_semantics.append(
                f"{feature_name} ({feature_type}) [{min_val:.2f}, {max_val:.2f}]"
            )
        else:
            feature_semantics.append(
                f"{feature_name} ({feature_type}) [{int(min_val)}, {int(max_val)}]"
            )

    assert (
        len(feature_semantics) == X_train.shape[1]
    ), "Number of feature semantics must match number of columns"

    feature_semantics_str = "[" + ", ".join(feature_semantics) + "]"

    return feature_semantics_str


def impute(X, cat_columns, cont_columns):
    if len(cont_columns) > 0:
        median_imputer = SimpleImputer(strategy="median")
        X[cont_columns] = median_imputer.fit_transform(X[cont_columns])
    if len(cat_columns) > 0:
        mode_imputer = SimpleImputer(strategy="most_frequent")
        X[cat_columns] = mode_imputer.fit_transform(X[cat_columns])
    return X


def preprocess_data(
    X: pd.DataFrame,
    y: pd.DataFrame,
    meta_data: Dict,
    train_val_test_split: List[float] = [0.6, 0.2, 0.2],
    seed: int = 42,
) -> Tuple[Dict, Dict]:
    """
    Preprocess data for training, val, test split
    """

    categorical_mask = meta_data["categorical_mask"]
    task_type = meta_data["task_type"]

    assert isinstance(categorical_mask, list), "categorical_mask must be a list"
    assert all(
        isinstance(item, bool) for item in categorical_mask
    ), "categorical_mask must be a list of booleans"

    # Step 1: impute missing values using median for numerical and most frequent for categorical
    cat_columns = [
        col
        for col, is_categorical in zip(X.columns, categorical_mask)
        if is_categorical
    ]
    cont_columns = [
        col
        for col, is_categorical in zip(X.columns, categorical_mask)
        if not is_categorical
    ]

    assert len(cat_columns) == sum(
        categorical_mask
    ), "Number of cateogrical columns must match sum of categorical_mask"
    assert len(cont_columns) == len(categorical_mask) - sum(
        categorical_mask
    ), "Number of continuous columns must match sum of negation of categorical_mask"

    X = impute(X, cat_columns, cont_columns)

    test_size = train_val_test_split[2]
    val_size = train_val_test_split[1]

    # Step 2: split data into train, validation and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=0
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=seed
    )

    # Step 3: target encoding
    if meta_data["target_type"] == "continuous":
        # Regression - standardization for target
        y_train_mean = y_train.mean()
        y_train_std = y_train.std()
        y_train = (y_train - y_train_mean) / y_train_std
        y_val = (y_val - y_train_mean) / y_train_std
        y_test = (y_test - y_train_mean) / y_train_std
        y_train = y_train.to_numpy()
        y_val = y_val.to_numpy()
        y_test = y_test.to_numpy()
    else:
        # Classification - label encoding for target
        ordinal_encoder_target = LabelEncoder()
        y_train = ordinal_encoder_target.fit_transform(y_train)
        y_val = ordinal_encoder_target.transform(y_val)
        y_test = ordinal_encoder_target.transform(y_test)

    # Step 4: ordinal encoding for categorical features
    if len(cat_columns) > 0:
        # Order categories by average response value - see ESL S9.2.4
        list_sorted_categories = sort_categories_by_response(
            X_train, y_train, cat_columns
        )

        ordinal_encoder = OrdinalEncoder(
            categories=list_sorted_categories,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        X_train[cat_columns] = ordinal_encoder.fit_transform(
            X_train[cat_columns].astype(str)
        )
        X_val[cat_columns] = ordinal_encoder.transform(X_val[cat_columns].astype(str))
        X_test[cat_columns] = ordinal_encoder.transform(X_test[cat_columns].astype(str))

        cat_feature_categories = ordinal_encoder.categories_
        cat_feature_names = ordinal_encoder.feature_names_in_

        cat_feature_categories_lists = [
            list(categories) for categories in cat_feature_categories
        ]

        cat_features_value_mapping = {
            feature_name: categories
            for feature_name, categories in zip(
                cat_feature_names, cat_feature_categories_lists
            )
        }
    else:
        cat_features_value_mapping = {}

    meta_data["n_samples"] = X_train.shape[0]
    meta_data["cat_feature_value_mapping"] = cat_features_value_mapping
    meta_data["cat_columns"] = cat_columns
    meta_data["label_information"] = get_label_information(y_train, task_type=task_type)
    meta_data["feature_semantics"] = get_feature_semantics(X_train, cat_columns)

    data = {}
    data["X_train"] = X_train
    data["X_val"] = X_val
    data["X_test"] = X_test
    data["y_train"] = y_train
    data["y_val"] = y_val
    data["y_test"] = y_test

    logger.info(f"[Data Processor] Successfully preprocessed data...")

    return data, meta_data


def binarize_dataset(
    X: pd.DataFrame,
    y: pd.DataFrame,
    meta_data: Dict,
    train_val_test_split: List[float] = [0.6, 0.2, 0.2],
    seed: int = 42,
) -> Tuple[Dict, Dict]:

    categorical_mask = meta_data["categorical_mask"]

    assert isinstance(categorical_mask, list), "categorical_mask must be a list"
    assert all(
        isinstance(item, bool) for item in categorical_mask
    ), "categorical_mask must be a list of booleans"

    # Step 1: impute missing values using median for numerical and most frequent for categorical
    cat_columns = [
        col
        for col, is_categorical in zip(X.columns, categorical_mask)
        if is_categorical
    ]
    cont_columns = [
        col
        for col, is_categorical in zip(X.columns, categorical_mask)
        if not is_categorical
    ]

    assert len(cat_columns) == sum(
        categorical_mask
    ), "Number of cateogrical columns must match sum of categorical_mask"
    assert len(cont_columns) == len(categorical_mask) - sum(
        categorical_mask
    ), "Number of continuous columns must match sum of negation of categorical_mask"

    X = impute(X, cat_columns, cont_columns)

    test_size = train_val_test_split[2]
    val_size = train_val_test_split[1]

    # Step 2: split data into train, validation and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=0
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=seed
    )

    # Step 3: label encoding for target
    ordinal_encoder_target = LabelEncoder()
    y_train = ordinal_encoder_target.fit_transform(y_train)
    y_val = ordinal_encoder_target.transform(y_val)
    y_test = ordinal_encoder_target.transform(y_test)

    # Step 4: Binarize the continuous features, and one hot encote the categorical features
    X_train_cat_one_hot = pd.DataFrame()
    X_val_cat_one_hot = pd.DataFrame()
    X_test_cat_one_hot = pd.DataFrame()
    X_train_cont_bin = pd.DataFrame()
    X_val_cont_bin = pd.DataFrame()
    X_test_cont_bin = pd.DataFrame()

    if len(cat_columns) > 0:
        X_train_cat_one_hot = pd.get_dummies(X_train[cat_columns])

        X_val_cat_one_hot = pd.get_dummies(X_val[cat_columns])
        X_val_cat_one_hot = X_val_cat_one_hot.reindex(
            columns=X_train_cat_one_hot.columns, fill_value=0
        )

        X_test_cat_one_hot = pd.get_dummies(X_test[cat_columns])
        X_test_cat_one_hot = X_test_cat_one_hot.reindex(
            columns=X_train_cat_one_hot.columns, fill_value=0
        )

    if len(cont_columns) > 0:
        feat_binarizer = FeatureBinarizer()
        feat_binarizer.fit(X_train, cont_columns)
        X_train_cont_bin = feat_binarizer.transform(X_train[cont_columns])
        X_val_cont_bin = feat_binarizer.transform(X_val[cont_columns])
        X_test_cont_bin = feat_binarizer.transform(X_test[cont_columns])

    X_train = pd.concat([X_train_cat_one_hot, X_train_cont_bin], axis=1)
    X_val = pd.concat([X_val_cat_one_hot, X_val_cont_bin], axis=1)
    X_test = pd.concat([X_test_cat_one_hot, X_test_cont_bin], axis=1)

    meta_data["n_samples"] = X_train.shape[0]

    data = {}
    data["X_train"] = X_train
    data["X_val"] = X_val
    data["X_test"] = X_test
    data["y_train"] = y_train
    data["y_val"] = y_val
    data["y_test"] = y_test

    logger.info(f"[Data Binarizer] Successfully binarized data...")
    return (data, meta_data)


class FeatureBinarizer:
    def __init__(self) -> None:
        pass

    def fit(self, X: pd.DataFrame, cont_columns: List[str]) -> None:
        self.cont_columns = cont_columns
        self.thresholds = {}
        for col in cont_columns:
            self.thresholds[col] = X[col].unique()

    def transform(self, X: pd.DataFrame):
        binarized_columns = pd.DataFrame()
        for col in self.cont_columns:
            thresholds = self.thresholds[col]
            for threshold in thresholds:
                binarized_columns[col + f"_gt_{threshold}"] = (
                    X[col] > threshold
                ).astype(int)
        return binarized_columns


def compute_sample_weights(y_train):
    """
    Compute sample weights for balanced classification.

    Parameters:
    -----------
    y_train : array-like
        Training labels

    Returns:
    --------
    sample_weights : array
        Weight for each sample to achieve balanced classes
    """
    # Get unique classes and their counts
    classes = np.unique(y_train)

    # Compute class weights (inverse of frequency)
    class_weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=y_train
    )

    # Create a dictionary mapping class to weight
    class_weight_dict = dict(zip(classes, class_weights))

    # Map weights to samples
    sample_weights = np.array([class_weight_dict[c] for c in y_train])

    return sample_weights, class_weight_dict
