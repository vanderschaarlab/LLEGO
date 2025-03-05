import os

import pandas as pd


def load_dataset(path_csv: str, expected_columns: list) -> pd.DataFrame:
    dataset = pd.read_csv(path_csv)

    assert set(dataset.columns) == set(expected_columns), "Invalid dataset provided."

    return dataset


def load_cutract():
    project_dir = os.environ["PROJECT_DIR"]
    path_csv = os.path.join(project_dir, "private_data/CUTRACT.csv")
    print("=" * 100)
    print("PROCESSING CUTRACT...")

    expected_columns = [
        "Identifier",
        "AGE at diagnosis",
        "PSA at diagnosis",
        "Clincial stage at diagnosis",
        "Primary gleason",
        "Secondary Gleason",
        "Composite Gleason Score",
        "Grade Group",
        "cancer related death",
        "any cause of  death",
        "Days to death or current survival status",
        "COMORBIDITY if blank then this is 0",
        "PRIMARY_TREATMENT_TYPE",
    ]
    dataset = load_dataset(path_csv, expected_columns)

    print(f"[CUTRACT] original dataframe shape: {dataset.shape}")

    X = dataset.drop(
        [
            "Identifier",
            "cancer related death",
            "any cause of  death",
            "Days to death or current survival status",
            "Composite Gleason Score",
            "Primary gleason",
            "Secondary Gleason",
        ],
        axis=1,
    )
    rename_cols = {
        "AGE at diagnosis": "Age at diagnosis",
        "PSA at diagnosis": "PSA (ng/ml)",
        "Clincial stage at diagnosis": "Clinical T stage",
        "PRIMARY_TREATMENT_TYPE": "Primary treatment type",
        "Grade Group": "Histological grade group",
        "COMORBIDITY if blank then this is 0": "Comorbidity",
    }
    X = X.rename(columns=rename_cols)
    X["Comorbidity"] = X["Comorbidity"].fillna(0)
    Y = dataset["cancer related death"]

    dataset = pd.concat([X, Y], axis=1)
    dataset = dataset.sample(n=1000, random_state=42)
    X = dataset.drop(["cancer related death"], axis=1)
    Y = dataset["cancer related death"]

    attribute_names = X.columns.tolist()

    target_name = Y.name

    return X, Y, None, attribute_names, target_name


def load_seer():
    project_dir = os.environ["PROJECT_DIR"]
    path_csv = os.path.join(
        project_dir, "private_data/SEER_Prostate_Cancer_v2_with_missing.csv"
    )

    # LOAD SEER
    print("=" * 100)
    print("PROCESSING SEER...")
    dataset = pd.read_csv(path_csv)
    print(f"[SEER] original dataframe shape: {dataset.shape}")

    X = dataset.drop(
        [
            "Censoring",
            "Days to death or current survival status",
            "any cause of  death",
            "Primary Gleason",
            "Secondary Gleason",
            "Composite Gleason",
            "Number of Cores Negative",
            "AJCC Stage",
        ],
        axis=1,
    )

    rename_cols = {
        "Age at Diagnosis": "Age at Diagnosis",
        "PSA Lab Value": "PSA (ng/ml)",
        "T Stage": "Clinical T stage",
        "Grade": "Histological grade group",
        "Number of Cores Positive": "Number of Cores Positive",
        "Number of Cores Examined": "Number of Cores Examined",
    }
    X = X.rename(columns=rename_cols)
    T = dataset["Days to death or current survival status"]

    remove_empty = T > 0
    X = X[remove_empty]
    T = T[remove_empty]

    X = X.dropna(axis=0)
    print(f"[SEER] dataframe shape after dropping NaNs: {X.shape}")

    class_0 = X[X["cancer related death"] == 0]
    class_1 = X[X["cancer related death"] == 1]

    min_count = min(class_0.shape[0], class_1.shape[0])

    subsample_class_0 = class_0.sample(min_count // 4, random_state=42)
    subsample_class_1 = class_1.sample(min_count, random_state=42)

    print(
        f"[SEER] minimum count of class 0 and 1: {min_count}, subsampling with {min_count} instances each."
    )

    balanced_dataset = pd.concat([subsample_class_0, subsample_class_1])

    X = balanced_dataset.drop(["cancer related death"], axis=1)
    Y = balanced_dataset["cancer related death"]

    attribute_names = X.columns.tolist()
    target_name = Y.name

    return X, Y, None, attribute_names, target_name


def load_maggic():
    project_dir = os.environ["PROJECT_DIR"]
    path_csv = os.path.join(project_dir, "private_data/Maggic.csv")
    # LOAD MAGGIC
    print("=" * 100)
    print("PROCESSING MAGGIC...")

    expected_columns = [
        "age",
        "bmi",
        "ef_quant",
        "sbp_combined",
        "dbp_combined",
        "hb_combined",
        "hf_duration",
        "creat_combined",
        "sodium_combined",
        "sobar_combined",
        "soboe_combined",
        "beta_blocker_combined",
        "acei_or_arb",
        "gender",
        "diabetes",
        "angina",
        "mi",
        "atrial_fib",
        "stroke",
        "copd",
        "htn",
        "rales",
        "ischaemic",
        "cabg",
        "pci",
        "lbbb",
        "oed",
        "current_smoker",
        "nyha",
        "death_all",
        "days_to_fu",
    ]

    dataset = load_dataset(path_csv, expected_columns)
    print(f"[MAGGIC] original dataframe shape: {dataset.shape}")

    X = dataset.drop(["death_all", "days_to_fu"], axis=1)
    Y = dataset[["death_all"]]

    dataset = pd.concat([X, Y], axis=1)
    dataset = dataset.sample(n=1000, random_state=42)
    X = dataset.drop(["death_all"], axis=1)
    Y = dataset["death_all"]

    attribute_names = X.columns.tolist()
    target_name = Y.name

    return X, Y, None, attribute_names, target_name
