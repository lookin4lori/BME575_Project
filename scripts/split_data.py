"""Create reproducible APTOS train, validation, and test splits."""

from py_compile import main

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split #python's ML library for splitting data
from config import ( # config is the file where we define all variables.. this is pulling from that
    BINARY_LABEL_COLUMN,
    EXPECTED_CLASS_COUNTS,
    EXPECTED_TOTAL_IMAGES,
    FILENAME_COLUMN,
    ID_COLUMN,
    IMAGE_DIR,
    IMAGE_EXTENSION,
    LABELS_CSV,
    ORIGINAL_LABEL_COLUMN,
    RANDOM_SEED,
    SPLIT_COLUMN,
    SPLIT_DIR,
    STRATIFY_COLUMN,
    TEST_SIZE,
    TRAIN_SIZE,
    VAL_SIZE,
)
def load_and_verify() -> pd.DataFrame:
    if not LABELS_CSV.is_file():
        raise FileNotFoundError(f"Labels CSV not found: {LABELS_CSV}")
    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")

    df=pd.read_csv(LABELS_CSV)

    required_columns = {ID_COLUMN, ORIGINAL_LABEL_COLUMN}
    missing_columns = required_columns - set(df.columns) ## turns columns from df into a set {} and then seeing if any missing

    if missing_columns:
        raise ValueError( "file missing required columns" f"{sorted(missing_columns)}"

        )
    if df[ID_COLUMN].isna().any(): #checks NaN in DF
        raise ValueError("ID column contains NaN values")

    if df[ORIGINAL_LABEL_COLUMN].isna().any():
        raise ValueError("Missing diagnosis labels were found")
    if not df[ID_COLUMN].is_unique:
        duplicate_ids=df.loc[df[ID_COLUMN].duplicated(keep=False), ID_COLUMN].tolist()
        raise ValueError(
            "Duplicate image IDs were found in train.csv. "
            f"Examples: {duplicate_ids[:10]}" )
    if len(df) != EXPECTED_TOTAL_IMAGES:
        raise ValueError(
        f"Expected {EXPECTED_TOTAL_IMAGES} metadata rows, "f"but found {len(df)}.")
        
    expected_labels=set(EXPECTED_CLASS_COUNTS)
    observed_labels = set(df[ORIGINAL_LABEL_COLUMN].unique())

    if observed_labels != expected_labels:
        raise ValueError(f"Expected labels {sorted(expected_labels)}, "f"but found {sorted(observed_labels)}.")

    observed_class_counts = (df[ORIGINAL_LABEL_COLUMN].value_counts().sort_index().to_dict())

    if observed_class_counts != EXPECTED_CLASS_COUNTS:
        raise ValueError(
            "The observed class counts do not match the "
            "APTOS counts reported in the paper.\n"
            f"Expected: {EXPECTED_CLASS_COUNTS}\n"
            f"Observed: {observed_class_counts}"
        )

    df[FILENAME_COLUMN] = (
        df[ID_COLUMN].astype(str) + IMAGE_EXTENSION #converts the ids to string (using .astype) and adds. png
    )

    df["_image_path"] = df[FILENAME_COLUMN].apply(lambda filename: IMAGE_DIR / filename) #converts to full paths

    missing_images = [ str(image_path)
        for image_path in df["_image_path"]
        if not image_path.is_file()
    ]

    if missing_images:
        raise FileNotFoundError(
            f"{len(missing_images)} labeled images are missing. "
            f"Examples: {missing_images[:10]}"
        )

    df[BINARY_LABEL_COLUMN] = (df[ORIGINAL_LABEL_COLUMN] > 0).astype(int)
    return df

def create_strat_splits( df: pd.DataFrame,) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ##Create  70/10/20 stratified splits.

    total_fraction = TRAIN_SIZE + VAL_SIZE + TEST_SIZE

    if abs(total_fraction - 1.0) > 1e-9:
        raise ValueError(
            "TRAIN_SIZE, VAL_SIZE, and TEST_SIZE "
            f"must sum to 1.0, but they sum to {total_fraction}."
        )

    train_val_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_SEED,
        shuffle=True,
        stratify=df[STRATIFY_COLUMN],
    )

    relative_validation_size = VAL_SIZE / (TRAIN_SIZE + VAL_SIZE)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_validation_size,
        random_state=RANDOM_SEED,
        shuffle=True,
        stratify=train_val_df[STRATIFY_COLUMN],
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df[SPLIT_COLUMN] = "train"
    val_df[SPLIT_COLUMN] = "val"
    test_df[SPLIT_COLUMN] = "test"

    return train_df, val_df, test_df

def assert_no_id_overlap( first_df: pd.DataFrame, second_df: pd.DataFrame, first_name: str, second_name: str,) -> None:
   ## to confirm that two splits share no image IDs.
    first_ids = set(first_df[ID_COLUMN])
    second_ids = set(second_df[ID_COLUMN])
    overlap = first_ids.intersection(second_ids)
    if overlap:
        examples = sorted(overlap)[:10]
        raise AssertionError(
            f"Image leakage detected between {first_name} "
            f"and {second_name}. "
            f"Found {len(overlap)} shared image IDs. "
            f"Examples: {examples}"
        )
#make sure no three sets share the same image
def run_leakage_checks(original_df: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame,test_df: pd.DataFrame,) -> None:
    assert_no_id_overlap( train_df, val_df, "train", "validation",)
    assert_no_id_overlap(train_df, test_df,"train","test",)
    assert_no_id_overlap(val_df, test_df, "validation", "test",)

    combined_df = pd.concat( [train_df, val_df, test_df], ignore_index=True,) #to cofnirm nothng disappeared or added when we split
    if len(combined_df) != len(original_df):
        raise AssertionError("The combined split size does not match the original dataset size." )

    if combined_df[ID_COLUMN].nunique() != len(original_df):
        raise AssertionError("An image ID is missing or repeated across the splits.")

    if set(combined_df[ID_COLUMN]) != set(original_df[ID_COLUMN]):
        raise AssertionError(
            "The split files do not contain exactly the original image IDs."
        )

    print("\nLeakage checks passed:")
    print("- No image ID occurs in two splits.")
    print("- Every original image occurs exactly once.")

    ###
def make_distribution_table(dataframe: pd.DataFrame, split_name: str, label_column: str,) -> pd.DataFrame:
    ##Create class counts and percentages for one split."""

    distribution = (dataframe[label_column].value_counts().sort_index().rename_axis("class").reset_index(name="count"))

    distribution["percentage"] = (
        100.0 * distribution["count"] / len(dataframe)).round(2)

    distribution.insert(0, "split", split_name)
    distribution.insert(1, "label_type", label_column)

    return distribution

def create_all_distribution_tables(train_df: pd.DataFrame, val_df: pd.DataFrame,test_df: pd.DataFrame,) -> pd.DataFrame:
    ##THIS IS TO: Create original and binary distribution tables.
    #creates 6 tables: 2 for each split (original and binary)
    split_dataframes = {"train": train_df, "val": val_df, "test": test_df,}
    tables = []

    for split_name, split_df in split_dataframes.items():
        tables.append(
            make_distribution_table(
                dataframe=split_df,
                split_name=split_name,
                label_column=ORIGINAL_LABEL_COLUMN,
            )
        )

        tables.append(
            make_distribution_table(
                dataframe=split_df,
                split_name=split_name,
                label_column=BINARY_LABEL_COLUMN,
            )
        )

    return pd.concat(tables, ignore_index=True)

def print_distribution_tables(distribution_df: pd.DataFrame,) -> None:
    #Print count and percentage tables.... helpful when moving forward

    for label_type in [ORIGINAL_LABEL_COLUMN, BINARY_LABEL_COLUMN,]:
        selected = distribution_df[ distribution_df["label_type"] == label_type]

        count_table = selected.pivot(
            index="class",
            columns="split",
            values="count",
        )

        percentage_table = selected.pivot(
            index="class",
            columns="split",
            values="percentage",
        )

        print(f"\n{label_type}: counts")
        print(count_table.to_string())

        print(f"\n{label_type}: percentages")
        print(percentage_table.to_string())
#NOW SAVE ALL THE SPLIT DF
def save_outputs(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, distribution_df: pd.DataFrame,) -> None:
    """Save split manifests and the class-distribution table."""

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    output_columns = [
        ID_COLUMN,
        FILENAME_COLUMN,
        ORIGINAL_LABEL_COLUMN,
        BINARY_LABEL_COLUMN,
        SPLIT_COLUMN,
    ]

    train_output = (
        train_df[output_columns]
        .sort_values(ID_COLUMN)
        .reset_index(drop=True)
    )

    val_output = (
        val_df[output_columns]
        .sort_values(ID_COLUMN)
        .reset_index(drop=True)
    )

    test_output = (
        test_df[output_columns]
        .sort_values(ID_COLUMN)
        .reset_index(drop=True)
    )

    all_output = pd.concat([train_output, val_output, test_output], ignore_index=True,)

    train_output.to_csv(
        SPLIT_DIR / "train.csv",
        index=False,
    )

    val_output.to_csv(
        SPLIT_DIR / "val.csv",
        index=False,
    )

    test_output.to_csv(
        SPLIT_DIR / "test.csv",
        index=False,
    )

    all_output.to_csv(
        SPLIT_DIR / "all_splits.csv",
        index=False,
    )

    distribution_df.to_csv(
        SPLIT_DIR / "class_distribution.csv",
        index=False,
    )

    print(f"\nSaved output files to: {SPLIT_DIR}") #Split manifest files n class in csv file

    ##THIS IS IS THE BIG FUNCTION THAT CALLS ALL THE PREVIOUS ONES
def main() -> None:

    print("Loading and validating APTOS metadata...")

    df = load_and_verify()

    print("\nDataset validation passed.")
    print(f"Total images: {len(df)}")

    print("\nOriginal five-class counts:")
    print(
        df[ORIGINAL_LABEL_COLUMN]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nBinary counts:")
    print(
        df[BINARY_LABEL_COLUMN]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        f"\nCreating stratified splits "
        f"with random seed {RANDOM_SEED}..."
    )

    train_df, val_df, test_df = create_strat_splits(df)

    print("\nSplit sizes:")
    print(f"Train:      {len(train_df)}")
    print(f"Validation: {len(val_df)}")
    print(f"Test:       {len(test_df)}")
    print(
        "Total:      "
        f"{len(train_df) + len(val_df) + len(test_df)}"
    )

    run_leakage_checks(
        original_df=df,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )

    distribution_df = create_all_distribution_tables(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )

    print_distribution_tables(distribution_df)

    save_outputs(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        distribution_df=distribution_df,
    )

    print("\nData splitting completed successfully.")


if __name__ == "__main__":
    main()