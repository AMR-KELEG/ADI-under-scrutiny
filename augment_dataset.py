import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from constants import COUNTRY_TO_REGION

tqdm.pandas()


def augment_df(df):
    # Rename the columns
    TEXT_COLUMNS = ["sent", "#2_content", "#2 tweet_content", "#2_tweet"]
    LABEL_COLUMNS = ["#3_label", "#3 country_label", "#3_country_label", "country"]

    TEXT_COLUMN = [c for c in df.columns if c in TEXT_COLUMNS][0]
    LABEL_COLUMN = [c for c in df.columns if c in LABEL_COLUMNS][0]
    df["text"] = df[TEXT_COLUMN]
    df["label"] = df[LABEL_COLUMN]
    df["label"] = df["label"].apply(lambda s: s.capitalize())
    df["label"] = df["label"].apply(
        lambda s: "Saudi_Arabia"
        if s in ["Ksa", "Saudi_arabia"]
        else "UAE"
        if s == "Uae"
        else s
    )

    # Map the label to Macro-label
    df["macro_label"] = df["label"].progress_apply(lambda l: COUNTRY_TO_REGION[l])

    OUTPUT_COLUMNS = ["text", "label", "macro_label"]

    ID_COLUMNS = ["#1_tweetid", "#1 tweet_ID", "#1_id", "sentID.BTEC"]
    if any([c in df.columns for c in ID_COLUMNS]):
        ID_COLUMN = [c for c in df.columns if c in ID_COLUMNS][0]
        df["ID"] = df[ID_COLUMN]
        OUTPUT_COLUMNS = ["ID"] + OUTPUT_COLUMNS

    MICROLABEL_COLUMNS = ["city", "#4_province_label", "#4 province_label"]
    if any([c in MICROLABEL_COLUMNS for c in df.columns]):
        MICROLABEL_COLUMN = [c for c in df.columns if c in MICROLABEL_COLUMNS][0]
        df["micro_label"] = df[MICROLABEL_COLUMN]
        OUTPUT_COLUMNS.append("micro_label")

    return df[OUTPUT_COLUMNS]


if __name__ == "__main__":
    NADI2023_BASE_DIR = "data/NADI2023_Release_Train/Subtask1/"
    OUTPUT_DIR = "data/NADI_datasets/"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    NADI_dfs = []
    for split in ["TRAIN", "DEV"]:
        df = pd.read_csv(
            str(Path(NADI2023_BASE_DIR, f"NADI2023_Subtask1_{split}.tsv")),
            sep="\t",
        )
        augmented_df = augment_df(df)
        NADI_dfs.append(augmented_df)
        augmented_df.to_csv(
            str(Path(OUTPUT_DIR, f"NADI2023_{split.lower()}.tsv")),
            sep="\t",
            index=False,
        )

    NADI2022_BASE_DIR = "data/NADI2022-Train/Subtask1/"
    for split in ["TRAIN", "DEV"]:
        df = pd.read_csv(
            str(Path(NADI2022_BASE_DIR, f"NADI2022_Subtask1_{split}.tsv")),
            sep="\t",
        )
        augmented_df = augment_df(df)
        NADI_dfs.append(augmented_df)
        augmented_df.to_csv(
            str(Path(OUTPUT_DIR, f"NADI2021_{split.lower()}.tsv")),
            sep="\t",
            index=False,
        )

    NADI_2021_2023_df = pd.concat(NADI_dfs)
    NADI_2021_2023_df.to_csv(
        str(Path(OUTPUT_DIR, f"NADI2021_2023.tsv")), sep="\t", index=False
    )
