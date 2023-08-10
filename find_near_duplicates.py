import os
import re
import pickle
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from constants import NADI_FILE_PATH
from Levenshtein import ratio
from scipy.sparse import vstack, csr_array

tqdm.pandas()


def compute_similarity(s1, s2, cutoff):
    """Compute the InDel similarity between two strings.
    InDel_distance is a version of Levenshtein distance that only allows insertion and deletion operations.
    InDel_distance = number of insertion/deletion operations to transform s1 into s2.
    InDel_similarity = 1 - (InDel_distance / (len(s1) + len (s2)))
    InDel_distance ∈ [0, len(s1) + len(s2)], InDel_similarity ∈ [0, 1].

    Args:
        s1: The first string.
        s2: The second string.
        cutoff: The minimum non-neglible similarity score.

    Returns:
        The similarity score between s1, and s2 if it's larger than cutoff, otherwise returns 0.
    """
    sim_score = ratio(s1, s2)
    return 0 if sim_score < cutoff else sim_score


def build_distance_matrix(text_samples, cutoff):
    """Build a upper triangular sparse matrix of similarity scores between the text samples of a corpus.

    Args:
        text_samples: A list of text samples
        cutoff: The minimum non-neglible similarity score.

    Returns:
        A sparse similarity matrix.
    """
    sim_matrix_rows = []
    for i in tqdm(range(len(text_samples) - 1)):
        text = text_samples[i]
        # Compute the similarity score between a sample and the following ones
        sim_row = [0 for _ in range(i)] + [
            compute_similarity(text, other_text, cutoff=cutoff)
            for other_text in text_samples[i + 1 :]
        ]
        sim_matrix_rows.append(csr_array(sim_row))

    # Vertically stack the sparse rows
    return vstack(sim_matrix_rows).tocsr()


def normalize_text_column(df, text_column):
    """Normalize Arabic text column of a dataframe.

    Args:
        df: A pandas dataframe of a dataset.
        text_column: The name of the column having the text.

    Returns:
        A dataframe with a normalize text column.
    """
    # Keep the original dataset
    df[f"orig_{text_column}"] = df[text_column]
    PUNCTUATION = ".#_\"'!…،:\\-)(*/%,=+؟?٠"
    df[text_column] = df[text_column].progress_apply(
        lambda t: re.sub(rf"[{PUNCTUATION}]", " ", t)
    )

    # Remove Roman letters
    df[text_column] = df[text_column].progress_apply(
        lambda t: re.sub(r"[a-zA-Z0-9éà]", " ", t)
    )

    # Normalize ههههه interjection
    df[text_column] = df[text_column].progress_apply(
        lambda t: re.sub(r"[ه]{4,}", "ههه", t)
    )

    # Remove diacritics and Arabic numerals
    DIACTRITICS_NUMERALS_AND_TATWEEL = r"[\u0617-\u061A\u064B-\u0652ـ١-٩]"
    df[text_column] = df[text_column].progress_apply(
        lambda t: re.sub(DIACTRITICS_NUMERALS_AND_TATWEEL, "", t)
    )

    # Normalize ALEF
    df[text_column] = df[text_column].progress_apply(lambda t: re.sub(r"[آأإ]", "ا", t))

    # Normalize consecutive whitespaces
    df[text_column] = df[text_column].progress_apply(
        lambda t: re.sub(r"\s{2,}", " ", t)
    )
    return df


def main():
    CUTOFF = 0.75
    TEXT_COLUMN = "text"
    OUTPUT_DIR = "output/NADI"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(NADI_FILE_PATH, sep="\t")

    # Normalize the Arabic text
    df = normalize_text_column(df, text_column=TEXT_COLUMN)

    # Compute the similarity matrix
    similarity_matrix = build_distance_matrix(df[TEXT_COLUMN], cutoff=CUTOFF)

    with open(str(Path(OUTPUT_DIR, "sim.pkl")), "wb") as f:
        pickle.dump(similarity_matrix, f)


if __name__ == "__main__":
    main()
