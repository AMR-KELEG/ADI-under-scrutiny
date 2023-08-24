import os
import re
import pandas as pd
from glob import glob
from pathlib import Path
from bs4 import BeautifulSoup
from collections import Counter
from camel_tools.utils.charmap import CharMapper
from camel_tools.utils.transliterate import Transliterator

from constants import MADAR_CITY_TO_COUNTRY, MultiDialect_COUNTRYCODE_TO_COUNTRY


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
    df[text_column] = df[text_column].apply(
        lambda t: re.sub(rf"[{PUNCTUATION}]", " ", t)
    )

    # Remove Roman letters
    df[text_column] = df[text_column].apply(lambda t: re.sub(r"[a-zA-Z0-9éà]", " ", t))

    # Normalize ههههه interjection
    df[text_column] = df[text_column].apply(lambda t: re.sub(r"[ه]{4,}", "ههه", t))

    # Remove diacritics and Arabic numerals
    DIACTRITICS_NUMERALS_AND_TATWEEL = r"[\u0617-\u061A\u064B-\u0652ـ١-٩]"
    df[text_column] = df[text_column].apply(
        lambda t: re.sub(DIACTRITICS_NUMERALS_AND_TATWEEL, "", t)
    )

    # Normalize ALEF
    df[text_column] = df[text_column].apply(lambda t: re.sub(r"[آأإ]", "ا", t))

    # Normalize consecutive whitespaces
    df[text_column] = df[text_column].apply(lambda t: re.sub(r"\s{2,}", " ", t))
    return df


def load_txt_file(filename):
    with open(filename, "r") as f:
        return [l.strip() for l in f]


# Multidialectal Parallel Corpus of Arabic
# https://nyuad.nyu.edu/en/research/faculty-labs-and-projects/computational-approaches-to-modeling-language-lab/resources.html
def build_MPCA_corpus(
    dialects,
    BASEDIR="data/MultiDial-Public-Version-2014/Multidialectal_Parallel_Corpus_of_Arabic/",
):
    # Load the lines from the txt files
    dialectal_lines = {
        dialect: load_txt_file(str(Path(BASEDIR, f"{dialect}.txt")))
        for dialect in dialects
    }
    parallel_sentences_df = pd.DataFrame(dialectal_lines)

    for dialect in dialects:
        parallel_sentences_df = normalize_text_column(parallel_sentences_df, dialect)

    # Transform the dataset into DI format
    TEXT_COLUMN_NAME = "text"
    di_df = pd.DataFrame(
        [
            {TEXT_COLUMN_NAME: sentence, "dialect": dialect}
            for dialect in dialects
            for sentence in dialectal_lines[dialect]
        ]
    )
    di_df = normalize_text_column(di_df, TEXT_COLUMN_NAME)

    di_df["country"] = di_df["dialect"].apply(
        lambda dialect: MultiDialect_COUNTRYCODE_TO_COUNTRY[dialect]
    )

    return parallel_sentences_df, di_df


# https://sourceforge.net/projects/padic/
def build_PADIC(dialects, BASEDIR="data/"):
    filepath = Path(BASEDIR, "PADIC.xml")
    with open(filepath, "r") as f:
        soup = BeautifulSoup("\n".join(f.readlines()), "html.parser").find("padic")

    bw2ar = CharMapper.builtin_mapper("bw2ar")
    bw2ar_translit = Transliterator(bw2ar)

    dialectal_lines = {}
    for i in soup.findAll("sentence"):
        for c in i.findChildren():
            bw_text = c.text.strip()
            ar_text = bw2ar_translit.transliterate(
                bw2ar_translit.transliterate(bw_text), strip_markers=True
            )
            dialect = c.name
            if dialect not in dialectal_lines:
                dialectal_lines[dialect] = []
            dialectal_lines[dialect].append(ar_text)

    parallel_sentences_df = pd.DataFrame(dialectal_lines)

    TEXT_COLUMN_NAME = "text"
    di_df = pd.DataFrame(
        [
            {TEXT_COLUMN_NAME: sentence, "dialect": dialect}
            for dialect in dialects
            for sentence in dialectal_lines[dialect]
        ]
    )
    di_df = normalize_text_column(di_df, TEXT_COLUMN_NAME)

    PADIC_CITY_TO_COUNTRY = {
        "algiers": "Algeria",
        "annaba": "Algeria",
        "syrian": "Syria",
        "palestinian": "Palestine",
        "moroccan": "Morocco",
    }
    di_df["country"] = di_df["dialect"].apply(
        lambda dialect: PADIC_CITY_TO_COUNTRY[dialect]
    )

    return parallel_sentences_df, di_df


# Multi-Arabic Dialect Corpus (MADAR Corpus)
# https://nyuad.nyu.edu/en/research/faculty-labs-and-projects/computational-approaches-to-modeling-language-lab/resources.html
def build_MADAR(
    no_dialects,
    BASEDIR="data/MADAR-SHARED-TASK-final-release-25Jul2019/MADAR-Shared-Task-Subtask-1/",
):
    assert str(no_dialects) in ["6", "26"]

    TEXT_COLUMN_NAME = "text"
    di_dfs = [
        pd.read_csv(filename, sep="\t", names=[TEXT_COLUMN_NAME, "dialect"])
        for filename in glob(str(Path(BASEDIR, f"MADAR-Corpus-{no_dialects}-*.tsv")))
    ]
    di_df = pd.concat(di_dfs)

    di_df = normalize_text_column(di_df, TEXT_COLUMN_NAME)

    di_df = di_df[di_df["dialect"] != "MSA"]
    di_df["country"] = di_df["dialect"].apply(
        lambda dialect: MADAR_CITY_TO_COUNTRY[dialect]
    )

    return None, di_df


def estimate_metrics(df):
    N_SAMPLES = df.shape[0]
    sentences_counts = {
        text: count for text, count in Counter(df["text"]).most_common()
    }
    df["#_valid_dialects"] = df["text"].apply(lambda t: sentences_counts[t])
    maximal_accuracy = sum(df["#_valid_dialects"].apply(lambda n: 1 / n))
    maximal_accuracy /= N_SAMPLES

    COUNTRY_LABELS = sorted(df["country"].unique().tolist())
    total_labels = len(COUNTRY_LABELS)

    print("# samples:", N_SAMPLES)
    for n_labels in range(2, total_labels + 1):
        print(
            f"Perc{n_labels}:",
            f'{round(100 * (df["#_valid_dialects"] == n_labels).sum() / N_SAMPLES, 1)}%',
        )

    print(
        f"sum(Perc_n):",
        f'{round(100 * (df["#_valid_dialects"] !=1).sum() / N_SAMPLES, 1)}%',
    )
    print("Expected Maximal Accuracy:", f"{round(100 * maximal_accuracy, 1)}%")


def main():
    OUTPUT_DIR = "data/preprocessed/"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    datasets = ["MPCA", "PADIC"]
    dataset_building_funcions = [build_MPCA_corpus, build_PADIC]
    dialects_lists = [
        ["EG", "TN", "SY", "JO", "PA"],
        ["algiers", "annaba", "syrian", "palestinian", "moroccan"],
    ]

    for dataset, dataset_building_funcion, dialects_list in zip(
        datasets, dataset_building_funcions, dialects_lists
    ):
        parallel_sentences_df, di_df = dataset_building_funcion(dialects=dialects_list)

        di_df.drop_duplicates(subset=["text", "country"], inplace=True)
        print(
            dataset,
        )
        estimate_metrics(di_df)

        parallel_sentences_df.to_csv(
            str(Path(OUTPUT_DIR, f"{dataset}_parallel.tsv")), sep="\t"
        )
        di_df.to_csv(str(Path(OUTPUT_DIR, f"{dataset}_single.tsv")), sep="\t")

        print("\n\n")

    for no_labels in [6, 26]:
        parallel_sentences_df, di_df = build_MADAR(no_labels)
        di_df.drop_duplicates(subset=["text", "country"], inplace=True)
        print(
            f"MADAR{no_labels}",
        )
        estimate_metrics(di_df)

        di_df.to_csv(str(Path(OUTPUT_DIR, f"MADAR_{no_labels}_single.tsv")), sep="\t")
        print("\n\n")


if __name__ == "__main__":
    main()
