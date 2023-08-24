import os
import re
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

from constants import DIALECTS


def remove_QADI_special_tokens(s):
    special_tokens = ["@USER", "EMOJI", "URL", "NUM"]
    for special_token in special_tokens:
        s = re.sub(special_token, "", s)
    s = re.sub("NEWLINE", "\n", s)
    return re.sub(r"\s{1,}", " ", s).strip()


if __name__ == "__main__":
    OUTPUT_DIR = "analysis_output/"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    QADI_df = pd.read_csv("models/basic_NADI2023/predictions/QADI_pred.tsv", sep="\t")
    # Generate the classification report
    report_df = pd.DataFrame(
        classification_report(
            y_true=QADI_df["label"],
            y_pred=QADI_df["prediction"],
            labels=DIALECTS,
            output_dict=True,
        )
    ).T
    for c in report_df.columns:
        if c == "support":
            report_df["support"] = report_df["support"].astype(int)
        else:
            report_df[c] = report_df[c].apply(lambda v: round(v, 2))

    print(
        report_df[["support", "precision", "recall", "f1-score"]].to_latex(index=True)
    )

    # Generate QADI's confusion matrix
    fig, ax = plt.subplots(figsize=(6.4, 5))
    cm = confusion_matrix(
        y_true=QADI_df["label"], y_pred=QADI_df["prediction"], labels=DIALECTS
    )
    disp = ConfusionMatrixDisplay(cm, display_labels=DIALECTS)
    disp.plot(cmap=plt.cm.Reds, xticks_rotation="vertical", colorbar=False, ax=ax)
    fig.savefig(str(Path(OUTPUT_DIR, "confusion_matrix_QADI.pdf")), bbox_inches="tight")

    # Generate False positive files
    QADI_df["clean_text"] = QADI_df["text"].apply(remove_QADI_special_tokens)
    FALSE_POSITIVES_DIR = Path(OUTPUT_DIR, "false_positives/")
    os.makedirs(FALSE_POSITIVES_DIR, exist_ok=True)

    number_sentences_per_country = {}
    for label in DIALECTS:
        country_errors_df = QADI_df[
            (QADI_df["prediction"] == label)
            & (QADI_df["prediction"] != QADI_df["label"])
        ]
        country_errors_df = country_errors_df.sample(frac=1, random_state=42)
        country_errors_df.to_excel(
            str(Path(FALSE_POSITIVES_DIR, f"{label}_gold.xlsx")), index=False
        )
        country_errors_df.to_csv(
            str(Path(FALSE_POSITIVES_DIR, f"{label}_gold.csv")), index=False
        )
        country_errors_df[["clean_text"]].to_excel(
            str(Path(FALSE_POSITIVES_DIR, f"{label}.xlsx")), index=False
        )
        country_errors_df[["clean_text"]].to_csv(
            str(Path(FALSE_POSITIVES_DIR, f"{label}.csv")), index=False
        )
        number_sentences_per_country[label] = country_errors_df.shape[0]

    print(
        "Average number of False positives per country:",
        pd.Series(number_sentences_per_country.values()).mean(),
    )
