import re
import pandas as pd
import matplotlib.pyplot as plt

DIALECTS = [
    "Algeria",
    "Libya",
    "Morocco",
    "Tunisia",

    "Egypt",
    "Sudan",

    "Bahrain",
    "Iraq",
    "Kuwait",
    "Oman",
    "Qatar",
    "Saudi_Arabia",
    "UAE",

    "Jordan",
    "Lebanon",
    "Palestine",
    "Syria",

    "Yemen",
]

W = 6.3 / 2
H = 2


def generate_country_error_distributions(label, ax):
    """Generates a bar plot of the error distribution for a given country.

    Args:
        label: The country to plot the error distribution for.
        ax: The axis to plot on.
    """
    annotations_df = pd.read_csv(f"../data/annotations/{label}.tsv", sep="\t")
    w = W / (18 + 19 / 2)
    x_cur = w / 2
    x_ticks = []

    for dialect in DIALECTS:
        # 18 bars + 19 space
        n_samples = (annotations_df["original_label"] == dialect).sum()
        n_yes_samples = (
            (annotations_df["original_label"] == dialect)
            & (annotations_df["decision"] == "y")
        ).sum()
        ax.bar(
            x=x_cur,
            bottom=n_yes_samples,
            height=n_samples - n_yes_samples,
            width=w,
            color="#92C5DE",
            align="edge",
            label="✓ Correct FP" if dialect == "Algeria" else "",
        )
        ax.bar(
            x=x_cur,
            height=n_yes_samples,
            color="#CA0020",
            width=w,
            align="edge",
            label="✗ Incorrect FP" if dialect == "Algeria" else "",
        )

        x_ticks.append(x_cur + w / 2)
        x_cur += 1.5 * w
    ax.set_xticks(
        ticks=x_ticks,
        labels=[re.sub("_", " ", d) for d in DIALECTS],
        rotation=90,
        fontsize=8,
    )
    for i in [4, 6, 13, 17]:
        ax.plot(
            [i * 1.5 * w + w / 8, i * 1.5 * w + w / 8],
            [0, 50],
            "k--",
            alpha=0.5,
            linewidth=0.5,
        )
    #     ax.ylim(0, 50)
    #     ax.set_yticks(fontsize=8)
    ax.set_title(f"Prediction: {re.sub('_', ' ', label)}", fontsize=10)
    ax.legend(fontsize=6, frameon=False, loc="upper center")
    ax.set_xlabel(
        "Original Label", fontsize=8,
    )


def generate_error_bars():
    """Generates a bar plot of the error distribution for all countries.

    Returns:
        The figure containing the bar plot.
    """
    figure, axes = plt.subplots(nrows=4, ncols=2, sharey=True, figsize=(2 * W, 4 * H))
    for dialect, ax in zip(
        ["Algeria", "Egypt", "Lebanon", "Palestine", "Saudi_Arabia", "Sudan", "Syria"],
        [a for a_l in axes for a in a_l],
    ):
        generate_country_error_distributions(dialect, ax)
    figure.tight_layout()
    figure.delaxes(axes[-1][-1])
    return figure


if __name__ == "__main__":
    figure = generate_error_bars()
    figure.savefig("fig5_errors_distribtion.pdf", bbox_inches="tight")
