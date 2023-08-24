import torch
import random
import argparse
import pandas as pd
from glob import glob
from pathlib import Path
import torch.nn as nn
from torch import tensor
from scipy.special import softmax
from transformers import Trainer, TrainingArguments
from transformers.integrations import TensorBoardCallback
from torch.utils.data import Dataset
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import (
    AutoTokenizer,
    BertPreTrainedModel,
    BertModel,
)

from constants import (
    DIALECTS,
    DIALECTS_INDEX_INVERTED_MAP,
    DIALECTS_INDEX_MAP,
    COUNTRY_TO_REGION,
    COUNTRIES_IN_SAME_REGION_int,
)

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)


def transform_input(tokenizer, filenames, alpha=0, beta=0):
    """Tokenize the input text and return the features in the HF dict format.

    Args:
        tokenizer: The model's tokenizer.
        filenames: A list of tsv files of NADI.

    Returns:
        Features in the form of a HF dict format
    """
    dfs = [pd.read_csv(filename, sep="\t") for filename in filenames]

    df = pd.concat(dfs)
    df["label_int"] = [DIALECTS_INDEX_MAP[dialect] for dialect in df["label"]]
    df["countries_in_same_region_int"] = df["label_int"].apply(
        lambda i: COUNTRIES_IN_SAME_REGION_int[i]
    )
    features_dict = tokenizer(
        df["text"].tolist(),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    n_dialects = len(DIALECTS)

    # Allow the dataset not to have a label
    if "label" in df.columns:
        # A 2D vector of hard labels
        if beta == 0 and alpha == 0:
            features_dict["labels"] = tensor(
                [
                    [i == DIALECTS_INDEX_MAP[dialect] for i in range(n_dialects)]
                    for dialect in df["label"].tolist()
                ],
                dtype=torch.float,
            )

        else:
            features_dict["labels"] = tensor(
                [
                    [
                        (
                            1
                            - ((n_dialects - 1) * alpha / n_dialects)
                            - (
                                (len(row["countries_in_same_region_int"]) - 1)
                                * beta
                                / len(row["countries_in_same_region_int"])
                                if len(row["countries_in_same_region_int"])
                                else 0
                            )
                        )
                        if i == row["label_int"]
                        else (
                            beta / len(row["countries_in_same_region_int"])
                            + alpha / n_dialects
                        )
                        if i in row["countries_in_same_region_int"]
                        else (alpha / n_dialects)
                        for i in range(n_dialects)
                    ]
                    for _, row in df.iterrows()
                ],
                dtype=torch.float,
            )

    return features_dict


class NADIDataset(Dataset):
    def __init__(self, tokenizer, dataset_file_path, alpha=0, beta=0):
        super(NADIDataset).__init__()
        dataset_filenames = [f for f in glob(dataset_file_path)]

        self.alpha = 0
        self.beta = 0

        self.features_dict = transform_input(
            tokenizer, dataset_filenames, alpha=alpha, beta=beta
        )
        self.input_keys = self.features_dict.keys()

    def __len__(self):
        return self.features_dict["labels"].shape[0]

    def __getitem__(self, idx):
        return {k: self.features_dict[k][idx, :] for k in self.input_keys}


class CustomBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss_function = nn.CrossEntropyLoss(reduction="mean")

        loss = loss_function(input=logits, target=labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def model_init(model_name):
    """This function is required for ensuring the reproducibility."""
    model = CustomBertForSequenceClassification.from_pretrained(
        model_name, num_labels=len(DIALECTS)
    )
    return model


def main():
    parser = argparse.ArgumentParser(
        "Train a classification model for Dialect Identification (DI)."
    )

    subparsers = parser.add_subparsers()
    training_subparser = subparsers.add_parser(
        "train",
        help="Train the DI model.",
    )
    training_subparser.set_defaults(mode="train")
    training_subparser.add_argument(
        "--train",
        "-d",
        required=True,
        help="The filename of the training dataset (allows for glob).",
    )
    training_subparser.add_argument(
        "--dev",
        required=False,
        help="The filename of the development dataset (allows for glob).",
    )
    training_subparser.add_argument(
        "-model_name",
        "-m",
        default="UBC-NLP/MARBERT",
        help="The model name.",
    )
    training_subparser.add_argument(
        "-label_smoothing_factor",
        "-alpha",
        default=0,
        help="The label smoothing factor in [0, 1].",
    )
    training_subparser.add_argument(
        "-o",
        required=True,
        help="The output directory.",
    )

    prediction_subparser = subparsers.add_parser(
        "predict",
        help="Generate predictions using a fine-tuned model model.",
    )
    prediction_subparser.set_defaults(mode="predict")
    prediction_subparser.add_argument(
        "-d",
        required=True,
        help="The path of the dataset.",
    )
    prediction_subparser.add_argument(
        "-model_name",
        "-m",
        default="UBC-NLP/MARBERT",
        help="The model name.",
    )
    prediction_subparser.add_argument(
        "-p",
        required=True,
        help="The trained model path.",
    )
    prediction_subparser.add_argument(
        "-o",
        required=True,
        help="The output filename.",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if args.mode == "train":
        # TODO: Update the training arguments
        NO_STEPS = 1000
        BATCH_SIZE = 32
        ALPHA = 0.3
        BETA = 0.3
        training_args = TrainingArguments(
            output_dir=args.o,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            eval_steps=NO_STEPS,
            load_best_model_at_end=True,
            # TODO: Avoid hard-coding this field
            metric_for_best_model="eval_NADI2023_dev_loss",
            greater_is_better=False,
            per_device_train_batch_size=BATCH_SIZE,
            seed=SEED,
            data_seed=SEED,
            label_smoothing_factor=float(args.label_smoothing_factor),
        )

        train_dataset = NADIDataset(tokenizer, args.train, alpha=ALPHA, beta=BETA)

        eval_dataset_filenames = glob(args.dev)
        eval_dataset = {
            Path(filename).name.split(".")[0]: NADIDataset(
                tokenizer, filename, alpha=ALPHA, beta=BETA
            )
            for filename in eval_dataset_filenames
        }

        trainer = Trainer(
            model_init=lambda: model_init(args.model_name),
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[TensorBoardCallback],
        )
        trainer.train()
    else:
        # Load from checkpoint
        model = CustomBertForSequenceClassification.from_pretrained(
            args.p, num_labels=len(DIALECTS)
        )
        test_dataset = NADIDataset(tokenizer, args.d)

        trainer = Trainer(model, args=None, train_dataset=None)
        predictions = trainer.predict(test_dataset)
        prediction_labels = predictions.predictions.argmax(1).tolist()
        prediction_probs = softmax(predictions.predictions, axis=1)
        prediction_probs = prediction_probs.max(axis=1).tolist()
        prediction_labels = predictions.predictions.argmax(1).tolist()
        prediction_strs = [DIALECTS_INDEX_INVERTED_MAP[l] for l in prediction_labels]

        df = pd.read_csv(args.d, sep="\t")
        df["prediction"] = prediction_strs
        df["prediction_macro"] = [COUNTRY_TO_REGION[pred] for pred in prediction_strs]
        df["prediction_prob"] = prediction_probs

        df.to_csv(args.o, index=False, sep="\t")


if __name__ == "__main__":
    main()
