# Arabic Dialect Identification under Scrutiny

* Download NADI2022 (training data is a subset from NADI2021) and NADI2023 datasets to `data/`.
* Extract both zip files into `data/`
* Unify the column names of both datasets by running `python augment_dataset.py`

## Error analysis

* Fine-tune MarBERT using NADI2023's training data
```
OUTPUT_DIR="models/basic_NADI2023"
mkdir -p "${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="0" python finetune-BERT.py train -label_smoothing_factor 0 -d data/NADI_datasets/NADI2023_train.tsv --dev data/NADI_datasets/NADI2023_dev.tsv -o "${OUTPUT_DIR}" 2>&1 | tee ~/outputfile.txt "${OUTPUT_DIR}/training_logs.txt"
```

* Generate the predictions for NADI2021's training and development data
```
MODEL_DIR="models/basic_NADI2023"

OUTPUT_DIR="${MODEL_DIR}/predictions/"
mkdir -p "${OUTPUT_DIR}"

for SPLIT in "train" "dev"
do
    CUDA_VISIBLE_DEVICES="0" python finetune-BERT.py predict -d data/NADI_datasets/NADI2021_"${SPLIT}".tsv -p "${MODEL_DIR}/checkpoint-1689" -o "${OUTPUT_DIR}"/NADI2021_"${SPLIT}"_pred.tsv 
done
```
