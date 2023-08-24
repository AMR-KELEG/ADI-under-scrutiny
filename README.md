# Arabic Dialect Identification under Scrutiny

## Experiment #1 - Estimating the Maximal Accuracy of Parallel Corpora
1. Create a directory for the data:
```
mkdir -p data/
```

2. Download the datasets to the `data/` directory:
- Multidialectal Parallel Corpus of Arabic: https://nyuad.nyu.edu/en/research/faculty-labs-and-projects/computational-approaches-to-modeling-language-lab/resources.html
- PADIC: https://sourceforge.net/projects/padic/
- Multi-Arabic Dialect Corpus (MADAR Corpus) - (Dialect Identification Shared Task Dataset (WANLP 2019) - Dataset and Code for MADAR Shared Task): https://nyuad.nyu.edu/en/research/faculty-labs-and-projects/computational-approaches-to-modeling-language-lab/resources.html

3. Extract the files in the `data/` directory. The expected structure is:
```
data
├── MADAR-SHARED-TASK-final-release-25Jul2019
│   ├── MADAR-DID-Scorer.py
│   ├── MADAR-Shared-Task-Subtask-1
│   │   ├── EXAMPLE.GOLD
│   │   ├── EXAMPLE.PRED
│   │   ├── MADAR-Corpus-26-dev.tsv
│   │   ├── MADAR-Corpus-26-test.tsv
│   │   ├── MADAR-Corpus-26-train.tsv
│   │   ├── MADAR-Corpus-6-dev.tsv
│   │   ├── MADAR-Corpus-6-train.tsv
│   │   └── MADAR-Corpus-Lexicon-License.txt
│   ├── MADAR-Shared-Task-Subtask-2
│   │   ├── EXAMPLE.GOLD
│   │   ├── EXAMPLE.PRED
│   │   ├── MADAR-Obtain-Tweets.py
│   │   ├── MADAR-Twitter-Corpus-License.txt
│   │   ├── MADAR-Twitter-Subtask-2.DEV.user-label.tsv
│   │   ├── MADAR-Twitter-Subtask-2.DEV.user-tweets-features.tsv
│   │   ├── MADAR-Twitter-Subtask-2.TEST.user-label.tsv
│   │   ├── MADAR-Twitter-Subtask-2.TEST.user-tweets-features.tsv
│   │   ├── MADAR-Twitter-Subtask-2.TRAIN.user-label.tsv
│   │   └── MADAR-Twitter-Subtask-2.TRAIN.user-tweets-features.tsv
│   ├── MADAR_SharedTask_Summary_Paper_WANLP_ACL_2019.pdf
│   └── README.txt
├── MADAR-SHARED-TASK-final-release-25Jul2019.zip
├── MultiDial-Public-Version-2014
│   ├── LICENSE.txt
│   ├── Multidialectal_Parallel_Corpus_of_Arabic
│   │   ├── EG.txt
│   │   ├── EN.txt
│   │   ├── JO.txt
│   │   ├── l.txt
│   │   ├── MSA.txt
│   │   ├── PA.txt
│   │   ├── SY.txt
│   │   └── TN.txt
│   ├── Multidialectal_Parallel_Corpus_of Arabic_Paper.pdf
│   └── README.txt
├── MultiDial-Public-Version-2014.zip
└── PADIC.xml
```

4. Run the Expected Maximal Accuracy estimation script:
```
python estimate_maximal_accuracy_parallel_corpora.py
```
**Note**: The script prints the metrics and saves intermediate data files to `data/preprocessed`.

## Error analysis
* Download NADI2022 (training data is a subset from NADI2021) and NADI2023 datasets to `data/`.
* Extract both zip files into `data/`
* Unify the column names of both datasets by running `python augment_dataset.py`

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

## Check duplicated samples in datasets (NADI, MADAR)
```
DATASET="NADI"
# Generate similairty matrix for a dataset (output/DATASET/sim.pkl)
python find_near_duplicates.py -d ${DATASET}

python extract_duplicates.py -sm output/${DATASET}/sim.pkl -d output/${DATASET}/${DATASET}.tsv -o output/${DATASET}/high_sim_th_1.0.tsv -th 1.0
```

### Links to datasets
- [MADAR corpus](https://camel.abudhabi.nyu.edu/madar-shared-task-2019/)