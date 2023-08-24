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

## Experiment #1 - Analyzing the errors of a MarBERT model
1. Create the data directory
```
mkdir -p data/
```

2. Download the datasets to the `data/` directory
- NADI 2023
- QADI: http://alt.qcri.org/resources/qadi/
** Note**: QADI's zip is password protected. The password can be found through the dataset's page.

3. Extract the zip files:
```
cd data && unzip NADI2023_Release_Train.zip && cd ..
cd data && unzip QADI_Corpus_r001.zip && cd ..
```

The expected structure is:
```
data
├── NADI2023_Release_Train
│   ├── NADI2023-README.txt
│   ├── NADI2023-Twitter-Corpus-License.txt
│   ├── Subtask1
│   │   ├── MADAR-2018.tsv
│   │   ├── NADI2020-TWT.tsv
│   │   ├── NADI2021-TWT.tsv
│   │   ├── NADI2023-ST1-Scorer.py
│   │   ├── NADI2023_Subtask1_DEV.tsv
│   │   ├── NADI2023_Subtask1_TRAIN.tsv
│   │   ├── subtask1_dev_GOLD.txt
│   │   └── ubc_subtask1_dev_1.txt.zip
│   ├── Subtask2
│   │   ├── NADI2023-ST2-Scorer.py
│   │   ├── subtask2_dev_GOLD.txt
│   │   ├── subtask2_dev.tsv
│   │   └── ubc_subtask2_dev_1.txt.zip
│   └── Subtask3
│       ├── NADI2023-ST3-Scorer.py
│       ├── subtask3_dev_GOLD.txt
│       ├── subtask3_dev.tsv
│       └── ubc_subtask3_dev_1.txt.zip
├── NADI2023_Release_Train.zip
├── QADI_Corpus
│   ├── dataset
│   │   ├── QADI_train_ids_AE.txt
│   │   ├── QADI_train_ids_BH.txt
│   │   ├── QADI_train_ids_DZ.txt
│   │   ├── QADI_train_ids_EG.txt
│   │   ├── QADI_train_ids_IQ.txt
│   │   ├── QADI_train_ids_JO.txt
│   │   ├── QADI_train_ids_KW.txt
│   │   ├── QADI_train_ids_LB.txt
│   │   ├── QADI_train_ids_LY.txt
│   │   ├── QADI_train_ids_MA.txt
│   │   ├── QADI_train_ids_OM.txt
│   │   ├── QADI_train_ids_PL.txt
│   │   ├── QADI_train_ids_QA.txt
│   │   ├── QADI_train_ids_SA.txt
│   │   ├── QADI_train_ids_SD.txt
│   │   ├── QADI_train_ids_SY.txt
│   │   ├── QADI_train_ids_TN.txt
│   │   └── QADI_train_ids_YE.txt
│   ├── README.md
│   └── testset
│       └── QADI_test.txt
└── QADI_Corpus_r001.zip
```

4. Unify the NADI datasets labels
```
python preprocess_NADI_QADI.py
```

5. Fine-tune MarBERT using NADI2023's training data
```
OUTPUT_DIR="models/basic_NADI2023"
mkdir -p "${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="0" python finetune-BERT.py train -label_smoothing_factor 0 -d data/preprocessed/NADI2023_train.tsv --dev data/preprocessed/NADI2023_dev.tsv -o "${OUTPUT_DIR}" 2>&1 | tee ~/outputfile.txt "${OUTPUT_DIR}/training_logs.txt"
```

6. Generate the predictions for NADI2021's training dataset and QADI's test dataset
```
MODEL_DIR="models/basic_NADI2023"

OUTPUT_DIR="${MODEL_DIR}/predictions/"
mkdir -p "${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="0" python finetune-BERT.py predict -d data/preprocessed/NADI2021_DA_train.tsv -p "${MODEL_DIR}/checkpoint-1689" -o "${OUTPUT_DIR}"/NADI2021_DA_train_pred.tsv
CUDA_VISIBLE_DEVICES="0" python finetune-BERT.py predict -d data/preprocessed/QADI.tsv -p "${MODEL_DIR}/checkpoint-1689" -o "${OUTPUT_DIR}"/QADI_pred.tsv
```

7. Generate the error files from QADI
```
python analyze_QADI.py
```
