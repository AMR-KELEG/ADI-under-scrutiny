FILES=$(ls data/**/Subtask1/NADI202*_Subtask1_*.tsv)
OUTPUT_FILENAME="data/NADI2021_2023.tsv"

echo "id\ttext\tlabel" > $OUTPUT_FILENAME
for FILE in $FILES
do
    tail -n +2 $FILE >> $OUTPUT_FILENAME
done
