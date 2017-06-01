file=/home/matt/software/matttest/data/pix_source_models_tim10.txt
while read -r line
do
    [[ "$line" =~ ^#.*$ ]] && continue
    python $MATTTEST/LAE.py -f $line -k tim10
done < "$file"
