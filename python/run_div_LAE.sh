file=/home/matt/software/matttest/data/pix_source_models_div10.txt
while read -r line
do
    [[ "$line" =~ ^#.*$ ]] && continue
    python $MATTTEST/LAE.py -f $line -k div10
done < "$file"
