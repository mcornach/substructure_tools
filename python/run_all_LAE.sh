file=/home/matt/software/matttest/data/pix_source_models.txt
while read -r line
do
    python $MATTTEST/LAE.py -aP -f $line
done < "$file"
