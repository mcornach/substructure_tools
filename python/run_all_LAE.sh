#file=/home/matt/software/matttest/data/pix_source_models.txt
file=/home/matt/software/matttest/results/sim_gxys/sim_gxys.txt
while read -r line
do
    python $MATTTEST/LAE.py -s -f $line
done < "$file"