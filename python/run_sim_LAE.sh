file=/home/matt/software/matttest/results/sim_gxys.txt
while read -r line
do
    python $MATTTEST/LAE.py -s -f $line -k fit
done < "$file"
