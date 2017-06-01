filename="$1"
add_txt="$2"
newfile="${filename:0:(-4)}_$add_txt${filename:(-4)}"
if [ -f $newfile ]; then
	rm $newfile
fi
while read -r line; do
	echo "${line:0:(-5)}_$add_txt${line:(-5)}" >> $newfile
done < "$filename"
