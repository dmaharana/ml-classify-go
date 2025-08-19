filename="$1"
# Get total lines (including header)
total_lines=$(wc -l < $filename)
# Exclude header line for calculation
data_lines=$((total_lines - 1))
# Calculate lines for 80% split (round down)
split_line=$((data_lines * 80 / 100))

# Get header
head -n 1 ${filename} > file_part_80.csv
head -n 1 ${filename} > file_part_20.csv

# Get 80% data
tail -n +2 ${filename} | head -n $split_line >> file_part_80.csv
# Get remaining 20%
tail -n +$((split_line + 2)) ${filename} >> file_part_20.csv
