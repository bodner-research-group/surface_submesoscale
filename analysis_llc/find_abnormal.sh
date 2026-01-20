#!/bin/bash

DIR="."
echo "Finding abnormal files in $DIR"

# Find the most common file size
COMMON_SIZE=$(ls -l wb_eddy_*.nc | awk '{print $5}' | sort | uniq -c | sort -nr | head -1 | awk '{print $2}')

echo "Most common file size: $COMMON_SIZE bytes"
echo "Abnormal files:"

ls -l wb_eddy_*.nc | awk -v s="$COMMON_SIZE" '$5 != s {print $0}'

#!/bin/bash

DIR="."
COMMON_SIZE=$(ls -l wb_eddy_*.nc | awk '{print $5}' | sort | uniq -c | sort -nr | head -1 | awk '{print $2}')

echo "Deleting files not equal to $COMMON_SIZE bytes"

ls -l wb_eddy_*.nc | awk -v s="$COMMON_SIZE" '$5 != s {print $9}' | xargs -r rm -v


