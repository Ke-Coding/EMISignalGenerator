#!/bin/bash

# 1 -> pink noise
# 2 -> brown
# 3 -> laplace
# 4 -> uniformrv
# 5 -> exponentialrv

USAGE_STRING="Usage: bash $0 [length] [pink] [brown] [laplace] [uniform] [exponential] [fout]"
# OUTPUT_FILE="../data.txt"
OUTPUT_FILE=$7

if [ -z "$1" ]; then echo $USAGE_STRING && exit 0; else SAMPLE_LENGTH=$1; fi
if [ -z "$2" ]; then pink_count=0; else pink_count=$2; fi
if [ -z "$3" ]; then brown_count=0; else brown_count=$3; fi
if [ -z "$4" ]; then laplace_count=0; else laplace_count=$4; fi
if [ -z "$5" ]; then uniform_count=0; else uniform_count=$5; fi
if [ -z "$6" ]; then exponential_count=0; else exponential_count=$6; fi

# $1: the command to execute
# $2: the label for the command
# $3: the number of samples to be generated
function execute() {
    for i in $(seq 1 $3)
    do
        $1 $i >> $OUTPUT_FILE
        echo $2 >> $OUTPUT_FILE
    done
}

echo 'Compiling signal generators with gcc...'
mkdir build 2>/dev/null
cd build
/mnt/lustre/share/cmake3.8/bin/cmake ../src
make
cd ../dist

echo "output file is at: $OUTPUT_FILE"

rm "$OUTPUT_FILE" 2>/dev/null

echo "Generating $pink_count sample(s) with pinkgen..."
execute "./pinkgen 10 $SAMPLE_LENGTH 1.0" 1 $pink_count

echo "Generating $brown_count sample(s) with brownrv..."
execute "./brownrv $SAMPLE_LENGTH 0.0 1.0" 2 $brown_count

echo "Generating $laplace_count sample(s) with laplace..."
execute "./laplace $SAMPLE_LENGTH 0.0 1.0" 3 $laplace_count

echo "Generating $uniform_count sample(s) with uniformrv..."
execute "./uniformrv $SAMPLE_LENGTH -1.0 1.0" 4 $uniform_count

echo "Generating $exponential_count sample(s) with exponentialrv..."
execute "./exponentialrv $SAMPLE_LENGTH 1.0" 5 $exponential_count

wc -l "$OUTPUT_FILE"