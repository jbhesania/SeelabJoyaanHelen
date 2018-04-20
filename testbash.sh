#!/bin/bash

#python testprogram.py <<joyaan
#"m"
#joyaan

echo "Starting file 1"
python Pow2nmRepeatedRetraining.py 1 50 >> testdata/pow2nmRR1.txt 2>&1
echo "Starting file 2"
python Pow2nmRepeatedRetraining.py 1 50 >> testdata/pow2nmRR2.txt 2>&1

echo "Starting file 3"
python Pow2nmRepeatedRetraining.py 64 50 >> testdata/pow2nmRR3.txt 2>&1

echo "Starting file 4"
python Pow2nmRepeatedRetraining.py 64 50 >> testdata/pow2nmRR4.txt 2>&1



echo "Starting file 5"
python Pow2nm.py 1 50 >> testdata/pow2nm1.txt 2>&1

echo "Starting file 6"
python Pow2nm.py 1 50 >> testdata/pow2nm2.txt 2>&1

echo "Starting file 7"
python Pow2nm.py 64 50 >> testdata/pow2nm3.txt 2>&1

echo "Starting file 8"
python Pow2nm.py 64 50 >> testdata/pow2nm4.txt 2>&1



echo "Starting file 9"
python Pow2Original.py 1 50 >> testdata/pow2og1.txt 2>&1

echo "Starting file 10"
python Pow2Original.py 1 50 >> testdata/pow2og2.txt 2>&1

echo "Starting file 11"
python Pow2Original.py 64 50 >> testdata/pow2og3.txt 2>&1

echo "Starting file 12"
python Pow2Original.py 64 50 >> testdata/pow2og4.txt 2>&1



echo "Starting file 13"
python RepeatedRetraining.py 1 50 >> testdata/fpRR1.txt 2>&1

echo "Starting file 14"
python RepeatedRetraining.py 1 50 >> testdata/fpRR2.txt 2>&1

echo "Starting file 15"
python RepeatedRetraining.py 64 50 >> testdata/fpRR3.txt 2>&1

echo "Starting file 16"
python RepeatedRetraining.py 64 50 >> testdata/fpRR4.txt 2>&1



echo "Starting file 17"
python SubclassesandRetraining.py 1 50 >> testdata/fpog1.txt 2>&1

echo "Starting file 18"
python SubclassesandRetraining.py 1 50 >> testdata/fpog2.txt 2>&1

echo "Starting file 19"
python SubclassesandRetraining.py 64 50 >> testdata/fpog3.txt 2>&1

echo "Starting file 20"
python SubclassesandRetraining.py 64 50 >> testdata/fpog4.txt 2>&1

