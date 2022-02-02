
# extract features
# 1 train 2 test 3logfile

dirname $0

python fextractor.py $1 -mode train
python fextractor.py $2 -mode test

#genlist
python gen-list.py ./options-kNN.txt $1 $2
# compile attack
#g++ flearner-head.cpp -o flearner-head

# run attack
./flearner-head ./options-kNN.txt $1 $2 >> $3


