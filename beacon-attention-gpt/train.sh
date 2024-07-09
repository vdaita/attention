pip install transformers accelerate datasets tqdm wandb
wandb login
# Default transformer
python beacon_train.py -w 1

# Transformers with 
python beacon_train.py -e -m -n -w 2
python beacon_train.py -e -m -w 2
python beacon_train.py -m -w 2

python beacon_train.py -e -m -n -w 4
python beacon_train.py -e -m -w 4
python beacon_train.py -m -w 4

python beacon_train.py -e -m -n -w 8
python beacon_train.py -e -m -w 8
python beacon_train.py -m -w 8

python beacon_train.py -e -m -n -w 16
python beacon_train.py -e -m -w 16
python beacon_train.py -m -w 16