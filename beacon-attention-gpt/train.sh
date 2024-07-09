pip install transformers accelerate datasets tqdm wandb
wandb login
# Default transformer
python beacon_train.py -w 1

# Transformers with 
python beacon_train.py -e -m -n 2
python beacon_train.py -e -m 2
python beacon_train.py -m 2

python beacon_train.py -e -m -n 4
python beacon_train.py -e -m 4
python beacon_train.py -m 4

python beacon_train.py -e -m -n 8
python beacon_train.py -e -m 8
python beacon_train.py -m 8

python beacon_train.py -e -m -n 16
python beacon_train.py -e -m 16
python beacon_train.py -m 16