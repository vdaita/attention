pip install transformers accelerate datasets tqdm wandb &
wait
wandb login &
wait
# Default transformer
python beacon_train.py -w 1 &
wait

# Transformers with 
python beacon_train.py -e -m -n -w 2 &
wait
python beacon_train.py -e -m -w 2 &
wait
python beacon_train.py -m -w 2 &
wait

python beacon_train.py -e -m -n -w 4 &
wait
python beacon_train.py -e -m -w 4 &
wait
python beacon_train.py -m -w 4 &
wait

python beacon_train.py -e -m -n -w 8 &
wait
python beacon_train.py -e -m -w 8 &
wait
python beacon_train.py -m -w 8 &
wait

python beacon_train.py -e -m -n -w 16 &
wait
python beacon_train.py -e -m -w 16 &
wait
python beacon_train.py -m -w 16 &
wait