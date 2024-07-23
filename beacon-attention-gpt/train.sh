pip install transformers accelerate datasets tqdm wandb &
wait
wandb login &
wait
# Default transformer
# python3 beacon_train.py -w 1 &
wait

python3 beacon_train.py -e -m -w 4 &
wait
python3 beacon_train.py -m -w 4 &
wait

python3 beacon_train.py -e -m -w 8 &
wait
python3 beacon_train.py -m -w 8 &
wait 

python3 beacon_train.py -e -m -w 16 &
wait
python3 beacon_train.py -m -w 16 &
wait
