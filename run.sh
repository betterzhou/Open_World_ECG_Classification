python run_1_train.py --dataset 'CPSC18' --model 'resnet34' --batch_size 100 --lr 0.00005 --transform_type 'hardneg' --seed 1
python run_1_train.py --dataset 'CPSC18' --model 'resnet34_LSTM' --batch_size 100 --lr 0.00005 --transform_type 'hardneg' --seed 1
python run_1_train.py --dataset 'PTB' --model 'resnet34' --batch_size 300 --lr 0.00005 --transform_type 'hardneg'  --seed 1
python run_1_train.py --dataset 'PTB' --model 'resnet34_LSTM' --batch_size 300 --lr 0.00005 --transform_type 'hardneg'  --seed 1
python run_1_train.py --dataset='Georgia' --model='resnet34' --batch_size=300 --lr=5e-05 --transform_type='hardneg' --seed=1
python run_1_train.py --dataset='Georgia' --model='resnet34_LSTM' --batch_size=300 --lr=5e-05 --transform_type='hardneg' --seed=1





