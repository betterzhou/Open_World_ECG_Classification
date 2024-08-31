# Open-World Electrocardiogram Classification via Domain Knowledge-Driven Contrastive Learning

## 1. Introduction
This repository contains code and the processed datasets for the paper "[Open-World Electrocardiogram Classification via Domain Knowledge-Driven Contrastive Learning](https://www.sciencedirect.com/science/article/pii/S0893608024004751)" (Neural Networks 2024).


## 2. Usage
### Requirements:
+ torch==1.10.1
+ python==3.7

See requirements.txt for details.

### Datasets:
The meta information on each ECG sample is in the label.csv file under the data folder.

For different variants of the dataset, e.g., CPSC18-L and CPSC18-S, we use the same data folder but different label.csv file.

The code assumes that the data has been augmented and saved into the data folder.

The provided data was augmented based on domain knowledge. For the augmentation strategy, please refer to our paper for details.

We use [Onedrive](https://1drv.ms/f/c/f8be98f7ec1588fa/EhIXx0LLh1pAgdgRW14va0oBl_6x52s9fhRFL5Vk4omxGA?e=Z4ozg3) to share the dataset (~10 GB).
Please contact shuang.zhou@connect.polyu.hk to obtain the passcode.


### Example:
Please modify the 'data_path' in the code to adapt to the path of your data folder.

+ python run_1_train.py --dataset 'CPSC18' --model 'resnet34' --batch_size 100 --lr 0.00005 --transform_type 'hardneg' --seed 1 
+ python run_1_train.py --dataset 'CPSC18' --model 'resnet34_LSTM' --batch_size 100 --lr 0.00005 --transform_type 'hardneg' --seed 1
+ python run_1_train.py --dataset 'PTB' --model 'resnet34' --batch_size 300 --lr 0.00005 --transform_type 'hardneg'  --seed 1
+ python run_1_train.py --dataset 'PTB' --model 'resnet34_LSTM' --batch_size 300 --lr 0.00005 --transform_type 'hardneg'  --seed 1
+ python run_1_train.py --dataset='Georgia' --model='resnet34' --batch_size=300 --lr=5e-05 --transform_type='hardneg' --seed=1
+ python run_1_train.py --dataset='Georgia' --model='resnet34_LSTM' --batch_size=300 --lr=5e-05 --transform_type='hardneg' --seed=1

After learning, run the following code to get the final results.
+ python run_2_results.py --dataset=CPSC18 --model=resnet34 --transform_type=hardneg --seed=1
+ python run_2_results.py --dataset=CPSC18 --model=resnet34_LSTM --transform_type=hardneg --seed=1

We implement InceptionNet based on the code from [TSAI](https://github.com/timeseriesAI/tsai). To maintain simplicity, we only release the code for ResNet and CRNN.

For research cooperation, please contact shuang.zhou@connect.polyu.hk

## 3. Citation
Please kindly cite the paper if you use the code or any resources in this repo:
```bib
@article{zhou2024openecg,
  title={Open-world electrocardiogram classification via domain knowledge-driven contrastive learning},
  author={Zhou, Shuang and Huang, Xiao and Liu, Ninghao and Zhang, Wen and Zhang, Yuan-Ting and Chung, Fu-Lai},
  journal={Neural Networks},
  volume={179},
  pages={106551},
  year={2024},
  publisher={Elsevier}
}
```
