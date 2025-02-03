# PTB-DDI

![](https://github.com/drunkprogrammer/PTB-DDI/blob/main/Overall-Architecture.png)

## Contents

- [Overview](#Overview)
- [Config PTB-DDI Environment](#Config-PTB-DDI-Environment)
- [Train and Test on the BIOSNAP Dataset](#Train-and-Test-on-the-BIOSNAP-dataset)
- [Train and Test on the DrugBank Dataset](#Train-and-Test-on-the-DrugBank-dataset)
- [Notice](#Notice)

## Overview

PTB-DDI: Accurate and Simple Framework for Drug-Drug Interaction Prediction Based on Pre-trained Tokenizer and BiLSTM Model

## Config PTB-DDI Environment

```
conda create -n PTB-DDI
conda activate PTB-DDI
conda install python==3.10.0
conda install scikit-learn
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c conda-forge matplotlib==3.5.1
conda install -c conda-forge numpy==1.22.0
conda install -c conda-forge pandas==1.3.5 tqdm==4.62.3
conda install -c conda-forge transformers
conda install -c anaconda seaborn
pip install torch_geometric
pip install rdkit
pip install protobuf untangle deepchem bertviz
```

## Train and Test on the BIOSNAP Dataset

**Parameter-sharing**
```
python3 main.py --train_root './datasets/BIOSNAP/biosnap_train/' --train_path 'train_val_biosnap_smiles_new.csv' --test_root './datasets/BIOSNAP/biosnap_test/' --test_path 'test_biosnap_smiles_new.csv' --batch_size 8 --epochs 30 --lr 2e-5 --weight_decay 2e-4 --gamma 0.8 --dropout 0 --mode train --shared --model_name biosnap --saved_root './trained_record/biosnap/'
```

**Parameter-independent**
```
python3 main.py --train_root './datasets/BIOSNAP/biosnap_train/' --train_path 'train_val_biosnap_smiles_new.csv' --test_root './datasets/BIOSNAP/biosnap_test/' --test_path 'test_ biosnap_smiles_new.csv' --batch_size 8 --epochs 30 --lr 2e-5 --weight_decay 2e-4 --gamma 0.8 --dropout 0 --mode train --model_name biosnap --saved_root './trained_record/biosnap/'
```

<!--### Test using our best model

**Parameter-sharing**
```
python3 test.py --test_root './datasets/BIOSNAP/biosnap_test/' --test_path 'test_ biosnap_smiles_new.csv' --batch_size 8 --epochs 30 --lr 2e-5 --weight_decay 2e-4 --gamma 0.8 --dropout 0 --mode test --shared --model_name biosnap --saved_root './trained_record/biosnap/' --load_model_path './trained_record/biosnap/parameter_sharing/'
```

**Parameter-independent**
```
python3 test.py --test_root './datasets/BIOSNAP/biosnap_test/' --test_path 'test_ biosnap_smiles_new.csv' --batch_size 8 --epochs 30 --lr 2e-5 --weight_decay 2e-4 --gamma 0.8 --dropout 0 --mode test --model_name biosnap --saved_root './trained_record/biosnap/' --load_model_path './trained_record/biosnap/parameter_independent/'
```
-->

## Train and Test on the DrugBank Dataset

Please unzip the DrugBank dataset archive first.

**Parameter-sharing**

```
python3 main.py --train_root './datasets/drugbank/drugbank_train/' --train_path 'train_ drugbank_smiles_new.csv' --test_root './datasets/drugbank/drugbank_test/' --test_path 'test_ drugbank_smiles_new.csv' --batch_size 16 --epochs 30 --lr 2e-5 --weight_decay 1e-2 --gamma 0.8 --dropout 0 --mode train --shared --model_name drugbank --saved_root './trained_record/drugbank/'
```

**Parameter-independent**

```
python main.py --train_root './datasets/drugbank/drugbank_train/' --train_path 'train_drugbank_smiles_new.csv' --test_root './datasets/drugbank/drugbank_test/' --test_path 'test_drugbank_smiles_new.csv' --batch_size 512 --epochs 30 --lr 2e-5 --weight_decay 1e-2 --gamma 0.8 --dropout 0 --mode train --model_name drugbank --saved_root './trained_record/drugbank/'
```

<!-- ### Test using our best model

**Parameter-sharing**
```
python3 ddi2013.py --test_root './datasets/drugbank/drugbank_test/' --test_path 'test_ drugbank_smiles_new.csv' --batch_size 16 --lr 2e-5 --weight_decay 1e-2 --gamma 0.8 --dropout 0 --mode test --shared --model_name drugbank --saved_root './trained_record/drugbank/' --load_model_path './trained_record/drugbank/parameter_sharing/'
```

**Parameter-independent**
```
python3 ddi2013.py --test_root './datasets/drugbank/drugbank_test/' --test_path 'test_ drugbank_smiles_new.csv' --batch_size 16 --lr 2e-5 --weight_decay 1e-2 --gamma 0.8 --dropout 0 --mode test --model_name drugbank --saved_root './trained_record/drugbank/' --load_model_path './trained_record/drugbank/parameter_independent/'
```
-->

## Notice

> [!NOTE]
> If you use this code, please cite our paper:

```
@article{Qiu2024ptb-ddi,
  title={PTB-DDI: An Accurate and Simple Framework for Drug-Drug Interaction Prediction Based on Pre-trained Tokenizer and BiLSTM Model},
  author={Qiu, Jiayue and Yan, Xiao and Tian, Yanan and Li, Qin and Liu, Xiaomeng and Yang, Yuwei and Tang, Haiyi and Liu, Huanxiang},
  journal={IJMS},
  url={https://www.mdpi.com/1422-0067/25/21/11385}
  year={2024}
}
```

The original BIOSNAP and DrugBank datasets are from the following paper:

```
@article{huang2019caster,
  title={CASTER: Predicting Drug Interactions with Chemical Substructure Representation},
  author={Huang, Kexin and Xiao, Cao and Hoang, Trong Nghia and Glass, Lucas M and Sun, Jimeng},
  journal={AAAI},
  year={2020}
}
```
