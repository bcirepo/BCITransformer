# BCI Transformer

## Requirement
### Environment
Make sure you have `Python==3.9` installed on the computer.

### Installation
1. [PyTorch](pytorch.org)
```bash
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 -c pytorch
```

2. [MOABB](http://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014001.html)
```bash
pip install moabb==0.4.5
```
   Please note that this library will download the dataset from third party website.


## Usage
Please follow these following steps to run the code.
### Download Dataset
Open [`generate_dataset.py`](https://github.com/dreamsentropy/BCITransformer/blob/main/generate_dataset.py) code through the IDE.
This code aims to download and generate the corresponding MI dataset for each subject. First, it will download raw datasets from MOABB and save it in the local directory.

Example to generate Dataset, use:
```bash
Dataset(dataset='Lee').get_dataset()
```

### Training and Evaluation
The code to train and evaluate this paradigm is inside [`main.py`](https://github.com/dreamsentropy/BCITransformer/blob/main/main.py). 
The fold must be an integer number between 1-10. The subject must be an integer represent the subject ID. 

Example to  train Subject Dependent, use:
```bash
Train(dataset='Lee').SVT(subject=1, fold=1) 
```

Example to  train Subject Independent, use:
```bash
Train(dataset='Lee').SIT(subject=1) 
```

