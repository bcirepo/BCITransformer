# BCI Transformer

## Requirement
### Environment
We recommend using [PyCharm](https://www.jetbrains.com/pycharm/) as IDE.

Make sure you have `Python==3.9` installed on the computer.

A. Create PyCharm Project using Conda environment 

B. Unzip the code from this repository, and put inside the project


### Installation
Open terminal in PyCharm and install the dependency using below code

1. [PyTorch](https://pytorch.org/)
```bash
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 -c pytorch
```

2. Other requirements
```bash
pip install -r requirements.txt
```
   Please note that this library will download the dataset from third party website.


## Usage
Please follow these following steps to run the code.
### Download Dataset
Open and RUN [`generate_dataset.py`](https://github.com/bcirepo/BCITransformer/blob/main/generate_dataset.py) code through the PyCharm IDE.
This code aims to download and generate the corresponding MI dataset for each subject. First, it will download raw datasets from MOABB and save it in the local directory.


### Training and Evaluation
The code for training and evaluating this paradigm is inside [`main.py`](https://github.com/bcirepo/BCITransformer/blob/main/main.py). 
The fold must be an integer number between 0-9. The subject must be an integer represent the subject ID. 

Example to  train Subject Dependent, use:
```bash
if __name__ == '__main__':
    Train(dataset='Lee').SVtrain(subject=6, fold=0) 
```

Example to  train Subject Independent, use:
```bash
if __name__ == '__main__':
    Train(dataset='Lee').SItrain(subject=6) 
```


