# My Paper Title

This is a reproduction of [Identity-Aware Graph Neural Networks](https://ojs.aaai.org/index.php/AAAI/article/view/17283) by Zhihao Dong and Jeremy Banks.

```
@report{You2021,
   author = {Jiaxuan You and Jonathan M Gomes-Selman and Rex Ying and Jure Leskovec},
   keywords = {Data Mining & Knowledge Management: Graph Mining & Social Network Analysis & Community,Machine Learning: Graph-based Machine Learning,Machine Learning: Relational Learning,Machine Learning: Representation Learning},
   title = {Identity-aware Graph Neural Networks},
   url = {http://snap.stanford.edu/idgnn},
   year = {2021},
}
```

## Requirements

To install requirements:

**Requirements**

- CPU or NVIDIA GPU, Linux, Python3
- PyTorch, various Python packages; Instructions for installing these dependencies are found below


**1. Python environment (Optional):**
We recommend using Conda package manager

```bash
conda create -n graphgym python=3.7
source activate graphgym
```

**2. Pytorch:**
Install [PyTorch](https://pytorch.org/). 
We have verified GraphGym under PyTorch 1.8.0, and GraphGym should work with PyTorch 1.4.0+. For example:
```bash
# CUDA versions: cpu, cu92, cu101, cu102, cu101, cu111
pip install torch==1.8.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

**3. Pytorch Geometric:**
Install [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html), 
follow their instructions. For example:
```bash
# CUDA versions: cpu, cu92, cu101, cu102, cu101, cu111
# TORCH versions: 1.4.0, 1.5.0, 1.6.0, 1.7.0, 1.8.0
CUDA=cu101
TORCH=1.8.0
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric
```

**4. GraphGym and other dependencies:**


```bash
git clone https://github.com/snap-stanford/GraphGym
cd GraphGym
pip install -r requirements.txt
pip install -e .  # From latest verion
pip install graphgym # (Optional) From pypi stable version
```

TF_GEOMETRIC NEEDS SPECIAL SPARSE OPERATIONS. ADD INFORMATION ABOUT HOW TO GET IT.

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing



>📋  Pick a licence and describe how to contribute to your code repository. 
