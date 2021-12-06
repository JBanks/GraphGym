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

## Training

To train the model(s) in the paper, run this command:

You can run the training by selecting the model using the following formula:
```bash
BASE=0
FAST=1
FULL=2
BASE_MODEL='gcn' # or any of the following: ['sage', 'gat', 'gin']
AUGMENT=$BASE # or any of the following: [FAST, FULL]
declare -a models=("${BASE_MODEL}conv_tf"
                   "${BASE_MODEL}convFast_tf"
                   "id${BASE_MODEL}_tf")
MODEL=${models[$AUGMENT]}
```

```bash
python main_zd.py --model $MODEL
```

You can also simply type the following, using the available models:

```bash
python main_zd.py --model ginconv_tf
```

## Evaluation

The evaluations of datasets are automatically placed in the results folder after training.  Simply navigate to the `results/val/final` folder and locate the file associated with the model that you trained.

## Results

Our model achieves the following performance on the Node classification task using the datasets from the original paper:

<div id="tab:node_class">

|                               |      |    ScaleFree     |    SmallWorld    |     Enzymes      |     Proteins     |
|:-----------------------------:|:----:|:----------------:|:----------------:|:----------------:|:----------------:|
|              GNN              | GCN  | **0.695** ± 0.01 | **0.489** ± 0.05 |   0.540 ± 0.06   |   0.481 ± 0.01   |
|                               | SAGE |   0.470 ± 0.03   |   0.271 ± 0.03   | **0.574** ± 0.08 |   0.491 ± 0.02   |
|                               | GAT  |   0.470 ± 0.03   |   0.271 ± 0.03   |   0.492 ± 0.07   |   0.441 ± 0.02   |
|                               | GIN  |   0.639 ± 0.01   |   0.470 ± 0.04   |   0.543 ± 0.06   | **0.530** ± 0.01 |
|        **ID-GNN Fast**        | GCN  |   0.764 ± 0.00   |   0.571 ± 0.05   |   0.724 ± 0.05   |   0.728 ± 0.01   |
|                               | SAGE | **0.909** ± 0.01 | **0.982** ± 0.01 | **0.956** ± 0.03 | **0.965** ± 0.01 |
|                               | GAT  |   0.581 ± 0.02   |   0.616 ± 0.04   |   0.636 ± 0.05   |   0.621 ± 0.02   |
|                               | GIN  |   0.687 ± 0.03   |   0.709 ± 0.04   |   0.663 ± 0.04   |   0.640 ± 0.03   |
|        **ID-GNN Full**        | GCN  |   0.964 ± 0.01   | **0.994** ± 0.00 |   0.970 ± 0.03   |   0.986 ± 0.01   |
|                               | SAGE |   0.579 ± 0.07   |   0.271 ± 0.03   |   0.608 ± 0.07   |   0.527 ± 0.01   |
|                               | GAT  | **0.987** ± 0.00 |   0.967 ± 0.04   | **0.981** ± 0.02 | **0.991** ± 0.00 |
|                               | GIN  |   0.660 ± 0.03   |   0.503 ± 0.05   |   0.521 ± 0.09   |   0.540 ± 0.01   |
| **Best ID-GNN over best GNN** |      |      29.2%       |      50.5%       |      40.7%       |      46.1%       |

Results of node classification: predicting clustering coefficient

</div>

<div id="tab:node_class_real">

|                               |      |       Cora       |     CiteSeer     |
|:-----------------------------:|:----:|:----------------:|:----------------:|
|              GNN              | GCN  | **0.879** ± 0.00 |   0.763 ± 0.01   |
|                               | SAGE | **0.879** ± 0.00 |   0.762 ± 0.02   |
|                               | GAT  |   0.878 ± 0.00   | **0.770** ± 0.01 |
|                               | GIN  |   0.835 ± 0.01   |   0.702 ± 0.02   |
|       **ID-GNNs Fast**        | GCN  |   0.880 ± 0.01   |   0.756 ± 0.01   |
|                               | SAGE |   0.878 ± 0.01   |   0.754 ± 0.01   |
|                               | GAT  | **0.881** ± 0.01 | **0.759** ± 0.00 |
|                               | GIN  |   0.809 ± 0.05   |   0.678 ± 0.01   |
|       **ID-GNNs Full**        | GCN  |   0.787 ± 0.03   |   0.767 ± 0.00   |
|                               | SAGE | **1.000** ± 0.00 |   0.938 ± 0.09   |
|                               | GAT  |   0.885 ± 0.00   |   0.771 ± 0.01   |
|                               | GIN  | **1.000** ± 0.00 | **0.948** ± 0.07 |
| **Best ID-GNN over best GNN** |      |      12.1%       |      17.8%       |

Results of node classification: real-world labels

</div>

We fully acknowledge that the results on the real-world datasets with greater than 90% accuracy are likely erroneous, and we ask you to bear with us while we investigate this issue.

## Contributing

We did not find a licence section on the page of the original authors, but they designed the GraphGym framework for the express purpose of allowing contributions.  I freely welcome you forking and continuing any of our work, however I am not willing to accept pull requests.  If you fork and continue this work, please let me know and I will place a link to your fork in this readme.  The original authors strongly encourage people submitting pull requests to contribute to their project.  I encourage you to contribute any PyTorch modules to the original GitHub repo available here: https://github.com/snap-stanford/GraphGym
