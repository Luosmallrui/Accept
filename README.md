# 环境安装基础

## 基础环境安装

- 查看python版本

```bash
python --version
```

- 查看cuda版本

```bash
nvcc --version
```

- 查看pytorch版本

```bash
python -c "import torch; print(torch.__version__)"
```

## 图神经网络库安装

> **whl离线下载地址:** [pytorch-geometric](https://pytorch-geometric.com/whl/)
> 下载完成之后使用 pip install xxx.whl安装

### torch_scatter

```bash 
pip install torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl
```

### torch_sparse

```bash 
pip install torch_sparse-0.6.13-cp38-cp38-linux_x86_64.whl
```

### dgl

```bash
pip install dgl-cu113 -f https://data.dgl.ai/wheels/repo.html
```

**初次安装会提示：DGL backend not selected or invalid. Assuming PyTorch for now.
Setting the default backend to "pytorch". You can change it in the <span style="color:red;">~/.dgl/config.json</span> file or export the DGLBACKEND environment variable. Valid options are: pytorch, mxnet, tensorflow (all lowercase)
Downloading https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/f1fc0d14b3b019c562737240d06ec83b07d16a8f/new_data/chameleon/out1_node_feature_label.txt
**

**使用 vim <span style="color:red;">~/.dgl/config.json</span>** 可修改为pytorch或 mxnet或tensorflow

**⚠️** 若torch_scatter以及torch_sparse版本与torch、python、cuda版本对应不上会报错:**Segmentation fault**

# Graph Neural Network Performance on Different Datasets

以下表格展示了在不同数据集上，不同层数的各种GNN模型的性能表现。

## 数据集

- cora
- citeseer
- wiki
- pubmed
- photo
- computers

## 模型

- AERO-GNN
- GCN
- APPNP
- A-DGN
- GAT
- MixHop
- GPRGNN
- DAGNN
- GraphSage
- GDC (Graph Diffusion Convolution)

## 层数

- 1
- 2
- 4
- 8
- 16
- 32
- 64

## AERO-GNN

| Dataset   | Model    | Layers | Performance |
|-----------|----------|--------|-------------|
| cora      | AERO-GNN | 1      |             |
| cora      | AERO-GNN | 2      |             |
| cora      | AERO-GNN | 4      |             |
| cora      | AERO-GNN | 8      |             |
| cora      | AERO-GNN | 16     |             |
| cora      | AERO-GNN | 32     |             |
| cora      | AERO-GNN | 64     |             |
| cora      | GCN      | 1      |             |
| cora      | GCN      | 2      |             |
| cora      | GCN      | 4      |             |
| cora      | GCN      | 8      |             |
| cora      | GCN      | 16     |             |
| cora      | GCN      | 32     |             |
| cora      | GCN      | 64     |             |
| ...       | ...      | ...    | ...         |
| ...       | ...      | ...    | ...         |
| ...       | ...      | ...    | ...         |
| computers | GDC      | 1      |             |
| computers | GDC      | 2      |             |
| computers | GDC      | 4      |             |
| computers | GDC      | 8      |             |
| computers | GDC      | 16     |             |
| computers | GDC      | 32     |             |
| computers | GDC      | 64     |             |


## texas数据集
| Model    | Layers | Performance     |
|----------|--------|-----------------|
| AERO-GNN | 1      | Accuracy: 0.8378|
| AERO-GNN | 2      | Accuracy: 0.7838|
| AERO-GNN | 4      | Accuracy: 0.7568|


## cora数据集
```bash
python3 ./AERO-GNN/main.py --model aero  --iterations 32 --dr 0.0001 --dr-prop 0.0001 --dropout 0.7 --add-dropout 0 --lambd 1.0 --num-layers 8 --dataset cora
```
| Layer | Epoch | Model | Trials | Dataset | Performance                  |
|-------|-------|-------|--------|---------|------------------------------|
| 1     | 50    | AERO  | 50     | cora    | Accuracy: 0.7496 ± 0.010     |
| 1     | 50    | AERO  | 50     | cora    | Accuracy: 0.7496 ± 0.010     |
| 2     | 50    | AERO  | 50     | cora    | Accuracy: 0.7810 ± 0.011     |
| 4     | 50    | AERO  | 50     | cora    | Accuracy: 0.8019 ± 0.011     |
| 8     | 50    | AERO  | 50     | cora    | Accuracy: 0.8191 ± 0.009     |
| 16    | 50    | AERO  | 50     | cora    | Accuracy: 0.8234 ± 0.008     |
| 32    | 50    | AERO  | 50     | cora    | Accuracy: 0.8176 ± 0.010     |
| 64    | 50    | AERO  | 50     | cora    | Accuracy: 0.8133 ± 0.010     |


## cora数据集
```bash
python3 ./AERO-GNN/main.py --model aero  --iterations 32 --dr 0.0001 --dr-prop 0.0001 --dropout 0.7 --add-dropout 0 --lambd 1.0 --num-layers 8 --dataset wiki
```
| Layer | Epoch | Model | Trials | Dataset | Performance                  |
|-------|-------|-------|--------|---------|------------------------------|
| 1     | 2000  | AERO  | 50     | wiki    | Accuracy: 0.7765 ± 0.007     |
| 2     | 2000  | AERO  | 50     | wiki    | Accuracy: 0.7850 ± 0.007     |
| 4     | 2000  | AERO  | 50     | wiki    | Accuracy: 0.7886 ± 0.007     |
| 8     | 2000  | AERO  | 50     | wiki    | Accuracy: 0.7881 ± 0.008     |
| 16    | 2000  | AERO  | 50     | wiki    | Accuracy: 0.7849 ± 0.010     |
| 32    | 2000  | AERO  | 50     | wiki    | Accuracy: 0.7826 ± 0.010     |
| 64    | 2000  | AERO  | 50     | wiki    | Accuracy: 0.7775 ± 0.025     |


