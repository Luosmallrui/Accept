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

### torch_sparse

### dgl

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