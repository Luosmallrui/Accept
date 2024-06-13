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
Setting the default backend to "pytorch". You can change it in the <span style="color:red;">~/.dgl/config.json</span>
file or export the DGLBACKEND environment variable. Valid options are: pytorch, mxnet, tensorflow (all lowercase)
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

## Result

| layer | epoch | Model | n trials | dataset | Mean test accuracy | std deviation | Mean Dirichlet energy | std Dirichlet energy |
|-------|-------|-------|----------|---------|--------------------|---------------|-----------------------|----------------------|
| 1     | 2000  | aero  | 50       | wiki    | 0.7765             | 0.007         | 40.5089               | 7.922                |
| 2     | 2000  | aero  | 50       | wiki    | 0.7850             | 0.007         | 25.9188               | 6.531                |
| 4     | 2000  | aero  | 50       | wiki    | 0.7886             | 0.007         | 23.7471               | 5.147                |
| 8     | 2000  | aero  | 50       | wiki    | 0.7879             | 0.008         | 42.2686               | 34.785               |
| 16    | 2000  | aero  | 50       | wiki    | 0.7844             | 0.011         | 2279.2766             | 12285.911            |
| 32    | 2000  | aero  | 50       | wiki    | 0.7777             | 0.025         | 177577.7500           | 1151560.000          |
| 64    | 2000  | aero  | 50       | wiki    | 0.7791             | 0.015         | 60581.2383            | 172291.781           |
| 1     | 2000  | aero  | 50       | cora    | 0.7496             | 0.010         | 2.2397                | 0.441                |
| 2     | 2000  | aero  | 50       | cora    | 0.7810             | 0.011         | 1.4156                | 0.257                |
| 4     | 2000  | aero  | 50       | cora    | 0.8019             | 0.011         | 1.2379                | 0.135                |
| 8     | 2000  | aero  | 50       | cora    | 0.8191             | 0.009         | 1.0681                | 0.202                |
| 16    | 2000  | aero  | 50       | cora    | 0.8234             | 0.008         | 0.8296                | 0.142                |
| 32    | 2000  | aero  | 50       | cora    | 0.8176             | 0.010         | 0.8440                | 0.185                |
| 64    | 2000  | aero  | 50       | cora    | 0.8133             | 0.010         | 1.2234                | 0.257                |
| 1     | 2000  | gcn   | 50       | wiki    | 0.7788             | 0.006         | 6.2591                | 0.315                |
| 2     | 2000  | gcn   | 50       | wiki    | 0.7739             | 0.011         | 7.7679                | 0.560                |
| 4     | 2000  | gcn   | 50       | wiki    | 0.7330             | 0.015         | 8.6933                | 0.764                |
| 8     | 2000  | gcn   | 50       | wiki    | 0.6084             | 0.030         | 6.3843                | 1.081                |
| 16    | 2000  | gcn   | 50       | wiki    | 0.2882             | 0.021         | 1.5710                | 0.255                |
| 32    | 2000  | gcn   | 50       | wiki    | 0.2848             | 0.010         | 1.7027                | 0.229                |
| 64    | 2000  | gcn   | 50       | wiki    | 0.2718             | 0.013         | 1.4893                | 0.342                |
| 1     | 2000  | gcn   | 50       | cora    | 0.7861             | 0.011         | 0.5897                | 0.032                |
| 2     | 2000  | gcn   | 50       | cora    | 0.7771             | 0.019         | 0.8480                | 0.071                |
| 4     | 2000  | gcn   | 50       | cora    | 0.5550             | 0.105         | 0.9699                | 0.242                |
| 8     | 2000  | gcn   | 50       | cora    | 0.3363             | 0.118         | 0.4394                | 0.240                |
| 16    | 2000  | gcn   | 50       | cora    | 0.2086             | 0.067         | 0.1115                | 0.066                |
| 32    | 2000  | gcn   | 50       | cora    | 0.1996             | 0.065         | 0.1004                | 0.064                |
| 64    | 2000  | gcn   | 50       | cora    | 0.1988             | 0.073         | 0.0916                | 0.069                |

## wiki数据集

| layer | epoch | Model | n trials | dataset | Mean test accuracy | std deviation | Mean Dirichlet energy | std Dirichlet energy |
|-------|-------|-------|----------|---------|--------------------|---------------|-----------------------|----------------------|
| 1     | 2000  | aero  | 50       | wiki    | 0.7765             | 0.007         | 40.5089               | 7.922                |
| 2     | 2000  | aero  | 50       | wiki    | 0.7850             | 0.007         | 25.9188               | 6.531                |
| 4     | 2000  | aero  | 50       | wiki    | 0.7886             | 0.007         | 23.7471               | 5.147                |
| 8     | 2000  | aero  | 50       | wiki    | 0.7879             | 0.008         | 42.2686               | 34.785               |
| 16    | 2000  | aero  | 50       | wiki    | 0.7844             | 0.011         | 2279.2766             | 12285.911            |
| 32    | 2000  | aero  | 50       | wiki    | 0.7777             | 0.025         | 177577.7500           | 1151560.000          |
| 64    | 2000  | aero  | 50       | wiki    | 0.7791             | 0.015         | 60581.2383            | 172291.781           |
| 1     | 2000  | gcn   | 50       | wiki    | 0.7788             | 0.006         | 6.2591                | 0.315                |
| 2     | 2000  | gcn   | 50       | wiki    | 0.7739             | 0.011         | 7.7679                | 0.560                |
| 4     | 2000  | gcn   | 50       | wiki    | 0.7330             | 0.015         | 8.6933                | 0.764                |
| 8     | 2000  | gcn   | 50       | wiki    | 0.6084             | 0.030         | 6.3843                | 1.081                |
| 16    | 2000  | gcn   | 50       | wiki    | 0.2882             | 0.021         | 1.5710                | 0.255                |
| 32    | 2000  | gcn   | 50       | wiki    | 0.2848             | 0.010         | 1.7027                | 0.229                |
| 64    | 2000  | gcn   | 50       | wiki    | 0.2718             | 0.013         | 1.4893                | 0.342                |

## cora数据集

| layer | epoch | Model | n trials | dataset | Mean test accuracy | std deviation | Mean Dirichlet energy | std Dirichlet energy |
|-------|-------|-------|----------|---------|--------------------|---------------|-----------------------|----------------------|
| 1     | 2000  | aero  | 50       | cora    | 0.7496             | 0.010         | 2.2397                | 0.441                |
| 2     | 2000  | aero  | 50       | cora    | 0.7810             | 0.011         | 1.4156                | 0.257                |
| 4     | 2000  | aero  | 50       | cora    | 0.8019             | 0.011         | 1.2379                | 0.135                |
| 8     | 2000  | aero  | 50       | cora    | 0.8191             | 0.009         | 1.0681                | 0.202                |
| 16    | 2000  | aero  | 50       | cora    | 0.8234             | 0.008         | 0.8296                | 0.142                |
| 32    | 2000  | aero  | 50       | cora    | 0.8176             | 0.010         | 0.8440                | 0.185                |
| 64    | 2000  | aero  | 50       | cora    | 0.8133             | 0.010         | 1.2234                | 0.257                |
| 1     | 2000  | gcn   | 50       | cora    | 0.7861             | 0.011         | 0.5897                | 0.032                |
| 2     | 2000  | gcn   | 50       | cora    | 0.7771             | 0.019         | 0.8480                | 0.071                |
| 4     | 2000  | gcn   | 50       | cora    | 0.5550             | 0.105         | 0.9699                | 0.242                |
| 8     | 2000  | gcn   | 50       | cora    | 0.3363             | 0.118         | 0.4394                | 0.240                |
| 16    | 2000  | gcn   | 50       | cora    | 0.2086             | 0.067         | 0.1115                | 0.066                |
| 32    | 2000  | gcn   | 50       | cora    | 0.1996             | 0.065         | 0.1004                | 0.064                |
| 64    | 2000  | gcn   | 50       | cora    | 0.1988             | 0.073         | 0.0916                | 0.069                |