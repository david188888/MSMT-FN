## 主要参数
ChConfig 里面制定的主要的参数

1. `--num_hidden_layers`：CME 层的层数
2. `num_layers_gru`: GRU 层的层数
3. `hidden_size_gru`: GRU 层的隐藏层大小



## 模型文件
ch_model 是主模型，cross_attn_encoder 定义了注意力机制模块 GRU模块和 Bottleneck模块


## 数据集的路径已经在dataloader最下面定义好了


## 训练模型
```python
python run.py
```



