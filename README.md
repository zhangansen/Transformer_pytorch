# Transformer Model

本仓库实现了比较简单的Transformer模型，记录了本人学习Transformer模型的过程。

![Alt text](<structure.jpg>)

## Setup

你可以运行如下代码来安装所需依赖：

```shell
pip install -r requirements.txt
```

## Repository structure

```plain
|-- data # 存放数据处理代码
|-- layers.py # Transformer各层定义
|-- train.py # 模型训练与推理
|-- model.py # Transformer模型定义
|-- requiremens.txt # 需要的依赖
|-- utils.py # 辅助工具
|-- bleu.py #bleu指标的实现代码
```

## Run pipeline

```plain
Optional arguments:
  --d_moel   Dimension of word embeddings(default:512)
  --d_ff     Dimension of feed-forward layer(default:2048)
  --d_k		 Dimension of key and value vectors(default:64)
  --d_v		 Dimension of value vectors(defalut:64)
  --n_layers Dimension of encoder and decoder layers(defalut:6)
  --n_heads  Dimension of 
  --lr       Learning rate(default:1e-3)
  --epochs   Number of epoch(default:200)
```

## Attribution

https://zhuanlan.zhihu.com/p/403433120
https://zhuanlan.zhihu.com/p/650703073