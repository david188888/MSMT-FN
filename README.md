# MSMT-FN: Multi-segment Multitask Fusion Network for Marketing Audio Classification


This repository contains code for multimodal user purchase intention judgment, which is suitable for processing and analyzing telephone recording data to predict user purchase intention.

## Environment Setup
1. Create a new environment using conda or pip
2. Clone this repository to your local machine:
3. Install the required packages by running `pip install -r requirements.txt`.

## Training parameters
```bash
python run.py

optional arguments:
    --lr : learning rate
    --seed: random seed
    --dropout: dropout rate
    --hidden_size_gru: hidden size of GRU
    --bottleneck_layers : bottleneck layers
    --n_bottlenecks : number of bottleneck layers
    --accumulation_steps : accumulation steps
    --scheduler : scheduler ,'scheduler: fixed, cosineAnnealingLR'
    --num_hidden_layers : number of hidden layers of CME
```


