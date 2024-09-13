import argparse
from utils.ch_train import ChConfig, ChRun
import nni
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


from setuptools import setup
import setuptools.command.install

def strtobool(val):
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError(f"Invalid truth value: {val}")

def main(args):
    ChRun(ChConfig(batch_size=args['batch_size'],learning_rate=args['lr'],seed=args['seed'],
              num_hidden_layers=args['num_hidden_layers'], scheduler_type=args['scheduler'], accumulation_steps=args['accumulation_steps'],
              n_bottlenecks=args['n_bottlenecks'], bottleneck_layers=args['bottleneck_layers'], hidden_size_gru=args['hidden_size_gru'],
              num_layers_gru=args['num_layers_gru'],epochs=args['epochs'], use_regularization=(args['use_regularization']), dropout=args['dropout']))

if __name__ == "__main__":
    try:
        tuner_params = nni.get_next_parameter() 
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', type=int, default=42, help='random seed')
        parser.add_argument('--batch_size', type=int, default=1, help='batch size')
        parser.add_argument('--epochs', type=int, default=35, help='number of epochs')
        parser.add_argument('--lr', type=float, default=0.00001, help='learning rate, recommended: 5e-6 for mosi, mosei, 1e-5 for sims')
        parser.add_argument('--num_hidden_layers', type=int, default=4, help='number of hidden layers for cross-modality encoder')
        parser.add_argument('--scheduler', type=str, default='fixed', help='scheduler: fixed, cosineAnnealingLR')
        parser.add_argument('--accumulation_steps', type=int, default=1, help='gradient accumulation steps')
        parser.add_argument('--n_bottlenecks', type=int, default=4, help='number of bottlenecks')
        parser.add_argument('--bottleneck_layers', type=int, default=2, help='number of bottleneck layers')
        parser.add_argument('--hidden_size_gru', type=int, default=128, help='hidden size for GRU')
        parser.add_argument('--num_layers_gru', type=int, default=2, help='number of layers for GRU')
        parser.add_argument('--use_regularization', type=str, default='none', help='use regularization or not')
        parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
        args = parser.parse_args()
        params = vars(args)
        params.update(tuner_params)
        main(params)
    except Exception as exception:
        nni.report_final_result(exception)
        raise





