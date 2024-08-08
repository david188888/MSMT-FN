import argparse
from utils.ch_train import ChConfig, ChRun
import nni
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



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
              num_hidden_layers=args['num_hidden_layers'], scheduler_type=args['scheduler']))

if __name__ == "__main__":
    try:
        tuner_params = nni.get_next_parameter() 
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', type=int, default=42, help='random seed')
        parser.add_argument('--batch_size', type=int, default=1, help='batch size')
        parser.add_argument('--lr', type=float, default=1e-5, help='learning rate, recommended: 5e-6 for mosi, mosei, 1e-5 for sims')
        parser.add_argument('--model', type=str, default='cme', help='concatenate(cc) or cross-modality encoder(cme)')
        parser.add_argument('--num_hidden_layers', type=int, default=3, help='number of hidden layers for cross-modality encoder')
        parser.add_argument('--scheduler', type=str, default='fixed', help='scheduler: exponentialLR, cosineAnnealingLR, reduceLROnPlateau')
        args = parser.parse_args()
        params = vars(args)
        # params.update(tuner_params)
        main(params)
    except Exception as exception:
        nni.report_final_result(exception)
        raise




