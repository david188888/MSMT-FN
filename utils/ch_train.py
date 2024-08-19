from typing import List
import torch
from torch import nn
from torch.utils.data import BatchSampler
from tqdm import tqdm
from utils.metricsTop import MetricsTop
from utils.ch_model import rob_hub_cme
import random
import numpy as np
from utils.data_loader import data_loader
import json
from torch.utils.tensorboard import SummaryWriter
# import nni


from datetime import datetime
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
train_log_dir = 'logs/origin/train/' + TIMESTAMP
test_log_dir = 'logs/origin/test/'   + TIMESTAMP
val_log_dir = 'logs/origin/val/' + TIMESTAMP


writer_train = SummaryWriter(train_log_dir)
writer_val = SummaryWriter(val_log_dir)
writer_test = SummaryWriter(test_log_dir)


# global variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def dict_to_str(src_dict):
    dst_str = ""
    for key in src_dict.keys():
        dst_str += " %s: %.4f " % (key, src_dict[key])
    return dst_str


class ChConfig(object):
    """Configuration class to store the configurations of training.
    """

    def __init__(self,
                 model_save_path='checkpoint/',
                 learning_rate=1e-5,
                 epochs=30,
                 early_stop=8,
                 dropout=0.1,
                 seed=42,
                 batch_size=1,
                 num_hidden_layers=2,
                 n_bottlenecks = 2,
                 bottleneck_layers =2,
                 scheduler_type = 'fixed',
                 num_layers_gru = 2,
                 accumulation_steps = 4,
                 hidden_size_gru = 128,
                 use_regularization = 'L2',
                 input_dim=99072,
                 output_dim=768,
                 ):

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dropout = dropout
        self.model_save_path = model_save_path
        self.early_stop = early_stop
        self.seed = seed
        self.accumulation_steps = accumulation_steps
        self.batch_size = batch_size
        self.num_hidden_layers = num_hidden_layers
        self.n_bottlenecks = n_bottlenecks
        self.bottleneck_layers = bottleneck_layers
        self.scheduler_type = scheduler_type
        self.num_layers_gru = num_layers_gru
        self.hidden_size_gru = hidden_size_gru
        self.use_regularization = use_regularization
        self.input_dim = input_dim
        self.output_dim = output_dim


class ChTrainer():
    def __init__(self, config):

        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        self.metrics = MetricsTop().getMetics()
        self.scheduler_type = config.scheduler_type
        
    
    def get_scheduler(self, optimizer, scheduler_type):
        if scheduler_type == 'fixed':
            scheduler_type == None
        elif scheduler_type == 'exponentialLR':
            return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        elif scheduler_type == 'cosineAnnealingLR':
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        elif scheduler_type == 'reduceLROnPlateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    def do_train(self, model, data_loader):
        model.train()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.config.learning_rate)
        large_epoch = self.config.epochs
        total_steps = len(data_loader) * large_epoch
        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer, num_warmup_steps=0.1*total_steps, num_training_steps=total_steps)
        total_loss = 0
        accumulation_steps = self.config.accumulation_steps
        input_size = 0
        # Loop over all batches.
        for i, batch in enumerate(tqdm(data_loader)):
            text_inputs = batch["text_tokens"].squeeze(0).to(device)
            audio_inputs = batch["audio_inputs"].squeeze(0).to(device)
            text_mask = batch["text_masks"].squeeze(0).to(device)
            audio_mask = batch["audio_masks"].squeeze(0).to(device)
            targets = batch["targets"]


            batch_size = self.config.batch_size

            outputs = model(text_inputs, text_mask, audio_inputs, audio_mask, batch_size)
            # Compute the training loss.
            loss = 0.0
            loss = self.criterion(
                outputs, targets.long().to(device).view(-1))
            
            # 实现L1正则化
            if self.config.use_regularization == 'L1':
                l1_weight = 0.5
                l1_regularization = l1_weight * sum(param.abs().sum() for param in model.parameters())
                loss += l1_regularization
                
            # 实现L2正则化
            elif self.config.use_regularization == 'L2':
                l2_weight = 0.5
                l2_regularization = l2_weight * sum(param.pow(2).sum() for param in model.parameters())
                loss += l2_regularization
                
            else:
                pass
            loss.backward()
            if (i+1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()  
            total_loss += loss.item()*text_inputs.size(0)
            input_size += text_inputs.size(0)
            final_loss = round(total_loss / input_size, 4)
        return final_loss

            # scheduler.step()

    def do_test(self, model, data_loader, mode):
        model.eval()
        y_pred = []
        y_true = []
        total_loss = 0
        input_size = 0
        with torch.no_grad():
            # Loop over all batches.
            for batch in tqdm(data_loader):

                text_inputs = batch["text_tokens"].to(device)
                audio_inputs = batch["audio_inputs"].to(device)
                text_mask = batch["text_masks"].to(device)
                audio_mask = batch["audio_masks"].to(device)
                targets = batch["targets"]

                # Predictions from 1 batch of data.
                outputs = model(text_inputs, text_mask,
                                audio_inputs, audio_mask, self.config.batch_size)

                # Compute the training loss.
                loss = 0.0
                loss = self.criterion(
                    outputs, targets.long().to(device).view(-1))
                total_loss += float(loss.item()*text_inputs.size(0))
                input_size += text_inputs.size(0)

                # add predictions
                y_pred.append(outputs.cpu())
                y_true.append(targets.cpu())

        total_loss = round(total_loss / input_size, 4)
        print(mode+" >> loss: ", total_loss)

        eval_results = {}
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        
        results = self.metrics(pred, true)
        print(dict_to_str(results))
        eval_results['M_result'] = results
        eval_results['Loss'] = total_loss
        
                
        return eval_results
    
    


def ChRun(config):
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache()

    train_loader, test_loader, val_loader = data_loader(batch_size=config.batch_size)
    
    model = rob_hub_cme(config).to(device)
    for param in model.hubert_model.feature_extractor.parameters():
        param.requires_grad = False

    trainer = ChTrainer(config)

    large_epoch = config.epochs
    lowest_eval_loss = 100
    highest_eval_acc_5 = 0
    highest_eval_acc_4 = 0
    highest_eval_acc_3 = 0
    highest_eval_acc_2 = 0
    total_acc_test = []
    epoch = 0
    best_epoch = 0
    for epoch in range(large_epoch):
        print('---------------------EPOCH: ', epoch, '--------------------')
        total_loss_train = trainer.do_train(model, train_loader)
        # writer_train.add_scalar('Loss/TRAIN', total_loss_train, epoch)
        eval_results = trainer.do_test(model, val_loader, "VAL")
        # nni.report_intermediate_result(eval_results['Mult_acc_5'])
        mode = "VAL"
        total_loss_val = eval_results['Loss']
        # writer_val.add_scalar('Loss/'+mode, total_loss_val, epoch)
        # writer_val.add_scalar('F1/'+mode, eval_results['F1_score'], epoch)
        # writer_val.add_scalar('ACC2/'+mode, eval_results['Mult_acc_2'], epoch)
        # writer_val.add_scalar('ACC3/'+mode, eval_results['Mult_acc_3'], epoch)
        # writer_val.add_scalar('ACC4/'+mode, eval_results['Mult_acc_4'], epoch)
        # writer_val.add_scalar('ACC5/'+mode, eval_results['Mult_acc_5'], epoch)
        # writer_val.add_scalar('MAE/'+mode, eval_results['MAE'], epoch)
        # writer_val.add_scalar('Corr/'+mode, eval_results['Corr'], epoch)
#         test_results = trainer.do_test(model, test_loader,"TEST")
        # if eval_results['Loss'] < lowest_eval_loss:
        #     lowest_eval_loss = eval_results['Loss']
        #     torch.save(model.state_dict(), config.model_save_path+'origin_loss.pth')
        #     best_epoch = epoch
        # if eval_results['Mult_acc_5'] >= highest_eval_acc_5:
        #     highest_eval_acc_5 = eval_results['Mult_acc_5']
        #     torch.save(model.state_dict(), config.model_save_path+'origin_acc_5.pth')
            
        # if eval_results['Mult_acc_4'] >= highest_eval_acc_4:
        #     highest_eval_acc_4 = eval_results['Mult_acc_4']
        #     torch.save(model.state_dict(), config.model_save_path+'origin_acc_4.pth')
            
        # if eval_results['Mult_acc_3'] >= highest_eval_acc_3:
        #     highest_eval_acc_3 = eval_results['Mult_acc_3']
        #     torch.save(model.state_dict(), config.model_save_path+'origin_acc_3.pth')
            
        # if eval_results['Mult_acc_2'] >= highest_eval_acc_2:
        #     highest_eval_acc_2 = eval_results['Mult_acc_2']
        #     torch.save(model.state_dict(), config.model_save_path+'origin_acc_2.pth')
        

        # with open('results/origin/val_results_origin.txt', 'a') as f:
        #     f.write('EPOCH: '+str(epoch)+'  '+dict_to_str(eval_results)+'\n')
        #     f.write('best_epoch: '+str(best_epoch)+'  '+'lowest_eval_loss: '+str(lowest_eval_loss)+'\n')

#     model.eval()

    model.eval()
    acc_test_results = trainer.do_test(model, test_loader, "TEST")
    # nni.report_final_result(test_results_acc['Mult_acc_5'])
    print('%s: >> ' % (f'TEST (highest val acc_{i}) ') +
            dict_to_str(acc_test_results))
    total_acc_test.append(dict_to_str(acc_test_results))
    
    

    model.load_state_dict(torch.load(config.model_save_path+'origin_loss.pth'))
    loss_test_results = trainer.do_test(model, test_loader, "TEST")
    # nni.report_final_result(test_results_acc['Mult_acc_5'])
    print('%s: >> ' % ('TEST (lowest val loss) ') +
          dict_to_str(loss_test_results))
    
    
    # # 将结果存进对应的文件
    # with open('results/origin/test_results_origin.txt', 'w+') as f:
    #     f.write('TEST (highest val acc) ' + ', '.join(map(str, total_acc_test)) + '\n')
    #     f.write('TEST (lowest val loss) ' + dict_to_str(loss_test_results) + '\n')
    
    # writer_train.close()
    # writer_test.close()
    # writer_val.close()
    


