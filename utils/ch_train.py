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
import nni


from datetime import datetime
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
train_log_dir = 'logs/meld/train/' + TIMESTAMP
# test_log_dir = 'logs/sims/test/'   + TIMESTAMP
val_log_dir = 'logs/meld/val/' + TIMESTAMP


writer_train = SummaryWriter(train_log_dir)
writer_val = SummaryWriter(val_log_dir)


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
                 model_save_path='checkpoint_new/',
                 learning_rate=1e-5,
                 epochs=20,
                 early_stop=8,
                 dropout=0.1,
                 seed=42,
                 batch_size=1,
                 num_hidden_layers=2,
                 n_bottlenecks = 2,
                 dataset_name = 'meld',
                 bottleneck_layers =2,
                 scheduler_type = 'fixed',
                 num_layers_gru = 2,
                 accumulation_steps = 16,
                 hidden_size_gru = 128,
                 use_regularization = 'none',
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
        self.dataset_name = dataset_name


class ChTrainer():
    def __init__(self, config):

        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        # self.criterion = nn.L1Loss()
        self.metrics = MetricsTop().getMetics()
        self.scheduler_type = config.scheduler_type
        
    
    def get_scheduler(self, optimizer, scheduler_type):
        if scheduler_type == 'fixed':
            scheduler_type == None
        elif scheduler_type == 'cosineAnnealingLR':
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    def do_train(self, model, data_loader):
        model.train()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.config.learning_rate, weight_decay=0.01)
        large_epoch = self.config.epochs
        total_steps = len(data_loader) * large_epoch
        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer, num_warmup_steps=0.1*total_steps, num_training_steps=total_steps)
        total_loss = 0
        accumulation_steps = self.config.accumulation_steps
        input_size = 0
        # Loop over all batches.
        for i, batches in enumerate(tqdm(data_loader)):
            for batch in batches:
                text_inputs = batch["text_tokens"].squeeze(0).to(device)
                audio_inputs = batch["audio_inputs"].squeeze(0).to(device)
                text_mask = batch["text_masks"].squeeze(0).to(device)
                audio_mask = batch["audio_masks"].squeeze(0).to(device)
                targets = batch["targets"].squeeze(0).to(device)
                batch_size = self.config.batch_size
                loss = 0.0
                outputs = model(text_inputs, text_mask, audio_inputs, audio_mask, batch_size)
                # targets = targets.unsqueeze(dim=1)
                # Compute the training loss.
                
                loss = self.criterion(
                    outputs, targets.long().to(device).view(-1))
                # print(f"output shape is {outputs.shape}")

                loss = self.criterion(outputs, targets)
                
                # 实现L1正则化
                if self.config.use_regularization == 'L1':
                    l1_weight = 0.5
                    l1_regularization = l1_weight * sum(param.abs().sum() for param in model.parameters())
                    loss += l1_regularization
                    
                # 实现L2正则化
                elif self.config.use_regularization == 'L2':
                    l2_weight = 0.5
                    l2_regularization = l2_weight * sum(param.pow(2).sum() for param in model.parameters())
                    # print("l2_regularization: ", l2_regularization)
                    loss += l2_regularization
                    
                else:
                    pass
                loss.backward()
                
                # optimizer.step()
                # optimizer.zero_grad()
                total_loss += loss.item()*text_inputs.size(0)
                input_size += text_inputs.size(0)
                
                

                # class_loss = 0.0
                # output_five, output_four, output_three, output_two = model(text_inputs, text_mask, audio_inputs, audio_mask, batch_size)
                # print("success")
                # output = {
                #     "five_class": output_five.unsqueeze(0),
                #     "four_class": output_four.unsqueeze(0),
                #     "three_class": output_three.unsqueeze(0),
                #     "two_class": output_two.unsqueeze(0)
                # }
                # targets = batch['targets']
                
                
                
                # for key in output.keys():
                #     loss = self.criterion(output[key], targets[key].to(device))
                #     class_loss += loss
                    
                # class_loss.backward()
                # print("loss backward success")
                # total_loss += class_loss.item()*text_inputs.size(0)
                # input_size += text_inputs.size(0)
                
                if (i+1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()  
                scheduler = self.get_scheduler(optimizer, self.scheduler_type)
                if scheduler is not None:
                    scheduler.step()
        final_loss = round(total_loss / input_size, 4)
        print("TRAIN >> loss: ", final_loss)
        return final_loss

        

    def do_test(self, model, data_loader, mode):
        model.eval()
        y_pred = []
        y_true = []
        
        # y_pred = {
        #     "five_class": [],
        #     "four_class": [],
        #     "three_class": [],
        #     "two_class": []
        # }
        
        # y_true = {
        #     "five_class": [],
        #     "four_class": [],
        #     "three_class": [],
        #     "two_class": []
        # }
        total_loss = 0
        input_size = 0
        with torch.no_grad():
            # Loop over all batches.
            for batches in tqdm(data_loader):
                for batch in batches:
                    text_inputs = batch["text_tokens"].squeeze(0).to(device)
                    audio_inputs = batch["audio_inputs"].squeeze(0).to(device)
                    text_mask = batch["text_masks"].squeeze(0).to(device)
                    audio_mask = batch["audio_masks"].squeeze(0).to(device)
                    targets = batch["targets"].squeeze(0).to(device)
                    batch_size = self.config.batch_size
                    # Predictions from 1 batch of data.
                    outputs = model(text_inputs, text_mask,
                                    audio_inputs, audio_mask, self.config.batch_size)
                    # targets = targets.unsqueeze(dim=1)
                    # Compute the training loss.
                    loss = 0.0
                    loss = self.criterion(
                        outputs, targets)
                    total_loss += float(loss.item()*text_inputs.size(0))
                    input_size += text_inputs.size(0)
                    
                    # class_loss = 0.0
                    # output_five, output_four, output_three, output_two = model(text_inputs, text_mask, audio_inputs, audio_mask, batch_size)
                    # output = {
                    #     "five_class": output_five.unsqueeze(0),
                    #     "four_class": output_four.unsqueeze(0),
                    #     "three_class": output_three.unsqueeze(0),
                    #     "two_class": output_two.unsqueeze(0)
                    # }
                    # targets = batch['targets']
                    # # 将target 所有的值都转换为cpu
                    # for key in targets.keys():
                    #     targets[key] = targets[key].cpu()
                                            
                    # for key in output.keys():
                    #     loss = self.criterion(output[key], targets[key].to(device))
                    #     class_loss += loss
                    #     y_pred[key].append(output[key].cpu())
                    #     y_true[key].append(targets[key].cpu())
                        
                    # total_loss += class_loss.item()*text_inputs.size(0)
                    # input_size += text_inputs.size(0)

                    # add predictions
                    y_pred.append(outputs)
                    y_true.append(targets)
            

        total_loss = round(total_loss / input_size, 4)
        print(mode+" >> loss: ", total_loss)
        eval_results = {}
        pred = torch.cat(y_pred,dim=0) 
        true = torch.cat(y_true,dim=0)
        
        results = self.metrics(pred, true)
        eval_results = results
        
        
        
        # y_pred_final = {
        #     "five_class": torch.cat(y_pred['five_class'], dim=0),
        #     "four_class": torch.cat(y_pred["four_class"], dim=0),
        #     "three_class": torch.cat(y_pred['three_class'],dim = 0),
        #     "two_class": torch.cat(y_pred["two_class"], dim=0)
        # }
        
        # y_true_final = {
        #     "five_class": torch.cat(y_true['five_class'], dim=0),
        #     "four_class": torch.cat(y_true["four_class"], dim=0),
        #     "three_class": torch.cat(y_true['three_class'], dim=0),
        #     "two_class": torch.cat(y_true["two_class"], dim=0)
        # }

        
        # for key in y_pred_final.keys():
        #     result = self.metrics(y_pred_final[key], y_true_final[key])
        #     # 分别加上各个类别的准确率和F1值
        #     for k in result.keys():
        #         eval_results[key + "_" + k] = result[k]

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
    for param in model.data2vec_model.feature_extractor.parameters():
            param.requires_grad = False
    # for param in model.hubert_model.feature_extractor.parameters():
    #     param.requires_grad = False

    trainer = ChTrainer(config)

    lowest_eval_loss = 100
    highest_eval_acc = 0
    epoch = 0
    best_epoch = 0
    
    # for epoch in range(large_epoch):
    #     print('---------------------EPOCH: ', epoch, '--------------------')
    #     total_loss_train = trainer.do_train(model, train_loader)
    #     writer_train.add_scalar('Loss/TRAIN', total_loss_train, epoch)
    #     print("TRAIN >> loss: ", total_loss_train)
    #     epoch += 1
        
    # test_results = trainer.do_test(model, test_loader, "TEST")
    # print('%s: >> ' %('TEST ') + dict_to_str(test_results))
    
    while True:
        epoch += 1
        print('---------------------EPOCH: ', epoch, '--------------------')
        total_loss_train = trainer.do_train(model, train_loader)
        writer_train.add_scalar('Loss/TRAIN', total_loss_train, epoch)
        
        eval_results = trainer.do_test(model, val_loader, "VAL")
        print('%s: >> ' %('VAL ') + dict_to_str(eval_results))
        # writer_val.add_scalar('Loss/VAL', eval_results['Loss'], epoch)
        # writer_val.add_scalar('Has0_acc_2/VAL', eval_results['Has0_acc_2'], epoch)
        # writer_val.add_scalar('Has0_F1_score/VAL', eval_results['Has0_F1_score'], epoch)
        # writer_val.add_scalar('Non0_acc_2/VAL', eval_results['Non0_acc_2'], epoch)
        # writer_val.add_scalar('Non0_F1_score/VAL', eval_results['Non0_F1_score'], epoch)
        # writer_val.add_scalar('Mult_acc_7/VAL', eval_results['Mult_acc_7'], epoch)
        # writer_val.add_scalar('Mult_acc_5/VAL', eval_results['Mult_acc_5'], epoch)
        # writer_val.add_scalar('MAE', eval_results['MAE'], epoch)
        
        writer_val.add_scalar('Mult_acc_7/VAL', eval_results['Mult_acc_7'], epoch)
        writer_val.add_scalar('F1_score_7/VAL', eval_results['F1_score_7'], epoch)
        
        # print(dict_to_str(eval_results['M_result']))
        # mode = "VAL"
        # total_loss_val = eval_results['Loss']
    #     acc2 = eval_results['M_result']['Non0_acc_2']
    #     Mult_acc_5 = eval_results['M_result']['Mult_acc_5']
    #     nni.report_intermediate_result(acc2)
    #     nni.report_intermediate_result(Mult_acc_5)
        
    # nni.report_final_result(acc2)
    # nni.report_final_result(Mult_acc_5)
        if eval_results['Loss']<lowest_eval_loss:
            lowest_eval_loss = eval_results['Loss']
            torch.save(model.state_dict(), config.model_save_path+f'{config.dataset_name}_loss.pth')
            best_epoch = epoch
        if eval_results['Mult_acc_7']>=highest_eval_acc:
            highest_eval_acc = eval_results['Mult_acc_7']
            torch.save(model.state_dict(), config.model_save_path+f'{config.dataset_name}_acc.pth')
        if epoch - best_epoch >= config.early_stop:
            break
    
    model.eval()
    model.load_state_dict(torch.load(config.model_save_path+f'{config.dataset_name}_acc.pth'))        
    test_results_loss = trainer.do_test(model, test_loader,"TEST")
    print('%s: >> ' %('TEST (highest val acc) ') + dict_to_str(test_results_loss))
    
    model.load_state_dict(torch.load(config.model_save_path+f'{config.dataset_name}_loss.pth'))
    test_results_acc = trainer.do_test(model, test_loader,"TEST")
    print('%s: >> ' %('TEST (lowest val loss) ') + dict_to_str(test_results_acc))
    
    
    # # 将结果存进对应的文件
    # with open('results/mosei/test_results.txt', 'w+') as f:
    #     f.write('TEST (val acc) ' + ', '.join(map(str, test_results_acc)) + '\n')
    #     f.write('TEST (val loss) ' + dict_to_str(test_results_loss) + '\n')
    
    writer_train.close()
    writer_val.close()

