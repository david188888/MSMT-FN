import torch
from torch import nn
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
from tqdm import tqdm
from utils.metricsTop import MetricsTop
from utils.ch_model import rob_hub_cc, rob_hub_cme
import random
import numpy as np
from utils.data_loader import data_loader
from torch.utils.tensorboard import SummaryWriter



from datetime import datetime
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
...
train_log_dir = 'logs/train/' + TIMESTAMP
test_log_dir = 'logs/test/'   + TIMESTAMP
val_log_dir = 'logs/val/' + TIMESTAMP


writer_train = SummaryWriter(train_log_dir)
writer_val = SummaryWriter(val_log_dir)
writer_test = SummaryWriter(test_log_dir)


# global variable
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


def dict_to_str(src_dict):
    dst_str = ""
    for key in src_dict.keys():
        dst_str += " %s: %.4f " % (key, src_dict[key])
    return dst_str


class ChConfig(object):
    """Configuration class to store the configurations of training.
    """

    def __init__(self,
                 train_mode='regression',
                 loss_weights={
                     'M': 1,
                     'T': 1,
                     'A': 1,
                     'V': 1,

                 },
                 model_save_path='checkpoint/',
                 learning_rate=1e-5,
                 epochs=50,
                 dataset_name='sims',
                 early_stop=8,
                 seed=0,
                 dropout=0.3,
                 model='cme',
                 audio_feature='raw',
                 batch_size=4,
                 cme_version='v1',
                 tasks='MTA',
                 num_hidden_layers=3
                 ):

        self.train_mode = train_mode
        self.loss_weights = loss_weights
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dataset_name = dataset_name
        self.model_save_path = model_save_path
        self.early_stop = early_stop
        self.seed = seed
        self.dropout = dropout
        self.model = model
        self.audio_feature = audio_feature
        self.batch_size = batch_size
        self.cme_version = cme_version
        self.tasks = tasks
        self.num_hidden_layers = num_hidden_layers


class ChTrainer():
    def __init__(self, config):

        self.config = config
        self.tasks = config.tasks
        self.criterion = nn.L1Loss() if config.train_mode == 'regression' else nn.CrossEntropyLoss()
        self.metrics = MetricsTop(
            config.train_mode).getMetics(config.dataset_name)

    def do_train(self, model, data_loader):
        model.train()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.config.learning_rate)
        train_loss = {
            'M': 0,
        }
        total_loss = 0
        input_size = 0
        # Loop over all batches.
        for batch in tqdm(data_loader):
            text_inputs = batch["text_tokens"].to(device)
            audio_inputs = batch["audio_inputs"].to(device)
            text_mask = batch["text_masks"].to(device)
            audio_mask = batch["audio_masks"].to(device)
            targets = batch["targets"]

            # To zero out the gradients.
            optimizer.zero_grad()

            outputs = model(text_inputs, text_mask, audio_inputs, audio_mask)

            # Compute the training loss.
            loss = 0.0

            m = self.tasks[0]
            sub_loss = self.config.loss_weights[m] * self.criterion(
                outputs[m], targets[m].to(device).view(-1, 1))
            loss += sub_loss

            total_loss += loss.item()*text_inputs.size(0)
            input_size += text_inputs.size(0)

            loss.backward()
            optimizer.step()

#         for m in self.tasks:
#             train_loss[m] = round(train_loss[m] / len(data_loader.dataset), 4)
        total_loss = round(total_loss / input_size, 4)
#         print('TRAIN'+" >> loss: ",total_loss, "   M_loss: ", train_loss['M'], "  T_loss: ", train_loss['T'], "  A_loss: ", train_loss['A'])

        writer_train.add_scalar('Loss/Train', total_loss, self.config.epochs)
        
        return total_loss

    def do_test(self, model, data_loader, mode):

        # Put the model in training mode.
        model.eval()
        y_pred = {'M': []}
        y_true = {'M': []}
        total_loss = 0
        val_loss = {
            'M': 0,
        }
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
                                audio_inputs, audio_mask)

                # Compute the training loss.
                loss = 0.0
                m = self.tasks[0]
                sub_loss = self.config.loss_weights[m] * self.criterion(
                    outputs[m], targets[m].to(device).view(-1, 1))
                loss += sub_loss
                val_loss[m] += sub_loss.item()*text_inputs.size(0)
                total_loss += loss.item()*text_inputs.size(0)
                input_size += text_inputs.size(0)

                # add predictions
                y_pred[m].append(outputs[m].cpu())
                y_true[m].append(targets[m].cpu())

        val_loss[m] = round(val_loss[m] / input_size, 4)

        total_loss = round(total_loss / input_size, 4)
        print(mode+" >> loss: ", total_loss, "   M_loss: ", val_loss['M'])

        eval_results = {}
        pred, true = torch.cat(y_pred[m]), torch.cat(y_true[m])
        results = self.metrics(pred, true)
        print('%s: >> ' % (m) + dict_to_str(results))
        eval_results[m] = results

        eval_results = eval_results[self.tasks[0]]
        eval_results['Loss'] = total_loss
        
        if mode == "VAL":
                    writer_val.add_scalar('Loss/'+mode, total_loss, self.config.epochs)
                    writer_val.add_scalar('Acc2/'+mode, eval_results['Mult_acc_2'], self.config.epochs)
                    writer_val.add_scalar('F1/'+mode, eval_results['F1_score'], self.config.epochs)
                    writer_val.add_scalar('ACC3/'+mode, eval_results['Mult_acc_3'], self.config.epochs)
                    writer_val.add_scalar('MAE/'+mode, eval_results['MAE'], self.config.epochs)
                    writer_val.add_scalar('Corr/'+mode, eval_results['Corr'], self.config.epochs)
                    
        elif mode == "TEST":
                    writer_test.add_scalar('Loss/'+mode, total_loss, self.config.epochs)
                    writer_test.add_scalar('Acc2/'+mode, eval_results['Mult_acc_2'], self.config.epochs)
                    writer_test.add_scalar('F1/'+mode, eval_results['F1_score'], self.config.epochs)
                    writer_test.add_scalar('ACC3/'+mode, eval_results['Mult_acc_3'], self.config.epochs)
                    writer_test.add_scalar('MAE/'+mode, eval_results['MAE'], self.config.epochs)
                    writer_test.add_scalar('Corr/'+mode, eval_results['Corr'], self.config.epochs)
        

        return eval_results


def ChRun(config):
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.backends.cudnn.deterministic = True

    train_loader, test_loader, val_loader = data_loader(
        config.batch_size)

    if config.model == 'cc':
        model = rob_hub_cc(config).to(device)
    elif config.model == 'cme':
        model = rob_hub_cme(config).to(device)
    for param in model.hubert_model.feature_extractor.parameters():
        param.requires_grad = False

    trainer = ChTrainer(config)

    lowest_eval_loss = 100
    highest_eval_acc = 0
    epoch = 0
    best_epoch = 0
    while True:
        print('---------------------EPOCH: ', epoch, '--------------------')
        epoch += 1
        trainer.do_train(model, train_loader)
        eval_results = trainer.do_test(model, val_loader, "VAL")
#         test_results = trainer.do_test(model, test_loader,"TEST")
        if eval_results['Loss'] < lowest_eval_loss:
            lowest_eval_loss = eval_results['Loss']
            torch.save(model.state_dict(), config.model_save_path+'loss.pth')
            best_epoch = epoch
        if eval_results['Mult_acc_2'] >= highest_eval_acc:
            highest_eval_acc = eval_results['Mult_acc_2']
            torch.save(model.state_dict(), config.model_save_path+'acc.pth')
        if epoch - best_epoch >= config.early_stop:
            break
    model.load_state_dict(torch.load(config.model_save_path+'acc.pth'))
    test_results_loss = trainer.do_test(model, test_loader, "TEST")
    print('%s: >> ' % ('TEST (highest val acc) ') +
          dict_to_str(test_results_loss))

    model.load_state_dict(torch.load(config.model_save_path+'loss.pth'))
    test_results_acc = trainer.do_test(model, test_loader, "TEST")
    print('%s: >> ' % ('TEST (lowest val loss) ') +
          dict_to_str(test_results_acc))
    
    
    # 将结果存进文件
    with open('logs/'+TIMESTAMP+'results.txt', 'w') as f:
        f.write('TEST (highest val acc) '+dict_to_str(test_results_loss)+'\n')
        f.write('TEST (lowest val loss) '+dict_to_str(test_results_acc)+'\n')
        
    

    writer_test.close()
    writer_val.close()
