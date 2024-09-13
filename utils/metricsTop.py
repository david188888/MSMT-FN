import numpy as np
from sklearn.metrics import accuracy_score, f1_score

#---------------------------------------------------
# The codes below are from https://github.com/thuiar/MMSA
# They are used to perform model evaluation
#----------------------------------------------------

__all__ = ['MetricsTop']

class MetricsTop():
    def __init__(self):
            self.metrics_dict = self.__classification_acc

    def __multiclass_acc(self, y_pred, y_true):
        """
        Compute the multiclass accuracy w.r.t. groundtruth

        :param preds: Float array representing the predictions, dimension (N,)
        :param truths: Float/int array representing the groundtruth classes, dimension (N,)
        :return: Classification accuracy
        """
        return np.sum(np.round(y_pred) == np.round(y_true)) / float(len(y_true))
    
    
    
    def __meld_classification(self, y_pred, y_true):
        '''
        计算meld数据集的分类准确率, 先计算七分类准确率，再计算每个类别的准确率，最后计算加权F1分数
        '''
        
        y_pred = y_pred.cpu().detach().numpy()
        y_true = y_true.cpu().detach().numpy()
        
        y_pred = np.argmax(y_pred, axis=1)
        y_true = y_true.flatten()
        
        # 计算七分类准确率
        Mult_acc_7 = self.__multiclass_acc(y_pred, y_true)
        emotion_labels = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'joy', 4: 'neutral', 5: 'sadness', 6: 'surprise'}
        # 计算每个类别的准确率
        class_acc = {}
        for i in range(7):
            idx = np.where(y_true == i)
            class_acc[emotion_labels[i]] = round(self.__multiclass_acc(y_pred[idx], y_true[idx]), 4)
            
        # 计算加权F1分数
        f_score = f1_score(y_pred=y_pred, y_true=y_true, average='weighted')
        
        eval_results = {
            "Mult_acc_7": round(Mult_acc_7, 4),
            "F1_score_7": round(f_score, 4),
            "Class_acc": class_acc
        }
        
        return eval_results
    
    
    def qa_data_classification(self, y_pred, y_true):
        '''
        计算qa数据集的分类准确率
        '''
        
        y_pred = y_pred.cpu().detach().numpy()
        y_true = y_true.cpu().detach().numpy()
        
        y_pred = np.argmax(y_pred, axis=1)
        y_true = y_true.flatten()
        
        
        
        # 计算五分类准确率
        Mult_acc_5 = self.__multiclass_acc(y_pred, y_true)
        
        # 计算四分类准确率，0为一类，1为一类，2为一类，3，4为一类
        y_pred_4 = y_pred.copy()
        y_true_4 = y_true.copy()
        
        y_pred_4[y_pred_4 == 3] = 3
        y_pred_4[y_pred_4 == 4] = 3
        
        y_true_4[y_true_4 == 3] = 3
        y_true_4[y_true_4 == 4] = 3

        
        Mult_acc_4 = self.__multiclass_acc(y_pred_4, y_true_4)
        
        # 计算三分类准确率 0为一类，1，2为一类，3，4为一类
        y_pred_3 = y_pred.copy()
        y_true_3 = y_true.copy()
        
        y_pred_3[y_pred_3 == 0] = 0
        y_pred_3[y_pred_3 == 1] = 1
        y_pred_3[y_pred_3 == 2] = 1
        y_pred_3[y_pred_3 == 3] = 2
        y_pred_3[y_pred_3 == 4] = 2
        
        y_true_3[y_true_3 == 0] = 0
        y_true_3[y_true_3 == 1] = 1
        y_true_3[y_true_3 == 2] = 1
        y_true_3[y_true_3 == 3] = 2
        y_true_3[y_true_3 == 4] = 2
        
        Mult_acc_3 = self.__multiclass_acc(y_pred_3, y_true_3)
        
        
        # 计算二分类准确率 0，1为一类，2，3，4为一类
        y_pred_2 = y_pred.copy()
        y_true_2 = y_true.copy()
        
        y_pred_2[y_pred_2 == 0] = 0
        y_pred_2[y_pred_2 == 1] = 0
        y_pred_2[y_pred_2 == 2] = 1
        y_pred_2[y_pred_2 == 3] = 1
        y_pred_2[y_pred_2 == 4] = 1
        
        y_true_2[y_true_2 == 0] = 0
        y_true_2[y_true_2 == 1] = 0
        y_true_2[y_true_2 == 2] = 1
        y_true_2[y_true_2 == 3] = 1
        y_true_2[y_true_2 == 4] = 1
        Mult_acc_2 = self.__multiclass_acc(y_pred_2, y_true_2)
        
        eval_results = {
            "Mult_acc_5": round(Mult_acc_5, 4),
            "Mult_acc_4": round(Mult_acc_4, 4),
            "Mult_acc_3": round(Mult_acc_3, 4),
            "Mult_acc_2": round(Mult_acc_2, 4)
        }
        
        return eval_results
        
        
        
        
        
    
    
    def __mosi_classification(self, y_pred, y_true):
        test_preds = y_pred.cpu().detach().numpy()
        test_truth = y_true.cpu().detach().numpy()
        test_preds_label = np.argmax(test_preds, axis=1)
        test_truth_label = test_truth.flatten()
        
        
        # 计算mosei三分类准确率
        Mult_acc_2 = accuracy_score(y_pred=test_preds_label, y_true=test_truth_label)
        # 计算 mosei 三分类F1 分数
        f_score = f1_score(y_pred=test_preds_label,y_true=test_truth_label, average='weighted')
        
        # #计算mosei 二分类准确率, 中性或者不是中性
        # y_pred_binary = np.array([[v[0], v[2]] for v in test_preds])
        # y_pred_2 = np.argmax(y_pred_binary, axis=1)
        # y_true_2 = []
        # for v in test_truth:
        #     if v <= 1:
        #         y_true_2.append(0)
        #     else:
        #         y_true_2.append(1)
        # y_true_2 = np.array(y_true_2)
        # Has0_acc_2 = accuracy_score(y_true_2, y_pred_2)
        # Has0_F1_score = f1_score(y_true_2, y_pred_2, average='weighted')
        
        # # mosi 的正或者负
        # non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 1])

        # y_pred_2 = test_preds[non_zeros]
        # y_pred_2 = np.argmax(y_pred_2, axis=1)
        # y_true_2 = test_truth[non_zeros]
        # Non0_acc_2 = accuracy_score(y_pred_2, y_true_2)
        # Non0_F1_score = f1_score(y_true_2, y_pred_2, average='weighted')
        
        eval_results = {
            # "Has0_acc_2":  round(Has0_acc_2, 4),
            # "Has0_F1_score": round(Has0_F1_score, 4),
            # "Non0_acc_2":  round(Non0_acc_2, 4),
            # "Non0_F1_score": round(Non0_F1_score, 4),
            "Mult_acc_2": round(Mult_acc_2, 4),
            "F1_score": round(f_score, 4)
        }
        return eval_results
    
    def __eval_mosei_regression(self, y_pred, y_true, exclude_zero=False):
        test_preds = y_pred.view(-1).cpu().detach().numpy()
        test_truth = y_true.view(-1).cpu().detach().numpy()

        test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
        test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
        test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
        test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)
        test_preds_a3 = np.clip(test_preds, a_min=-1., a_max=1.)
        test_truth_a3 = np.clip(test_truth, a_min=-1., a_max=1.)


        mae = np.mean(np.absolute(test_preds - test_truth)).astype(np.float64)   # Average L1 distance between preds and truths
        corr = np.corrcoef(test_preds, test_truth)[0][1]
        mult_a7 = self.__multiclass_acc(test_preds_a7, test_truth_a7)
        mult_a5 = self.__multiclass_acc(test_preds_a5, test_truth_a5)
        mult_a3 = self.__multiclass_acc(test_preds_a3, test_truth_a3)
        
        non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])
        non_zeros_binary_truth = (test_truth[non_zeros] > 0)
        non_zeros_binary_preds = (test_preds[non_zeros] > 0)

        non_zeros_acc2 = accuracy_score(non_zeros_binary_preds, non_zeros_binary_truth)
        non_zeros_f1_score = f1_score(non_zeros_binary_truth, non_zeros_binary_preds, average='weighted')

        binary_truth = (test_truth >= 0)
        binary_preds = (test_preds >= 0)
        acc2 = accuracy_score(binary_preds, binary_truth)
        f_score = f1_score(binary_truth, binary_preds, average='weighted')
        
        print(f"test_preds{test_preds[:5]}")
        print(f"test_truth{test_truth[:5]}")
        
        print(f"binary pred{binary_preds[:5]}")
        print(f"binary truth{binary_truth[:5]}")
        
        eval_results = {
            "Has0_acc_2":  round(acc2, 4),
            "Has0_F1_score": round(f_score, 4),
            "Non0_acc_2":  round(non_zeros_acc2, 4),
            "Non0_F1_score": round(non_zeros_f1_score, 4),
            "Mult_acc_5": round(mult_a5, 4),
            "Mult_acc_7": round(mult_a7, 4),
            "MAE": round(mae, 4),
            "Corr": round(corr, 4)
        }
        return eval_results
    
    def __eval_sims_regression(self, y_pred, y_true):
        test_preds = y_pred.view(-1).cpu().detach().numpy()
        test_truth = y_true.view(-1).cpu().detach().numpy()
        test_preds = np.clip(test_preds, a_min=-1., a_max=1.)
        test_truth = np.clip(test_truth, a_min=-1., a_max=1.)

        # two classes{[-1.0, 0.0], (0.0, 1.0]}
        ms_2 = [-1.01, 0.0, 1.01]
        test_preds_a2 = test_preds.copy()
        test_truth_a2 = test_truth.copy()
        for i in range(2):
            test_preds_a2[np.logical_and(test_preds > ms_2[i], test_preds <= ms_2[i+1])] = i
        for i in range(2):
            test_truth_a2[np.logical_and(test_truth > ms_2[i], test_truth <= ms_2[i+1])] = i

        # three classes{[-1.0, -0.1], (-0.1, 0.1], (0.1, 1.0]}
        ms_3 = [-1.01, -0.1, 0.1, 1.01]
        test_preds_a3 = test_preds.copy()
        test_truth_a3 = test_truth.copy()
        for i in range(3):
            test_preds_a3[np.logical_and(test_preds > ms_3[i], test_preds <= ms_3[i+1])] = i
        for i in range(3):
            test_truth_a3[np.logical_and(test_truth > ms_3[i], test_truth <= ms_3[i+1])] = i
        
        # five classes{[-1.0, -0.7], (-0.7, -0.1], (-0.1, 0.1], (0.1, 0.7], (0.7, 1.0]}
        ms_5 = [-1.01, -0.7, -0.1, 0.1, 0.7, 1.01]
        test_preds_a5 = test_preds.copy()
        test_truth_a5 = test_truth.copy()
        for i in range(5):
            test_preds_a5[np.logical_and(test_preds > ms_5[i], test_preds <= ms_5[i+1])] = i
        for i in range(5):
            test_truth_a5[np.logical_and(test_truth > ms_5[i], test_truth <= ms_5[i+1])] = i
 
        mae = np.mean(np.absolute(test_preds - test_truth)).astype(np.float64)   # Average L1 distance between preds and truths
        corr = np.corrcoef(test_preds, test_truth)[0][1]
        mult_a2 = self.__multiclass_acc(test_preds_a2, test_truth_a2)
        mult_a3 = self.__multiclass_acc(test_preds_a3, test_truth_a3)
        mult_a5 = self.__multiclass_acc(test_preds_a5, test_truth_a5)
        f_score = f1_score(test_truth_a2, test_preds_a2, average='weighted')

        eval_results = {
            "Mult_acc_2": round(mult_a2, 4),
            "Mult_acc_3": round(mult_a3, 4),
            "Mult_acc_5": round(mult_a5, 4),
            "F1_score": round(f_score, 4),
            "MAE": round(mae, 4),
            "Corr": round(corr, 4), # Correlation Coefficient
        }
        return eval_results

    def __classification_acc(self, y_pred, y_true):    
        test_preds = y_pred.cpu().detach().numpy()
        test_truth = y_true.cpu().detach().numpy()

        
        test_preds_label = np.argmax(test_preds, axis=1)
        test_truth_label = test_truth.flatten()
        
        # print(f"test_preds: {test_preds[:5]}")
    
        
        # eval_results = {
        #     "Acc": round(accuracy_score(y_pred=test_preds_label, y_true=test_truth_label), 4),
        #     "F1": round(f1_score(y_pred=test_preds_label, y_true=test_truth_label, average='weighted'), 4)
        # }   
        # eval_results = self.qa_data_classification(y_pred, y_true)
        # eval_results = self.__mosi_classification(y_pred, y_true)
        eval_results = self.__eval_mosei_regression(y_pred, y_true)
        # eval_results = self.__eval_sims_regression(y_pred, y_true)
        # eval_results = self.__meld_classification(y_pred, y_true)
        return eval_results
    
    
    def getMetics(self):
        return self.metrics_dict
