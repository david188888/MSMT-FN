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

    def __classification_acc(self, y_pred, y_true):    
        test_preds = y_pred.cpu().detach().numpy()
        test_truth = y_true.cpu().detach().numpy()
        
        print(f"test_preds: {test_preds}")
        
        test_preds_label = np.argmax(test_preds, axis=1)
        test_truth_label = test_truth.flatten()
        
        print(f"test_preds_label: {test_preds_label}")
        
        
        mae = np.mean(np.absolute(test_preds_label - test_truth_label)).astype(np.float64)

        # 计算相关系数
        corr = np.corrcoef(test_preds_label, test_truth_label)[0][1]

        # 计算多分类准确率
        mult_a5 = self.__multiclass_acc(test_preds_label, test_truth_label)

        # 计算 F1 分数
        f_score = f1_score(test_truth_label, test_preds_label, average='weighted')
            # 计算二分类准确度
        test_truth_binary = (test_truth_label < 2).astype(int)  # A, B -> 1; C, D, E -> 0
        test_preds_binary = (test_preds_label < 2).astype(int)
        mult_a2 = accuracy_score(test_truth_binary, test_preds_binary)
        
        # 计算四分类准确度
        def map_to_four_class(label):
            if label == 3 or label == 4:
                return 3
            elif label == 0:
                return 0
            elif label == 1:
                return 1
            elif label == 2:
                return 2
        
    # 计算三分类准确度
        def map_to_three_class(label):
            if label == 0:
                return 0  # A -> 0
            elif label == 1 or label == 2:
                return 1  # B, C -> 1
            elif label == 3 or label == 4:
                return 2  # D, E -> 2
        test_truth_four_class = np.array([map_to_four_class(label) for label in test_truth_label])
        test_preds_four_class = np.array([map_to_four_class(label) for label in test_preds_label])
        mult_a4 = accuracy_score(test_truth_four_class, test_preds_four_class)
        
        test_truth_three_class = np.array([map_to_three_class(label) for label in test_truth_label])
        test_preds_three_class = np.array([map_to_three_class(label) for label in test_preds_label])
        mult_a3 = accuracy_score(test_truth_three_class, test_preds_three_class)
        
        eval_results = {
                "Mult_acc_5": mult_a5,
                "Mult_acc_2": mult_a2,
                "Mult_acc_4": mult_a4,
                "Mult_acc_3": mult_a3,
                "F1_score": f_score,
                "MAE": mae,
                "Corr": corr # Correlation Coefficient
            }
        return eval_results
    
    
    def getMetics(self):
        return self.metrics_dict
