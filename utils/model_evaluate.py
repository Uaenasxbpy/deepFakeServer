import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score

class ModelEvaluator:
    def __init__(self, model, test_loader, device):
        '''
        评估函数
        :param model: 导入模型
        :param test_loader: 导入测试集
        :param device: cuda or cpu
        '''
        self.model = model
        self.test_loader = test_loader
        self.device = device

    def evaluate(self):
        print('Evaluating model...')
        self.model.eval()
        y_true = []
        y_pred = []
        y_scores = []

        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                preds = outputs.argmax(dim=1)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
                y_scores.extend(outputs.cpu().numpy()[:, 1])
                total += labels.size(0)
                correct += (preds == labels).sum().item()

            print('Test Accuracy of the model on the test images: {} %'.format((correct / total) * 100))

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_scores = np.array(y_scores)

        self._plot_confusion_matrix(y_true, y_pred)
        self._plot_roc_curve(y_true, y_scores)
        self._print_metrics(y_true, y_pred)
        print('Evaluation finished!')

    def _plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        print("Confusion Matrix:")
        print(cm)
        cm_df = pd.DataFrame(cm, index=['Real', 'Fake'], columns=['Real', 'Fake'])

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')
        plt.show()

    def _plot_roc_curve(self, y_true, y_scores):
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        print("AUC:", roc_auc)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.savefig('roc_curve.png')
        plt.show()

    def _print_metrics(self, y_true, y_pred):
        recall = self._calculate_recall(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        print("Recall:", recall)
        print("F1 Score:", f1)

    def _calculate_recall(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        recall = cm[1, 1] / (cm[1, 0] + cm[1, 1])
        return recall
