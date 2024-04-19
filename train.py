import torch
import os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix, precision_recall_curve, \
    precision_score
from models import loss_function
from prettytable import PrettyTable
from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import r2_score,matthews_corrcoef
from scipy.stats import pearsonr

class Training(object):
    def __init__(self, model, optim, device, train_data, val_data, test_data, **config):
        self.model = model
        self.optim = optim
        self.device = device
        self.epochs = config["SETUP"]["MAX_EPOCH"]
        self.current_epoch = 0
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.batch_size = config["SETUP"]["BATCH_SIZE"]
        self.nb_training = len(self.train_data)
        self.step = 0

        self.val_best_model = None
        self.val_best_epoch = None
        self.val_best_auc = 0

        self.train_loss_epoch = []
        self.train_model_loss_epoch = []
        self.val_loss_epoch, self.val_auroc_epoch = [], []
        self.test_results = {}
        self.config = config
        self.output_dir = config["RESULT"]["PATH"]
        test_metric_header = ["Results at Best Model", "AUC", "AUPR", "Sensitivity", "Specificity", "ACC"]
        self.test_table = PrettyTable(test_metric_header)

    def train(self):
        float2str = lambda x: '%0.4f' % x
        for i in range(self.epochs):
            self.current_epoch += 1
            train_loss = self.train_pred()
            auroc, auprc, val_loss = self.test(trainType="val")
            self.val_auroc_epoch.append(auroc)
            if auroc >= self.val_best_auc:
                self.val_best_auc = auroc
                self.val_best_epoch = self.current_epoch
            print('Validation loss for the Epoch ' + str(self.current_epoch) + ' is ' + str(val_loss), " AUC "
                  + str(auroc) + " AUPR " + str(auprc))
        auroc, auprc, sensitivity, specificity, accuracy = self.test(
            trainType="test")
        test_lst = ["epoch " + str(self.val_best_epoch)] + list(map(float2str, [auroc, auprc, sensitivity, specificity,
                                                                            accuracy]))
        self.test_table.add_row(test_lst)
        print('Test dataset at Best Model of Epoch ' + str(self.val_best_epoch) + " AUC "
              + str(auroc) + " AUPR " + str(auprc) + " Sensitivity " + str(sensitivity) + " Specificity " +
              str(specificity) + " ACC " + str(accuracy))
        self.test_results["auroc"] = auroc
        self.test_results["auprc"] = auprc
        self.test_results["accuracy"] = accuracy
        self.test_results["sensitivity"] = sensitivity
        self.test_results["specificity"] = specificity
        self.test_results["best_epoch"] = self.val_best_epoch
        self.save_result()
        return self.test_results

    def save_result(self):
        test_prettytable_file = os.path.join(self.output_dir, "test_record.txt")
        with open(test_prettytable_file, 'w') as fp:
            fp.write(self.test_table.get_string())

    def train_pred(self):
        self.model.train()
        loss_epoch = 0
        num_batches = len(self.train_data)
        for i, (drug_Initial, protein_Initial, labels) in enumerate(tqdm(self.train_data)):
            self.step += 1
            drug_Initial, protein_Initial, labels = drug_Initial.to(self.device), protein_Initial.to(self.device), labels.float().to(
                self.device)
            self.optim.zero_grad()
            score, enc_mean, enc_std = self.model(drug_Initial, protein_Initial)
            n, loss = loss_function(score, labels, enc_mean, enc_std, 1e-3)
            loss.backward()
            self.optim.step()
            loss_epoch += loss.item()
        loss_epoch = loss_epoch / num_batches

        print('Training loss for the epoch ' + str(self.current_epoch) + ' is ' + str(loss_epoch))
        return loss_epoch

    def test(self, trainType="test"):
        test_loss = 0
        rel_label, pred = [], []
        if trainType == "test":
            data_loader = self.test_data
        elif trainType == "val":
            data_loader = self.val_data
        else:
            raise ValueError(f"Error key value {trainType}")
        num_batches = len(data_loader)
        with torch.no_grad():
            self.model.eval()
            for i, (drug_Initial, protein_Initial, labels) in enumerate(data_loader):
                drug_Initial, protein_Initial, labels = drug_Initial.to(self.device), protein_Initial.to(self.device), labels.float().to(
                    self.device)
                if trainType == "val":
                    prob, vec_mean, vec_cov = self.model(drug_Initial, protein_Initial)
                elif trainType == "test":
                    prob, vec_mean, vec_cov = self.model(drug_Initial, protein_Initial)
                    torch.set_printoptions(threshold=float('inf'))

                n, loss = loss_function(prob, labels, vec_mean, vec_cov, 1e-3)

                test_loss += loss.item()
                rel_label = rel_label + labels.to("cpu").tolist()
                pred = pred + n.to("cpu").tolist()

        auroc = roc_auc_score(rel_label, pred)
        auprc = average_precision_score(rel_label, pred)
        test_loss = test_loss / num_batches

        if trainType == "test":
            # mse = metrics.mean_squared_error(y_label, y_pred)
            # rmse = np.sqrt(metrics.mean_squared_error(y_label, y_pred))
            # r2 = r2_score(y_label, y_pred)
            # pc = pearsonr(y_label, y_pred)

            fpr, tpr, thresholds = roc_curve(rel_label, pred)
            prec, recall, _ = precision_recall_curve(rel_label, pred)
            precision = tpr / (tpr + fpr)
            f1 = 2 * precision * tpr / (tpr + precision + 0.00001)
            thred_optim = thresholds[5:][np.argmax(f1[5:])]
            pred_s = [1 if i else 0 for i in (pred >= thred_optim)]
            cm1 = confusion_matrix(rel_label, pred_s)
            accuracy = (cm1[0, 0] + cm1[1, 1]) / sum(sum(cm1))
            sensitivity = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
            specificity = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
            # precision1 = precision_score(y_label, y_pred_s)
            return auroc, auprc, sensitivity, specificity, accuracy
        else:
            return auroc, auprc, test_loss
