import torch
import pandas as pd
import os
import torch.nn as nn
import seaborn as sns
import numpy as np
from sklearn.metrics import precision_score, recall_score
from sklearn import metrics
import matplotlib.pyplot as plt

def read_file(path):
    rfile = pd.read_csv(path, header=None, names=None).reset_index(drop=True)
    return rfile

def process_data(df, columns_new):
    df_new = pd.DataFrame(df, columns=columns_new)
    df_new.replace({'FALSE': 0, 'TRUE': 1}, inplace=True)
    return df_new

def write_data(data, path, file_type = 'csv'):
    if file_type == 'csv':
        data.to_csv(path, encoding='utf-8', index=False, header=False, mode='x')

def get_label_pred_ddi_binary(model, val_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pred, label, prob = [], [], []
    total_loss = 0
    criterion = nn.BCELoss()

    model.eval()
    with torch.no_grad():
        for i, batch_data in enumerate(val_loader):
            batch_data = batch_data.to(device)
            outputs, logits = model(batch_data)
            batch_data.y = batch_data.y.unsqueeze(1).to(torch.float32)
            outputs = outputs.to(torch.float32)
            valid_loss = criterion(outputs, batch_data.y)
            total_loss += valid_loss.item()
            prob.extend(outputs.detach().cpu().numpy().flatten())
            pred.extend(torch.round(outputs).detach().cpu().numpy().flatten())
            label.extend(batch_data.y.detach().cpu().numpy().flatten())

    return np.array(label), np.array(pred), np.array(prob), total_loss

def sk_p_r(label, pred):
    # when "true positive + false positive == 0", precision is undefined.
    ma_p = precision_score(np.array(label), np.array(pred), average='macro', zero_division=1)
    # when "true positive + false negative == 0", recall is undefined.
    ma_r = recall_score(np.array(label), np.array(pred), average='macro', zero_division=1)
    ma_f1 = (2 * ma_p * ma_r) / (ma_p + ma_r)
    return ma_f1


def logging(log_file, logs):
    logfile = open(
        log_file, 'a+'
    )
    logfile.write(logs)
    logfile.close()

def get_logging(log, log_filename, eval='training'):

    logfile = open(
        log_filename, 'a+'
    )

    if eval == 'training':
        logfile.write("== == training phrase == == " + "\n")
    if eval == 'test':
        logfile.write("== == test phrase == == " + "\n")

    logfile.write(log)
    logfile.close()

def draw_loss_curve(train_losses, valid_losses, path):
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epochs', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.title('Loss Curve', fontsize=20)
    plt.legend(fontsize=18)
    if os.path.exists(path + 'loss.png'):
        os.remove(path + 'loss.png')
        plt.savefig(path + 'loss.png')
    else:
        plt.savefig(path + 'loss.png')
    #plt.show()
    plt.clf() # clear

def draw_roc_pr_curve(label_test, prob_test, path):
    # Plot AUC-ROC curve
    fpr, tpr, thresholds = metrics.roc_curve(label_test, prob_test, pos_label=1)
    random_probs = [0 for i in range(len(label_test))]
    p_fpr, p_tpr, _ = metrics.roc_curve(label_test, random_probs, pos_label=1)
    roc_auc_score_test = metrics.roc_auc_score(label_test, prob_test)
    plt.style.use('seaborn')
    plt.xlim(0, None)
    plt.ylim(0, 1.1)
    plt.plot(fpr, tpr, linestyle='-', color='orange', label='ROC curve (area = %0.3f)' % roc_auc_score_test)
    plt.plot(p_fpr, p_tpr, linestyle='--', color='orange')
    plt.title('ROC Curve', fontsize=20)
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.legend(fontsize=18)
    if os.path.exists(path + 'ROC.png'):
        os.remove(path + 'ROC.png')
        plt.savefig(path + 'ROC.png')
    else:
        plt.savefig(path + 'ROC.png')
    plt.clf()

    # Print PR-ROC curve
    precision, recall, thresholds = metrics.precision_recall_curve(label_test, prob_test)
    pr_auc_score_test = metrics.average_precision_score(label_test, prob_test)
    plt.xlim(0, None)
    plt.ylim(0, 1.1)
    plt.plot([0, 1], [1, 0], color='blue', linestyle='--')
    plt.plot(recall, precision, linestyle='-', color='blue', label='PR curve (area = %0.3f)' % pr_auc_score_test)
    plt.title('PR Curve', fontsize=20)
    plt.xlabel('Recall', fontsize=18)
    plt.ylabel('Precision', fontsize=18)
    plt.legend(fontsize=18)
    if os.path.exists(path + 'PR.png'):
        os.remove(path + 'PR.png')
        plt.savefig(path + 'PR.png')
    else:
        plt.savefig(path + 'PR.png')
    plt.clf()

def draw_confusion_matrix(config, cf_matrix, path):
    xlables = [str(_) for _ in range(0, config.num_class)]
    ylables = [str(_) for _ in range(0, config.num_class)]
    sns.set(font_scale=1.5)
    if config.model_name == "biosnap":
        sns.heatmap(cf_matrix, annot=True, fmt="d", cmap='Blues', xticklabels=xlables, yticklabels=ylables)
    elif config.model_name == "drugbank":
        sns.heatmap(cf_matrix, annot=True, fmt="d", cmap='Oranges', xticklabels=xlables, yticklabels=ylables)
    plt.ylabel("True Class", fontsize=18)
    plt.xlabel("Predicted Class", fontsize=18)
    plt.tight_layout()
    if os.path.exists(path + 'Confusion Matrix.png'):
        os.remove(path + 'Confusion Matrix.png')
        plt.savefig(path + 'Confusion Matrix.png')
    else:
        plt.savefig(path + 'Confusion Matrix.png')
    plt.close()

