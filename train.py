import datetime
import torch
import sklearn
import torch.nn as nn
import logging
from tqdm import tqdm
from time import strftime

from sklearn.model_selection import KFold, train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from torch_geometric.loader import DataLoader
from dataset import DDIDataset
from model import D1Model
from utils import *

import os


def train_eval(config):
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Define DDI dataset and dataloader~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    dataset = DDIDataset(root=config.train_root, path=config.train_path)
    train_dataset, valid_dataset = train_test_split(dataset, test_size=0.2, random_state=3, shuffle=True)
    test_dataset = DDIDataset(root=config.test_root, path=config.test_path)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True, follow_batch=['pos1', 'pos2'])
    val_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True, follow_batch=['pos1', 'pos2'])
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True, follow_batch=['pos1', 'pos2'])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    model = D1Model(config)
    model.to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)

    # ============================Training Process==========================================
    train_losses, valid_losses = [], []
    nowtime = strftime('%Y-%m-%d-%H:%M:%S')
    # make a saved path directory
    if not os.path.exists(config.saved_root):
        os.makedirs(config.saved_root)
        print("Directory ", config.saved_root, " Created ")
    else:
        print("Directory ", config.saved_root, " already exists")
    # in saved path directory, make a nowtime directory
    if not os.path.exists(config.saved_root + nowtime):
        os.makedirs(config.saved_root + nowtime)
        print("Directory ", config.saved_root + nowtime, " Created ")
    else:
        print("Directory ", config.saved_root + nowtime, " already exists")

    saved_path = config.saved_root + nowtime + '/'
    log_filename = '{0}{1}.log'.format(saved_path, nowtime)

    # ======================================================================

    print('-----Training-----')
    starttime = datetime.datetime.now()
    last_epoch_time = starttime
    best_epoch = 0
    model.best_metric = -1.0
    model.best_ma_f1 = -1.0
    model.best_roc_auc = -1.0
    model.best_pr_auc = -1.0
    criterion = nn.BCELoss()

    for epoch in range(config.epochs):
        endtime = datetime.datetime.now()
        print('total run time: ', endtime - starttime)
        print('last epoch run time: ', endtime - last_epoch_time)
        last_epoch_time = endtime
        model.train()
        total_loss = 0

        for i, batch_data in enumerate(tqdm(train_loader)):
            batch_data = batch_data.to(config.device)
            outputs, hidden_states = model(batch_data)
            batch_data.y = batch_data.y.unsqueeze(1)
            batch_data.y = batch_data.y.to(torch.float32)
            outputs = outputs.to(torch.float32)
            train_loss = criterion(outputs, batch_data.y)
            total_loss += train_loss.item()
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        scheduler.step()

        tr_loss = total_loss / (len(train_loader) - 1)
        train_losses.append(tr_loss)

        label, pred, prob, total_loss = get_label_pred_ddi_binary(model, val_loader)
        va_loss = total_loss / (len(val_loader) - 1)
        valid_losses.append(va_loss)

        cls_report = metrics.classification_report(label, pred, target_names=['class 0', 'class 1'])

        roc_auc_score = metrics.roc_auc_score(label, prob)
        if roc_auc_score > model.best_roc_auc:
            model.best_roc_auc = roc_auc_score

        pr_auc_score = metrics.average_precision_score(label, prob)
        if pr_auc_score > model.best_pr_auc:
            model.best_pr_auc = pr_auc_score

        f1 = sk_p_r(label, pred)
        if f1 > model.best_ma_f1:
            model.best_ma_f1 = f1
            best_epoch = epoch

        acc = metrics.accuracy_score(label, pred)

        if acc > model.best_metric:
            model.best_metric = acc

        print(f"epoch:{epoch}  train_loss:{tr_loss}")
        print(f"valid_loss:{va_loss}")
        print(cls_report)
        print('roc_auc:', roc_auc_score)
        print('pr_auc:', pr_auc_score)
        print('macro f1:', f1)
        print('acc:', acc)

        # ======================================================================
        log = f"epoch:{epoch}  train_loss:{tr_loss}" + "\n" + f"valid_loss:{va_loss}" +  "\n" + \
              f"roc_auc:{roc_auc_score}" + "\n" + f"pr_auc:{pr_auc_score}" + "\n" +\
              f"macro f1: {f1}" + "\n" + f"acc:{acc}" + "\n" + \
              cls_report + "\n"


        get_logging(log, log_filename, eval='training')

    draw_loss_curve(train_losses, valid_losses, saved_path)

    path = saved_path + config.model_name + '.pt'
    torch.save(model.state_dict(), path)
    label_test, pred_test, prob_test, _ = get_label_pred_ddi_binary(model, test_loader)
    roc_auc_score_test = metrics.roc_auc_score(label_test, prob_test)
    pr_auc_score_test = metrics.average_precision_score(label_test, prob_test)

    draw_roc_pr_curve(label_test, prob_test, saved_path)

    f1_test= sk_p_r(label_test, pred_test)
    acc_test = metrics.accuracy_score(label_test, pred_test)
    cls_report_test = metrics.classification_report(label_test, pred_test, target_names=['class 0', 'class 1'])

    print('-----Test-----')
    print(cls_report_test)
    print('test dataset roc_auc:', roc_auc_score_test)
    print('test dataset pr_auc:', pr_auc_score_test)
    print('test dataset macro f1:', f1_test)
    print('test dataset acc:', acc_test)

    cf_matrix = confusion_matrix(label_test, pred_test)
    draw_confusion_matrix(config, cf_matrix, saved_path)
    print('confusion matrix:\n', cf_matrix)

    log = f"test dataset roc_auc:{roc_auc_score_test}" + "\n" + f"test dataset pr_auc:{pr_auc_score_test}" + "\n" + \
          f"test dataset macro f1: {f1_test}" + "\n" + f"test dataset acc:{acc_test}" + "\n" + \
          cls_report_test + "\n"

    get_logging(log, log_filename, eval='test')
