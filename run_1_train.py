import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from dataset import ECGDataset_unseen, ECGDataset_unseen_MHL_stage2
from resnet import resnet18, resnet34, resnet34_LSTM, resnet34_MHL
from utils import cal_f1s_naive, gen_label_csv_unseen_setting, gen_label_csv_unseen_setting_2_MHL
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import shutil
from loss_func import SupConLoss, CELoss
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_szhou_data', type=str, default='True', help='use my CPSC18')
    parser.add_argument('--dataset', type=str, default='CPSC18', help='dataset name:  Physionet17 / CPSC18 / PTB / Georgia')
    parser.add_argument('--Georgia_aug', type=bool, default=True, help='whether Georgia has been augmented')
    parser.add_argument('--model', type=str, default='ResNet34', help='dataset name:  ResNet18 / ResNet34 / resnet34LSTM / TSAI_Inception')
    parser.add_argument('--data_path', type=str, default='../data_szhou/', help='saved results path')
    parser.add_argument('--whether_tsne', type=bool, default=True, help='TSNE')
    parser.add_argument('--loss_func', type=str, default='contrastive', help='cross_entropy / contrastive')
    parser.add_argument('--CL_alpha', type=float, default=0.6, help='alpha in supervised_contrastive')
    parser.add_argument('--CL_temp', type=float, default=0.1, help='temperature in  supervised_contrastive')
    parser.add_argument('--transform_type', type=str, default='scaling_up', help='reverse / scaling_up / scaling_down / .../  hardneg')     # szhou add

    parser.add_argument('--leads_zdd', type=str, default='all', help='only for zdd dataset')
    parser.add_argument('--leads', type=int, default=12, help='ECG leads to use; Data-dependent param')
    parser.add_argument('--classes', type=int, default=9, help='Num of diagnostic classes; Data-dependent param')
    parser.add_argument('--Hz', type=int, default=500, help='frequency; Data-dependent param')
    parser.add_argument('--duration', type=int, default=30, help='the duration of an ECG record; Data-dependent param')

    parser.add_argument('--epochs', type=int, default=200, help='Training epochs')
    parser.add_argument('--lr', '--learning-rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Num of workers to load data')
    parser.add_argument('--phase', type=str, default='train', help='Phase: train or test')
    parser.add_argument('--use_gpu', default=True, action='store_true', help='Use GPU')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--model_path', type=str, default='', help='Path to saved model')
    parser.add_argument('--test_results_path', type=str, default='./results/', help='saved results path')

    parser.add_argument('--val_results_path', type=str, default='./results_val/', help='saved results path')
    parser.add_argument('--open_world_path', type=str, default='./OpenMax/', help='saved results path')
    parser.add_argument('--seed', type=int, default=42, help='Seed to split data')
    parser.add_argument('--trn_ratio', type=float, default=0.7, help='train data ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='valid data ratio')
    parser.add_argument('--tes_ratio', type=float, default=0.2, help='test data ratio')
    return parser.parse_args()


def train(dataloader, net, args, criterion, epoch, scheduler, optimizer, device):
    print('Training epoch %d:' % epoch)
    net.train()
    running_loss = 0
    output_list, labels_list = [], []
    loss_batch_list = []
    steps_num = 0
    for step_k, (data, labels) in enumerate(tqdm(dataloader)):
        data, labels = data.to(device), labels.to(device)
        hidden_emb, output = net(data)
        loss = criterion(hidden_emb, output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        output_list.append(output.data.cpu().numpy())
        labels_list.append(labels.data.cpu().numpy())
        loss_batch_list.append(loss.item())
        steps_num = step_k
        label_vec = labels.data.cpu().numpy()
        trn_label = np.argmax(label_vec, axis=1)
        df_label = pd.DataFrame(trn_label)
        df_label.to_csv(args.trn_y_path, index=False, header=False, mode='a')  # continually append
        if args.whether_tsne:
            hidden_emb = hidden_emb.detach().cpu().numpy()
            df_emb = pd.DataFrame(hidden_emb)
            df_emb.to_csv(args.TSNEemb_trn_path, index=False, header=False, mode='a')   # continually append
    # scheduler.step()
    print('Loss: %.4f' % running_loss)
    loss_avg = running_loss / steps_num
    return loss_avg, loss_batch_list
    

def evaluate(dataloader, net, args, criterion, device, optimizer):
    if args.phase == 'test':
        print('Testing...')
    else:
        print('Validating...')
    net.eval()
    running_loss = 0
    output_list, labels_list = [], []
    loss_batch_list = []
    steps_num = 0

    for step_j, (data, labels) in enumerate(tqdm(dataloader)):
        data, labels = data.to(device), labels.to(device)
        hidden_emb, output = net(data)
        loss = criterion(hidden_emb, output, labels)
        running_loss += loss.item()
        output = torch.sigmoid(output)
        output_list.append(output.data.cpu().numpy())
        labels_list.append(labels.data.cpu().numpy())
        loss_batch_list.append(loss.item())
        steps_num = step_j

    print('Loss: %.4f' % running_loss)
    y_trues = np.vstack(labels_list)
    y_scores = np.vstack(output_list)
    f1_classes, f1_micro, f1_all, report_df = cal_f1s_naive(y_trues, y_scores, args)

    avg_f1 = np.mean(f1_classes)
    f1s = f1_all.tolist()

    if args.phase == 'train' and avg_f1 > args.best_metric:
        args.best_metric = avg_f1
        checkpoint = {'model': net,
                      'model_state_dict': net.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict()}
        torch.save(checkpoint, args.model_path)
        val_metric = np.array([f1s])
        colunms = report_df.columns.tolist()
        val_df = pd.DataFrame(val_metric, columns=colunms)
        val_df.to_csv(args.val_results_path + args.dataset + '_' + str(args.model) + str(args.transform_type) + '_lr_' + str(args.lr) + '_seed_' + str(args.seed) + '.csv')
        shutil.copy(args.trn_y_path, args.trn_y_best_path)
        if args.whether_tsne:
            shutil.copy(args.TSNEemb_trn_path, args.TSNEemb_trn_best_path)
    avg_val_loss = sum(loss_batch_list) / steps_num

    return avg_val_loss, loss_batch_list


def testing(dataloader, net, args, device):
    if args.phase == 'test':
        print('Testing...')
    else:
        print('Validating...')
    net.eval()
    for step_j, (data, labels) in enumerate(tqdm(dataloader)):
        data, labels = data.to(device), labels.to(device)
        hidden_emb = net(data)

        if args.phase == 'test':
            if args.whether_tsne:
                hidden_emb = hidden_emb.detach().cpu().numpy()
                df_emb = pd.DataFrame(hidden_emb)
                df_emb.to_csv(args.TSNEemb_tst_path, index=False, header=False, mode='a')   # continually append


def load_checkpoints(net, checkpoint_path, mode='Test'):
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'], strict=False)
    if mode == 'Train':
        net.train()
    if mode == 'Test':
        net.eval()
    return net


def euclidean_distance(node_embedding, c):
    return torch.sum((node_embedding - c) ** 2)

def nor_loss(node_embedding_list, c):
    s = 0
    num_node = node_embedding_list.shape[0]
    for i in range(num_node):
        s = s + euclidean_distance(node_embedding_list[i], c)
    return s


def run_MHL(dataloader, net, args, epoch, optimizer, class_prototype_list, device, seen_classes_trn_size_list):
    print('Training epoch %d:' % epoch)
    net.train()
    running_loss = 0
    loss_batch_list = []
    steps_num = 0
    # For a batch of data, it may not contain certain class_j data !!!
    for step_k, (data, labels) in enumerate(tqdm(dataloader)):
        data, labels = data.to(device), labels
        hidden_emb = net(data)
        loss_tmp = 0
        label_indx_array = np.argmax(labels, axis=1)
        for j in range(args.seen_classes_num):  # seen class num
            class_j_local_indx = np.where(label_indx_array == j)[0]
            class_j_embs = hidden_emb[class_j_local_indx]
            class_prototype_vec = class_prototype_list[j]
            seen_class_j_trn_size = seen_classes_trn_size_list[j]  # the order is from 0 to 8
            class_j_num = class_j_local_indx.shape[0]
            loss_tmp = loss_tmp + nor_loss(class_j_embs, class_prototype_vec)
        loss = loss_tmp
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        loss_batch_list.append(loss.item())
        steps_num = step_k
        df_label = pd.DataFrame(label_indx_array)
        df_label.to_csv(args.trn_y_path, index=False, header=False, mode='a')  # continually append
        if args.whether_tsne:
            hidden_emb = hidden_emb.detach().cpu().numpy()
            df_emb = pd.DataFrame(hidden_emb)
            df_emb.to_csv(args.TSNEemb_trn_path, index=False, header=False, mode='a')   # continually append
    # scheduler.step()
    loss_avg = running_loss / steps_num
    return loss_avg, loss_batch_list


if __name__ == "__main__":
    args = parse_args()
    args.best_metric = 0

    args.trn_y_path = args.test_results_path + args.dataset + '_' + args.model + '_y_trn_' + str(args.transform_type) +'_'+ str(args.seed) + '.csv'
    args.trn_y_best_path = args.open_world_path + args.dataset + '_' + args.model + '_y_trn_' + str(args.transform_type) +'_'+ str(args.seed) + '.csv'          # best model results
    args.trn_probs_path = args.test_results_path + args.dataset + '_' + args.model + '_trn_probs_' + str(args.transform_type) +'_'+ str(args.seed) + '.csv'
    args.trn_probs_best_path = args.open_world_path + args.dataset + '_' + args.model + '_trn_probs_' + str(args.transform_type) +'_'+ str(args.seed) + '.csv'  # best model results

    args.tst_y_path = args.open_world_path + args.dataset + '_' + args.model + '_y_tst_' + str(args.transform_type) +'_'+str(args.seed) + '.csv'  # best model results
    args.tst_probs_path = args.open_world_path + args.dataset + '_' + args.model + '_tst_probs_' + str(args.transform_type) +'_'+ str(args.seed) + '.csv' # best model results
    # for TSNE
    if args.whether_tsne:
        args.TSNEemb_trn_path = args.test_results_path + args.dataset + '_' + args.model + '_emb_trn_' + str(args.transform_type) +'_'+ str(args.seed) + '.csv'
        args.TSNEemb_trn_best_path = args.open_world_path + args.dataset + '_' + args.model + '_emb_trn_' + str(args.transform_type) +'_'+ str(args.seed) + '.csv'
        args.TSNEemb_tst_path = args.open_world_path + args.dataset + '_' + args.model + '_emb_tst_' + str(args.transform_type) +'_'+ str(args.seed) + '.csv'

    if not args.model_path:
        args.model_path = f'models/resnet34_{args.seed}.pth'
        args.model_path2 = f'models/resnet34_{args.seed}_stage2.pth'
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda:'+str(args.gpu_id))
    else:
        device = 'cpu'

    assert args.use_szhou_data == 'True'
    # ---------
    args.leads = 12
    args.classes = (9 - 1) * 2
    args.seen_classes_num = 9 - 1
    args.Hz = 500
    args.duration = 30
    args.length = args.Hz * args.duration
    unseen_class_name = 4
    # ---------
    if args.use_szhou_data == 'True' and args.dataset == 'PTB':
        args.leads= 12
        args.classes= (5 - 1) * 2
        args.seen_classes_num = 5 - 1
        args.Hz = 500
        args.duration = 10
        args.length = args.Hz * args.duration
        unseen_class_name = 3
    # ---------
    if args.use_szhou_data == 'True' and args.dataset == 'Georgia':
        args.leads = 12
        args.classes = (6 - 1) * 2
        args.seen_classes_num = 6 - 1
        args.Hz = 500
        args.duration = 10
        args.length = args.Hz * args.duration
        unseen_class_name = 3   # (TAb, SA)
    # ---------
    if args.use_szhou_data == 'True' and args.dataset == 'CPSC18_U3':
        args.leads = 12
        args.classes = (7 - 1) * 2
        args.seen_classes_num = 7 - 1
        args.Hz = 500
        args.duration = 30
        args.length = args.Hz * args.duration
        unseen_class_name = 4
    # ---------
    if args.use_szhou_data == 'True' and args.dataset == 'Georgia_U3':
        args.leads = 12
        args.classes = (5 - 1) * 2
        args.seen_classes_num = 5 - 1
        args.Hz = 500
        args.duration = 10
        args.length = args.Hz * args.duration
        unseen_class_name = 3
    # ---------
    args.label_file = args.data_path + args.dataset +'_label_all.csv'
    data_dir_szhou = args.data_path + args.dataset + '_szhou_all'

    if args.dataset == 'CPSC18_U2' or args.dataset == 'CPSC18-STE' or args.dataset == 'CPSC18_U3':
        data_dir_szhou = args.data_path + 'CPSC18_szhou_all'
        # the same ecg folder as CPSC18, but different label_all.csv
    if args.dataset == 'Georgia_U3':
        data_dir_szhou = args.data_path + 'Georgia_szhou_all'
        # the same ecg folder as Georgia, but different label_all.csv

    label_csv_stage_1 = args.open_world_path + args.dataset + '_' + args.model + '_label_' + str(args.transform_type) + '_Stage1_'+ str(args.seed) + '.csv'
    label_csv = args.open_world_path + args.dataset + '_' + args.model + '_label_' + str(args.transform_type) + '_'+ str(args.seed) + '.csv'

    # ----------------------
    gen_label_csv_unseen_setting(data_dir_szhou, args.label_file, label_csv_stage_1, unseen_class_name, args.trn_ratio, args.val_ratio, args.seed, args.transform_type)

    if args.dataset == 'Georgia' and args.Georgia_aug == True:
        print('Georgia directly aug 4 times')

    train_dataset = ECGDataset_unseen('train', data_dir_szhou, label_csv_stage_1, args.leads, args.length )
    val_dataset = ECGDataset_unseen('valid', data_dir_szhou, label_csv_stage_1, args.leads, args.length )
    test_dataset = ECGDataset_unseen('test', data_dir_szhou, label_csv_stage_1, args.leads, args.length )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    net = resnet34(input_channels=args.leads, num_classes=args.classes).to(device)
    if args.model == 'ResNet18':
        net = resnet18(input_channels=args.leads, num_classes=args.classes).to(device)
    if args.model == 'resnet34_LSTM':
        net = resnet34_LSTM(input_channels=args.leads, num_classes=args.classes).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.1)

    # ------- for multi-class classification -----------
    criterion = CELoss()
    if args.loss_func == 'contrastive':
        print('using SupConLoss function ...   \n    pytorch version >= 1.10 ...')
        criterion = SupConLoss(args.CL_alpha, args.CL_temp)
    else:
        print('using Cross_Entropy function ...   \n')
    # ------------------------------------------------------------------------

    t_total = time.time()
    # --------------------- Training & Validation ---------------------
    assert args.phase == 'train'
    trn_loss_batch_list = []
    trn_loss_epoch_list = []
    val_loss_batch_list = []
    val_loss_epoch_list = []
    trn_loss_epoch_list_2 = []
    # ---------------------- (1) stage 1 -----------------------
    for epoch in range(args.epochs):
        if os.path.exists(args.trn_probs_path):
            os.remove(args.trn_probs_path)
        if os.path.exists(args.trn_y_path):
            os.remove(args.trn_y_path)
        if args.whether_tsne:
            if os.path.exists(args.TSNEemb_trn_path):
                os.remove(args.TSNEemb_trn_path)
        avg_loss_epoch, loss_batch_list = train(train_loader, net, args, criterion, epoch, scheduler, optimizer, device)
        avg_val_loss, valLoss_batch_list = evaluate(val_loader, net, args, criterion, device, optimizer)
        trn_loss_batch_list.extend(loss_batch_list)
        val_loss_batch_list.extend(valLoss_batch_list)
        trn_loss_epoch_list.append(avg_loss_epoch)
        val_loss_epoch_list.append(avg_val_loss)
    # ---------------------- (2) MHL loss -----------------------
    seen_class_prototype_list = []
    assert os.path.exists(args.TSNEemb_trn_best_path)
    df_emb_trn = pd.read_csv(args.TSNEemb_trn_best_path, sep=",", header=None)
    emb_trn = df_emb_trn.values[:]
    df_y_trn = pd.read_csv(args.trn_y_best_path, sep=",", header=None)
    y_trn = df_y_trn.values[:]

    for j in range(args.seen_classes_num):
        class_j_local_indx = np.where(y_trn == j)[0]
        class_j_trn_embs = emb_trn[class_j_local_indx]
        class_j_prototype = np.mean(class_j_trn_embs, axis=0)
        assert len(list(class_j_prototype)) == emb_trn.shape[1]
        class_j_prototype = torch.from_numpy(class_j_prototype)
        class_j_prototype = class_j_prototype.to(device)
        seen_class_prototype_list.append(class_j_prototype)

    finetune_Net = resnet34_MHL(input_channels=args.leads).to(device)
    finetune_Net = load_checkpoints(finetune_Net, args.model_path, mode='Train')
    optimizer = torch.optim.Adam(finetune_Net.parameters(), lr=args.lr)
    finetune_Net = finetune_Net.to(device)

    seen_classes_trn_size_list = gen_label_csv_unseen_setting_2_MHL(args.label_file, label_csv, unseen_class_name, args.trn_ratio, args.val_ratio, args.seed)
    train_dataset_MHL = ECGDataset_unseen_MHL_stage2('train_valid', data_dir_szhou, label_csv, args.leads, args.length)
    train_loader_MHL = DataLoader(train_dataset_MHL, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    if os.path.exists(args.TSNEemb_trn_best_path):
        os.remove(args.TSNEemb_trn_best_path)
    if os.path.exists(args.trn_y_best_path):
        os.remove(args.trn_y_best_path)

    lowest_avg_loss_2 = 100000000
    for epoch in range(args.epochs):
        if os.path.exists(args.trn_y_path):
            os.remove(args.trn_y_path)
        if args.whether_tsne:
            if os.path.exists(args.TSNEemb_trn_path):
                os.remove(args.TSNEemb_trn_path)
        avg_loss_epoch_2, loss_batch_list_2 = run_MHL(train_loader_MHL, finetune_Net, args, epoch, optimizer, seen_class_prototype_list, device, seen_classes_trn_size_list)
        trn_loss_epoch_list_2.append(avg_loss_epoch_2)
        if avg_loss_epoch_2 < lowest_avg_loss_2:
            lowest_avg_loss_2 = avg_loss_epoch_2
            torch.save(finetune_Net.state_dict(), args.model_path2)
            shutil.copy(args.trn_y_path, args.trn_y_best_path)
            if args.whether_tsne:
                shutil.copy(args.TSNEemb_trn_path, args.TSNEemb_trn_best_path)

    with open('./logs/'+args.dataset + '_' + args.model + '_epochs_' + str(args.epochs) + '_seed_'+str(args.seed), 'w') as fwr:
        running_time = time.time() - t_total
        fwr.writelines(str(running_time))
    print('Training & Validation over... \n')

    if os.path.exists(args.trn_y_path):
        os.remove(args.trn_y_path)
    if args.whether_tsne:
        if os.path.exists(args.TSNEemb_trn_path):
            os.remove(args.TSNEemb_trn_path)
    # --------------------- Testing ---------------------
    args.phase = 'test'
    finetune_Net.load_state_dict(torch.load(args.model_path2, map_location=device))
    if args.phase == 'test' and args.whether_tsne:
        if os.path.exists(args.TSNEemb_tst_path):
            os.remove(args.TSNEemb_tst_path)
        if os.path.exists(args.tst_y_path):
            os.remove(args.tst_y_path)
        if os.path.exists(args.tst_probs_path):
            os.remove(args.tst_probs_path)
    testing(test_loader, finetune_Net, args, device)



