import copy
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import precision_recall_fscore_support, classification_report
import scipy.spatial.distance as spd
from scipy.special import softmax
from collections import Counter
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CPSC18', help='dataset name: CPSC18 / PTB / Georgia')
    parser.add_argument('--model', type=str, default='ResNet34', help='dataset name:  ResNet18 / ResNet34 / resnet34LSTM / inception / TSAI_Inception')
    parser.add_argument('--seed', type=int, default=1, help='')
    parser.add_argument('--Gaussian_std', type=float, default=2, help='Gaussian std, use mean + 2*std as cutoff')
    parser.add_argument('--unseen_class_name', type=int, default=4, help='will assign value later')
    parser.add_argument('--seen_class_number', type=int, default=9-1, help='will assign value later')
    parser.add_argument('--transform_type', type=str, default='scaling_up', help='reverse / scaling_up / scaling_down / .../  hardneg')
    return parser.parse_args()

args = parse_args()
open_world_path = './OpenMax/'
org_data_path = '../data_szhou/'
dataset = args.dataset
seed = args.seed
model = args.model
unseen_class_name = args.unseen_class_name
seen_class_number = args.seen_class_number
if args.dataset == 'CPSC18':
    unseen_class_name = 4
    seen_class_number = 9 - 1
if args.dataset == 'PTB':
    unseen_class_name = 3
    seen_class_number = 5 - 1
elif args.dataset == 'Georgia':
    unseen_class_name = 3
    seen_class_number = 6 - 1
if args.dataset == 'CPSC18_U3':
    unseen_class_name = 4
    seen_class_number = 7 - 1
if args.dataset == 'Georgia_U3':
    unseen_class_name = 3
    seen_class_number = 5 - 1


use_gaussian_std = True
std_number = args.Gaussian_std  # 2

trn_probs_path = open_world_path + dataset + '_' + model + '_' + 'trn_probs_' + str(args.transform_type) +'_'+ str(seed) + '.csv'
TSNEy_trn_path = open_world_path + dataset + '_' + model + '_' + 'y_trn_' + str(args.transform_type) +'_'+ str(seed) + '.csv'    # ground-truth
tst_probs_path = open_world_path + dataset + '_' + model + '_' + 'tst_probs_' + str(args.transform_type) +'_'+ str(seed) + '.csv'
split_label_path = open_world_path + str(dataset) + '_' + model + '_label_' + str(args.transform_type) +'_'+ str(seed) + '.csv'  # ground-truth
df_y_trn = pd.read_csv(TSNEy_trn_path, sep=",", header=None)
y_trn = df_y_trn.values[:]      # ground-truth
unseen_class_posit = seen_class_number
# [A, B, C, D]  4 seen_classes;
# all the unseen class will be taken as a new class, i.e., [A, B, C, D, unseen]
# #####################################################################
df_split_label = pd.read_csv(split_label_path, sep=",")
y_tst_org_gnd = df_split_label['org_label'].loc[ df_split_label['split'] == 'test'].values[:]  # retrieve org label
y_tst_contaminated_label = df_split_label['label'].loc[ df_split_label['split'] == 'test' ].values[:]
y_tst_label_open_setting = copy.deepcopy(y_tst_contaminated_label)
unseen_nodes_local_indx = np.where(y_tst_org_gnd == unseen_class_name)[0]
y_tst_label_open_setting[unseen_nodes_local_indx] = unseen_class_posit
# #####################################################################
# #####################################################################

# --------------------------------------------------------------------------------------------

# now, the unseen class will be taken as the last class, i.e., [A, B, C, D, unseen]
TSNEemb_trn_path = open_world_path + dataset + '_' + model + '_emb_trn_' + str(args.transform_type) +'_'+ str(seed) + '.csv'
TSNEemb_tst_path = open_world_path + dataset + '_' + model + '_emb_tst_' + str(args.transform_type) +'_'+ str(seed) + '.csv'
df_emb_trn = pd.read_csv(TSNEemb_trn_path, sep=",", header=None)
emb_trn = df_emb_trn.values[:]
df_emb_tst = pd.read_csv(TSNEemb_tst_path, sep=",", header=None)
emb_tst = df_emb_tst.values[:]


def compute_euclidean_distances(mean_feature, all_feature):
    eu_dist, cos_dist, eucos_dist = [], [], []
    for feat in all_feature:
        eu_dist += [spd.euclidean(mean_feature, feat)]
        cos_dist += [spd.cosine(mean_feature, feat)]
        eucos_dist += [spd.euclidean(mean_feature, feat)/200. + spd.cosine(
            mean_feature, feat)]
    return eu_dist


def compute_distances(mean_feature, all_feature):
    eu_dist, cos_dist, eucos_dist = [], [], []
    for feat in all_feature:
        eu_dist += [spd.euclidean(mean_feature, feat)]
        cos_dist += [spd.cosine(mean_feature, feat)]
        eucos_dist += [spd.euclidean(mean_feature, feat)/200. + spd.cosine(
            mean_feature, feat)]
    distances = {'eucos': eucos_dist, 'cosine': cos_dist, 'euclidean': eu_dist}

    return distances


seen_class_prototype_list = []
dis_2_center_mean_list = []
dis_2_center_std_list = []
distance_set_dict = {}
for j in range(seen_class_number):
    # [A, B, C, D]  4 seen_classes;
    # the unseen class will be taken as the last class, i.e., [A, B, C, D, unseen]
    class_j_local_indx = np.where(y_trn == j)[0]
    print('class_j_local_indx.shape', class_j_local_indx.shape)
    class_j_trn_embs = emb_trn[class_j_local_indx]
    class_j_prototype = np.mean(class_j_trn_embs, axis=0)  # avg of all embs
    assert len(list(class_j_prototype)) == emb_trn.shape[1]  # check dim the same
    seen_class_prototype_list.append(class_j_prototype)
    distance_set_dict[j] = compute_distances(class_j_prototype, class_j_trn_embs)
    euclidean_dis_list = distance_set_dict[j]['euclidean']
    dis_2_center_mean_list.append(np.mean(euclidean_dis_list))
    dis_2_center_std_list.append(np.std(euclidean_dis_list))


# --------------------------------------------------------------------------------------------------------------------
testing_OpenMax_pred = []
if use_gaussian_std == True:
    for i in range(emb_tst.shape[0]):
        tst_i_emb = emb_tst[i]
        qry_to_center_dis_list = compute_euclidean_distances(tst_i_emb, seen_class_prototype_list)
        nearest_cluster_indx = qry_to_center_dis_list.index(min(qry_to_center_dis_list))
        class_j_dis_mean = dis_2_center_mean_list[nearest_cluster_indx]
        class_j_dis_std = dis_2_center_std_list[nearest_cluster_indx]
        distance_cutoff = class_j_dis_mean + std_number * class_j_dis_std
        qry_dis_to_nearest_center = min(qry_to_center_dis_list)
        if qry_dis_to_nearest_center > distance_cutoff:
            testing_OpenMax_pred.append(unseen_class_posit)
        else:
            testing_OpenMax_pred.append(nearest_cluster_indx)
# --------------------------------------------------------------------------------------------------------------------
all_trn_labels = sorted(testing_OpenMax_pred)
trn_labels_counter = Counter(all_trn_labels)
for i, count in enumerate(dict(trn_labels_counter).items()):
    print(count)

precision, recall, fscore, _ = precision_recall_fscore_support(y_tst_label_open_setting, testing_OpenMax_pred)
clf_report = classification_report(y_tst_label_open_setting, testing_OpenMax_pred, output_dict=True)
report_df = pd.DataFrame(clf_report)
testing_f1 = report_df.loc[['f1-score']].values[0]
colunms = report_df.columns.tolist()
metrics_num = len(colunms)
report_df_more = pd.DataFrame(testing_f1.reshape(-1, metrics_num), columns=colunms, index=None)
report_df_more.to_csv('./results_final/' + str(dataset) + '_' + str(model) + '_seed_' + str(seed) + '.csv')

