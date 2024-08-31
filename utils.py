import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, classification_report
import wfdb
import matplotlib.pyplot as plt
import pandas as pd
import os, copy
import random
from collections import Counter


def cal_f1s_naive(y_trues, y_scores, args):
    y_pred = np.argmax(y_scores, axis=1)
    y_true = np.argmax(y_trues, axis=1)
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report)
    class_colunms = [str(i) for i in range(args.classes)]
    classes_f1 = report_df.loc[['f1-score'], class_colunms]
    classes_f1 = classes_f1.values[0]
    f1_all = report_df.loc[['f1-score'], :]
    f1_all = f1_all.values[0]
    f1_micro = report_df.loc[['f1-score'], ['accuracy']].values[0][0]
    return classes_f1, f1_micro, f1_all, report_df


def gen_label_csv_unseen_setting(data_dir_szhou, org_label_csv, split_label_csv, unseen_class_index, trn_ratio, val_ratio, seed, transform_type):
    if os.path.exists(org_label_csv):
        df_label = pd.read_csv(org_label_csv)
        org_y = df_label["label"].values
        org_class_list = list(set(org_y))
        seen_class_num = len(org_class_list) - 1
        org_class_list = list(set(org_y))
        org_class_list_copy = copy.deepcopy(org_class_list)
        org_class_list_copy.remove(unseen_class_index)
        seen_class_name_list = org_class_list_copy
        seen_indices = np.where(org_y != unseen_class_index)[0]
        unseen_indices = np.where(org_y == unseen_class_index)[0]

        # ----------------------------  Split data ----------------------------
        data_num = org_y.shape[0]
        test_ratio = 1 - trn_ratio - val_ratio
        testing_data_num = int(test_ratio * data_num)
        trn_data_num = int(trn_ratio * data_num)
        ratio_of_seen_class_data = seen_indices.shape[0] / data_num
        assert testing_data_num > unseen_indices.shape[0]
        assert seen_indices.shape[0] > trn_data_num

        trn_indx = []
        val_indx = []
        for seen_class_k in seen_class_name_list:
            class_k_indx = np.where(org_y == seen_class_k)[0]
            class_k_indx_shuffle = shuffle_array(class_k_indx, seed)
            class_k_trn_num = int(class_k_indx_shuffle.shape[0] * trn_ratio / ratio_of_seen_class_data)
            class_k_val_num = int(class_k_indx_shuffle.shape[0] * val_ratio / ratio_of_seen_class_data)
            class_k_trn_indices = class_k_indx_shuffle[0: class_k_trn_num]
            class_k_val_indices = class_k_indx_shuffle[class_k_trn_num: class_k_trn_num + class_k_val_num]
            trn_indx.extend(list(class_k_trn_indices))
            val_indx.extend(list(class_k_val_indices))
        trn_indx = np.array(trn_indx)
        val_indx = np.array(val_indx)
        trn_val_index = np.append(trn_indx, val_indx)
        total_index = np.array([i for i in range(data_num)])
        all_test_data = set(total_index).difference(set(trn_val_index))
        test_indx = np.array(list(all_test_data))

        row_num = df_label.shape[0]
        new_col_value_list = np.array([None for i in range(row_num)])
        new_col_value_list[trn_indx] = 'train'
        new_col_value_list[val_indx] = 'valid'
        new_col_value_list[test_indx] = 'test'
        df_label['split'] = new_col_value_list

        all_trn_labels = sorted(list(org_y[trn_indx]))
        trn_labels_counter = Counter(all_trn_labels)
        for i, count in enumerate(dict(trn_labels_counter).items()):
            print(count)
        # ----------------------------  Add transformed label info ----------------------------
        df_label['aug_label'] = df_label["label"].values
        trn_val_set = np.append(trn_indx, val_indx, axis=0)
        generated_data_indx = np.array([i for i in range(data_num, data_num+trn_val_set.shape[0])])
        assert trn_val_set.shape[0] == generated_data_indx.shape[0]
        for j in range(generated_data_indx.shape[0]):
            source_file_indx_j = trn_val_set[j]
            new_file_indx_j = generated_data_indx[j]
            source_file_name = df_label.iloc[source_file_indx_j]['Recording']
            new_file_name = transform_type + '_' + source_file_name
            new_age = df_label.iloc[source_file_indx_j]['age']
            new_sex = df_label.iloc[source_file_indx_j]['sex']
            new_split_type = df_label.iloc[source_file_indx_j]['split']
            new_label = df_label.iloc[source_file_indx_j]['label']
            if df_label.iloc[source_file_indx_j]['label'] > unseen_class_index:
                new_aug_label = df_label.iloc[source_file_indx_j]['label'] + seen_class_num
            else:
                new_aug_label = df_label.iloc[source_file_indx_j]['label'] + seen_class_num + 1
            df_label.loc[new_file_indx_j] = [new_file_name, new_age, new_sex, new_label, new_split_type, new_aug_label]
            # ----------------------------
            source_file_path = os.path.join(data_dir_szhou, source_file_name)
            new_file_path = os.path.join(data_dir_szhou, new_file_name)
            if transform_type == 'hardneg':
                assert os.path.exists(new_file_path)
                # the Hard Negatives files are already generated based on human-knowledge, not generate here
            if os.path.exists(new_file_path):
                print(new_file_path, '\t exist \t')
            else:
                assert transform_type != 'hardneg'
                print("transform_type:", transform_type)
                print("Please generate the files in advance")
        # ----------------------------  Re-organize label ----------------------------
        auged_class_list = list(set(df_label['aug_label'].values))
        new_df_label = reorganize_label_for_unseen_setting(auged_class_list, unseen_class_index, df_label)
        new_df_label.to_csv(split_label_csv, index=None)


def reorganize_label_for_unseen_setting(org_class_list, unseen_class_index, gnd_df):
    org_class_list.remove(unseen_class_index)
    seen_class_list = org_class_list
    new_df = copy.deepcopy(gnd_df)
    new_df['org_label'] = gnd_df['label'].values
    rows_num = gnd_df.shape[0]
    new_col_value_list = np.array([0 for i in range(rows_num)])
    for j in range(len(seen_class_list)):
        seen_class_org_j = seen_class_list[j]
        new_tmp_label = j
        seen_class_local_indx_list = gnd_df[gnd_df['aug_label']==seen_class_org_j].index.tolist()
        new_col_value_list[np.array(seen_class_local_indx_list)] = new_tmp_label
    new_df['label'] = new_col_value_list
    return new_df

def gen_label_csv_unseen_setting_2_MHL(org_label_csv, split_label_csv, unseen_class_index, trn_ratio, val_ratio, seed):
    if os.path.exists(org_label_csv):
        df_label = pd.read_csv(org_label_csv)
        org_y = df_label["label"].values
        org_class_list = list(set(org_y))
        seen_class_num = len(org_class_list) - 1
        org_class_list = list(set(org_y))
        org_class_list_copy = copy.deepcopy(org_class_list)
        org_class_list_copy.remove(unseen_class_index)
        seen_class_name_list = org_class_list_copy
        seen_indices = np.where(org_y != unseen_class_index)[0]
        unseen_indices = np.where(org_y == unseen_class_index)[0]

        # ----------------------------  Split data ----------------------------
        data_num = org_y.shape[0]
        test_ratio = 1 - trn_ratio - val_ratio
        testing_data_num = int(test_ratio * data_num)
        trn_data_num = int(trn_ratio * data_num)
        ratio_of_seen_class_data = seen_indices.shape[0] / data_num
        assert testing_data_num > unseen_indices.shape[0]
        assert seen_indices.shape[0] > trn_data_num
        trn_indx = []
        val_indx = []
        for seen_class_k in seen_class_name_list:
            class_k_indx = np.where(org_y == seen_class_k)[0]
            class_k_indx_shuffle = shuffle_array(class_k_indx, seed)
            class_k_trn_num = int(class_k_indx_shuffle.shape[0] * trn_ratio / ratio_of_seen_class_data)
            class_k_val_num = int(class_k_indx_shuffle.shape[0] * val_ratio / ratio_of_seen_class_data)
            class_k_trn_indices = class_k_indx_shuffle[0: class_k_trn_num]
            class_k_val_indices = class_k_indx_shuffle[class_k_trn_num: class_k_trn_num + class_k_val_num]
            trn_indx.extend(list(class_k_trn_indices))
            val_indx.extend(list(class_k_val_indices))
        trn_indx = np.array(trn_indx)
        val_indx = np.array(val_indx)
        trn_val_index = np.append(trn_indx, val_indx)
        total_index = np.array([i for i in range(data_num)])
        all_test_data = set(total_index).difference(set(trn_val_index))
        test_indx = np.array(list(all_test_data))

        row_num = df_label.shape[0]
        new_col_value_list = np.array([None for i in range(row_num)])
        new_col_value_list[trn_indx] = 'train'
        new_col_value_list[val_indx] = 'valid'
        new_col_value_list[test_indx] = 'test'
        df_label['split'] = new_col_value_list
        all_trn_labels = sorted(list(org_y[trn_indx]))
        trn_labels_counter = Counter(all_trn_labels)
        seen_class_trn_size_list = []
        for i, count in enumerate(dict(trn_labels_counter).items()):
            seen_class_trn_size_list.append(count[1])
        # ----------------------------  Re-organize label ----------------------------
        new_df_label = reorganize_label_for_unseen_setting_MHL(org_class_list, unseen_class_index, df_label)
        new_df_label.to_csv(split_label_csv, index=None)
        print(split_label_csv )
    return seen_class_trn_size_list


def reorganize_label_for_unseen_setting_MHL(org_class_list, unseen_class_index, gnd_df):
    org_class_list.remove(unseen_class_index)
    seen_class_list = org_class_list
    new_df = copy.deepcopy(gnd_df)
    new_df['org_label'] = gnd_df['label'].values
    rows_num = gnd_df.shape[0]
    new_col_value_list = np.array([0 for i in range(rows_num)])
    for j in range(len(seen_class_list)):
        seen_class_org_j = seen_class_list[j]
        new_tmp_label = j
        seen_class_local_indx_list = gnd_df[gnd_df['label']==seen_class_org_j].index.tolist()
        print('len(seen_class_local_indx_list)', len(seen_class_local_indx_list))
        new_col_value_list[np.array(seen_class_local_indx_list)] = new_tmp_label
    new_df['label'] = new_col_value_list
    return new_df


def shuffle_array(input_arr, seed):
    total_node = input_arr.shape[0]
    randomlist = [i for i in range(total_node)]
    random.seed(seed)
    random.shuffle(randomlist)
    ouptut_arr = input_arr[randomlist]
    return ouptut_arr
