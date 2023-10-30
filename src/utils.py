# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# -*- coding: utf-8 -*-
"""
# @Time    : 2019/5/25
# @Author  : Jiaqi&Zecheng
# @File    : utils.py
# @Software: PyCharm
"""

import json
import time

import copy
import numpy as np
import os
import torch
from nltk.stem import WordNetLemmatizer

# from src.dataset import Example
# from src.rule import lf
# from src.rule.semQL import Sup, Sel, Order, Root, Filter, A, N, C, T, Root1

wordnet_lemmatizer = WordNetLemmatizer()

def load_dataSets(args):
    with open(args.table_path, 'r', encoding='utf8') as f:
        table_datas = json.load(f)
    import os
    # print(os.getcwd())
    # print(list(os.listdir(os.getcwd())))
    with open(args.data_path, 'r', encoding='utf8') as f:
        datas = json.load(f)

    output_tab = {}
    tables = {}
    tabel_name = set()
    for i in range(len(table_datas)):
        table = table_datas[i]
        temp = {}
        # map column names from table id
        temp['col_map'] = table['column_names']
        temp['table_names'] = table['table_names']
        # deduplicate the column names to get col set
        tmp_col = []
        for cc in [x[1] for x in table['column_names']]:
            if cc not in tmp_col:
                tmp_col.append(cc)
        table['col_set'] = tmp_col

        # add db names to table name
        db_name = table['db_id']
        tabel_name.add(db_name)

        table['schema_content'] = [col[1] for col in table['column_names']]
        table['col_table'] = [col[0] for col in table['column_names']]
        output_tab[db_name] = temp
        tables[db_name] = table

    for d in datas:
        for u in d['interaction']:
            u['query_toks'] = u['query'].split(' ')

        d['names'] = tables[d['database_id']]['schema_content']
        d['table_names'] = tables[d['database_id']]['table_names']
        d['col_set'] = tables[d['database_id']]['col_set']
        d['col_table'] = tables[d['database_id']]['col_table']
        # process primary keys and foreign keys, map bidirectional.
        keys = {}
        for kv in tables[d['database_id']]['foreign_keys']:
            keys[kv[0]] = kv[1]
            keys[kv[1]] = kv[0]
        for id_k in tables[d['database_id']]['primary_keys']:
            keys[id_k] = id_k
        d['keys'] = keys
    return datas, tables

def load_word_emb(file_name, use_small=False):
    print ('Loading word embedding from %s'%file_name)
    ret = {}
    with open(file_name) as inf:
        for idx, line in enumerate(inf):
            if (use_small and idx >= 500000):
                break
            info = line.strip().split(' ')
            if info[0].lower() not in ret:
                ret[info[0]] = np.array(list(map(lambda x:float(x), info[1:])))
    return ret

def lower_keys(x):
    if isinstance(x, list):
        return [lower_keys(v) for v in x]
    elif isinstance(x, dict):
        return dict((k.lower(), lower_keys(v)) for k, v in x.items())
    else:
        return x

def get_table_colNames(tab_ids, tab_cols):
    table_col_dict = {}
    for ci, cv in zip(tab_ids, tab_cols):
        if ci != -1:
            table_col_dict[ci] = table_col_dict.get(ci, []) + cv
    result = []
    for ci in range(len(table_col_dict)):
        result.append(table_col_dict[ci])
    return result

def get_col_table_dict(tab_cols, tab_ids, sql):
    # map tab id to ids in col_set
    table_dict = {}
    for c_id, c_v in enumerate(sql['col_set']):
        for cor_id, cor_val in enumerate(tab_cols):
            if c_v == cor_val:
                table_dict[tab_ids[cor_id]] = table_dict.get(tab_ids[cor_id], []) + [c_id]

    # map col_set id to table is
    col_table_dict = {}
    for key_item, value_item in table_dict.items():
        for value in value_item:
            col_table_dict[value] = col_table_dict.get(value, []) + [key_item]
    col_table_dict[0] = [x for x in range(len(table_dict) - 1)]
    return col_table_dict


def schema_linking(question_arg, question_arg_type, one_hot_type, col_set_type, col_set_iter, sql):

    for count_q, t_q in enumerate(question_arg_type):
        t = t_q[0]
        if t == 'NONE':
            continue
        elif t == 'table':
            one_hot_type[count_q][0] = 1
            question_arg[count_q] = ['table'] + question_arg[count_q]
        elif t == 'col':
            one_hot_type[count_q][1] = 1
            try:
                # exact match to collated question tokens
                col_set_type[col_set_iter.index(question_arg[count_q])][1] = 5
                question_arg[count_q] = ['column'] + question_arg[count_q]
            except:
                print(col_set_iter, question_arg[count_q])
                raise RuntimeError("not in col set")
        elif t == 'agg':
            one_hot_type[count_q][2] = 1
        elif t == 'MORE':
            one_hot_type[count_q][3] = 1
        elif t == 'MOST':
            one_hot_type[count_q][4] = 1
        elif t == 'value':
            one_hot_type[count_q][5] = 1
            question_arg[count_q] = ['value'] + question_arg[count_q]
        else:
            if len(t_q) == 1:
                for col_probase in t_q:
                    if col_probase == 'asd':
                        continue
                    try:
                        # conceptNet exact match
                        col_set_type[sql['col_set'].index(col_probase)][2] = 5
                        question_arg[count_q] = ['value'] + question_arg[count_q]
                    except:
                        print(sql['col_set'], col_probase)
                        raise RuntimeError('not in col')
                    one_hot_type[count_q][5] = 1
            else:
                for col_probase in t_q:
                    if col_probase == 'asd':
                        continue
                    # conceptNet exact match
                    col_set_type[sql['col_set'].index(col_probase)][3] += 1


def process(sql, entry, turn_level):

    process_dict = {}
    origin_sql = []

    for e in entry['interaction'][:turn_level+1]:
        origin_sql.extend(e['utterance_toks'])

    # origin_sql = sql['utterance_toks']
    sql['pre_sql'] = copy.deepcopy(sql)

    col_set_iter = [[wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(' ')] for x in entry['col_set']]
    q_iter_small = [wordnet_lemmatizer.lemmatize(x).lower() for x in origin_sql]
    question_arg = copy.deepcopy(sql['utterance_arg'])
    question_arg_type = sql['utterance_arg_type']
    one_hot_type = np.zeros((len(question_arg_type), 6))

    col_set_type = np.zeros((len(col_set_iter), 4))
    # process_dict['utterance'] = sql['utterance']
    process_dict['col_set_iter'] = col_set_iter
    process_dict['q_iter_small'] = q_iter_small
    process_dict['col_set_type'] = col_set_type
    process_dict['utterance_arg'] = question_arg
    process_dict['utterance_arg_type'] = question_arg_type
    process_dict['one_hot_type'] = one_hot_type

    for c_id, col_ in enumerate(process_dict['col_set_iter']):
        for q_id, ori in enumerate(process_dict['q_iter_small']):
            if ori in col_:
                process_dict['col_set_type'][c_id][0] += 1

    schema_linking(process_dict['utterance_arg'], process_dict['utterance_arg_type'],
                   process_dict['one_hot_type'], process_dict['col_set_type'],
                   process_dict['col_set_iter'], entry)
    process_dict['col_set_iter'][0] = ['count', 'number', 'many']
    return process_dict


def process_table(table, entry):
    process_dict = {}
    table_names = [[wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(' ')] for x in table['table_names']]
    tab_cols = [col[1] for col in table['column_names']]
    tab_ids = [col[0] for col in table['column_names']]
    col_set_iter = [[wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(' ')] for x in entry['col_set']]
    col_iter = [[wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(" ")] for x in tab_cols]

    process_dict['tab_cols'] = tab_cols
    process_dict['tab_ids'] = tab_ids
    process_dict['col_iter'] = col_iter
    process_dict['table_names'] = table_names
    process_dict['col_set_iter'] = col_set_iter
    return process_dict

def eval_acc(preds, sqls):
    sketch_correct, best_correct = 0, 0
    for i, (pred, sql) in enumerate(zip(preds, sqls)):
        if pred['model_result'] == sql['rule_label']:
            best_correct += 1
    print(best_correct / len(preds))
    return best_correct / len(preds)


def load_data_new(sql_path, table_data, use_small=False):
    sql_data = []

    print("Loading data from %s" % sql_path)
    with open(sql_path) as inf:
        data = lower_keys(json.load(inf))
        sql_data += data

    table_data_new = {table['db_id']: table for table in table_data}

    if use_small:
        return sql_data[:80], table_data_new
    else:
        return sql_data, table_data_new


def load_dataset(dataset_dir, use_small=False):
    print("Loading from datasets...")

    TABLE_PATH = os.path.join(dataset_dir, "tables.json")
    TRAIN_PATH = os.path.join(dataset_dir, "train.json")
    DEV_PATH = os.path.join(dataset_dir, "dev.json")
    with open(TABLE_PATH) as inf:
        print("Loading data from %s"%TABLE_PATH)
        table_data = json.load(inf)

    train_sql_data, train_table_data = load_data_new(TRAIN_PATH, table_data, use_small=use_small)
    val_sql_data, val_table_data = load_data_new(DEV_PATH, table_data, use_small=use_small)

    return train_sql_data, train_table_data, val_sql_data, val_table_data


def save_checkpoint(model, checkpoint_name):
    torch.save(model.state_dict(), checkpoint_name)


def save_args(args, path):
    with open(path, 'w') as f:
        f.write(json.dumps(vars(args), indent=4))

def init_log_checkpoint_path(args):
    save_path = args.save
    # dir_name = save_path + str(int(time.time()))
    # save_path = os.path.join(os.path.curdir, 'saved_model', dir_name)
    dir_name = save_path
    save_path = os.path.join(os.path.curdir, dir_name)
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    return save_path
