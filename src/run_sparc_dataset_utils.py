import logging
import json
import copy
import pickle
import os
import sys
import torch
import numpy as np
import argparse
from utils import load_dataSets
from args import init_arg_parser, init_config
from transformers import *
import random
from interval import Interval
from tqdm import tqdm
import pandas as pd
from nltk.grammar import Nonterminal
from nltk.grammar import Production
from translate import Translator
from RuleGenerator import RuleGenerator
from shared_variables import COL_TYPE, TAB_TYPE, RULE_TYPE, NONE_TYPE, PAD_id, EOS_id, SOS_id, VAL_TYPE
from logger import get_logger
from parser import FromSQLParser, Rule
from utils import *
from nltk.parse.stanford import StanfordParser
from tree import TreeWithPara

logger = get_logger('create_dataset.log')

class Batch(object):
    def __init__(self, examples, grammar, args):
        if args.cuda:
            self.device = torch.device('cuda', args.cuda_device_num)
        else:
            self.device = torch.device('cpu')
        self.examples = examples
        self.interactions = [e.interactions for e in self.examples]
        self.table_sents_word = [[" ".join(x) for x in e.tab_cols] for e in self.examples]
        self.schema_sents_word = [[" ".join(x) for x in e.table_names] for e in self.examples]
        self.table_sents = [e.tab_cols for e in self.examples]
        self.col_num = [e.col_num for e in self.examples]
        self.tab_ids = [e.tab_ids for e in self.examples]
        self.table_names = [e.table_names for e in self.examples]
        self.table_len = [e.table_len for e in examples]
        self.col_table_dict = [e.col_table_dict for e in examples]
        self.table_col_name = [e.table_col_name for e in examples]
        self.table_col_len = [e.table_col_len for e in examples]
        self.col_pred = [e.col_pred for e in examples]
        self.col_set = [e.col_set for e in examples]
        self.db_ids = [e.db_id for e in examples]
        self.origin_table_names = [e.origin_table_name for e in examples]
        self.origin_column_names = [e.origin_column_name for e in examples]
        self.origin_column_names = [e.origin_column_name for e in examples]
        self.column_names = [e.column_name for e in examples]
        self.grammar = grammar
        # self.src_sents_mask = [[self.generate_mask([u.src_sent_len], args.max_src_seq_length)
        #                         for u in e.utterance_features]
        #                        for e in self.examples]
        self.col_set_mask = self.generate_mask(self.col_num, args.max_col_length)
        self.table_mask = self.generate_mask(self.table_len, args.max_table_length)

    def generate_mask(self, lengthes, max_length):
        b_mask = []

        for l in lengthes:
            if l <= max_length:
                mask = [1]*l
                padding = [0]*(max_length-l)
                mask.extend(padding)
                b_mask.append(mask)
            else:
                raise Exception('token length exceed max length')
        mask = torch.tensor(b_mask, dtype=torch.long, device=self.device)
        return mask

    def to_tensor(self, obj):
        tensor_obj = torch.tensor(np.asarray(obj).squeeze(),
                                  dtype=torch.long,
                                  device=self.device)
        return tensor_obj


class BatchExample(object):
    def __init__(self, examples, batch_size,  grammar, args, drop_last=True, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        if len(examples) > 1:
            print(examples)
            self.batched_examples = self.to_batch(examples)
        else:
            self.batched_examples = [examples]

        self.batches = []
        for e in self.batched_examples:
            batch = Batch(e, grammar, args)
            self.batches.append(batch)

    def to_batch(self, examples):
        batched_example = []
        if self.shuffle:
            perm = np.random.permutation(len(examples))
        else:
            perm = np.arange(len(examples))
        st = 0
        while st < len(examples):
            ed = st + self.batch_size if st + self.batch_size < len(perm) else len(perm)
            one_batch = []
            for i in range(st, ed):
                example = examples[perm[i]]
                one_batch.append(example)
            batched_example.append(one_batch)
            st = ed
        return batched_example


class UtteranceFeature:
    """

    """
    def __init__(self, src_sent, tgt_actions=None, sql=None, one_hot_type=None, col_hot_type=None,
                 tokenized_src_sent=None, pre_actions=None,  parent_actions=None,
                 question_input_feature=None, masked_target_action=None, masked_pre_action=None,
                 masked_parent_action=None, masked_index=None, zh_src=None, fr_src=None, processed_ast=None,
                 src_sent_ner=None, copy_ast_arg=None, src_sent_origin = None, bert_features=None, group_by=None, max_seq_length=None):

        self.src_sent = src_sent
        self.src_sent_len = len(src_sent)
        self.src_sent_origin = src_sent_origin
        self.tokenized_src_sent = tokenized_src_sent
        self.sql = sql
        self.one_hot_type = one_hot_type
        self.col_hot_type = col_hot_type
        self.tgt_actions = self.to_tensor(tgt_actions) if tgt_actions is not None else None
        self.pre_actions = self.to_tensor(pre_actions) if pre_actions is not None else None
        self.parent_actions = self.to_tensor(parent_actions) if parent_actions is not None else None
        self.question_input_feature = question_input_feature
        self.masked_target_actions = self.to_tensor(masked_target_action) if masked_target_action is not None else None
        self.masked_pre_actions = self.to_tensor(masked_pre_action) if masked_pre_action is not None else None
        self.masked_parent_actions = self.to_tensor(masked_parent_action) if masked_parent_action is not None else None
        self.masked_index = masked_index if masked_index is not None else None
        self.src_sent_mask = self.generate_mask([self.src_sent_len], max_seq_length)
        self.zh_src = zh_src
        self.fr_src = fr_src
        self.processed_ast = processed_ast
        self.src_sent_ner = src_sent_ner
        self.copy_ast_arg = copy_ast_arg
        self.bert_features = bert_features
        self.group_by = group_by

    def to_tensor(self, obj):
        tensor_obj = torch.tensor(np.asarray(obj).squeeze(),
                                  dtype=torch.long)
        return tensor_obj

    def generate_mask(self, lengthes, max_length):
        b_mask = []

        for l in lengthes:
            if l <= max_length:
                mask = [1]*l
                padding = [0]*(max_length-l)
                mask.extend(padding)
                b_mask.append(mask)
            else:
                raise Exception('token length exceed max length')
        mask = torch.tensor(b_mask, dtype=torch.long)
        return mask

class Feature:
    def __init__(self, vis_seq=None, tab_cols=None, col_num=None, schema_len=None, tab_ids=None,
                 table_names=None, table_len=None, col_table_dict=None, cols=None,
                 table_col_name=None, table_col_len=None, col_set=None,
                 col_pred=None, interactions=None, db_id=None, origin_table_name=None,
                 origin_column_name=None, column_name=None, group_by=None):

        self.vis_seq = vis_seq
        self.tab_cols = tab_cols
        self.col_num = col_num
        self.schema_len = schema_len
        self.tab_ids = tab_ids
        self.table_names = table_names
        self.table_len = table_len
        self.col_table_dict = col_table_dict
        self.cols = cols
        self.table_col_name = table_col_name
        self.table_col_len = table_col_len
        self.col_pred = col_pred
        self.col_set = col_set
        self.interactions = interactions
        self.db_id = db_id
        self.origin_table_name = origin_table_name
        self.origin_column_name = origin_column_name
        self.column_name = column_name
        self.group_by = group_by


class Action(object):
    def __init__(self, act_type, rule_id, table_id, column_id, action_seq_mask, val_id_start=0, val_id_end=0):
        self.act_type = act_type
        self.rule_id = rule_id
        self.table_id = table_id
        self.column_id = column_id
        self.action_seq_mask = action_seq_mask
        self.val_id_start = val_id_start
        self.val_id_end = val_id_end



class QuestionInputFeature(object):
    def __init__(self, question_input_ids, question_input_mask, question_type_ids, target_actions=None,
                 pre_actions=None, parent_actions=None):
        """

        :param question_input_ids:
        :param question_input_mask:
        :param question_type_ids:
        :param target_actions:

        """
        self.question_input_ids = question_input_ids
        self.question_input_mask = question_input_mask
        self.question_type_ids = question_type_ids
        self.target_actions = target_actions
        self.pre_actions = pre_actions
        self.parent_actions = parent_actions


class ColumnInputFeature(object):
    def __init__(self, db_ids, col_ids, col_mask, col_token_type_ids, column_input, col_table_ids):
        self.db_ids = db_ids
        self.col_ids = col_ids
        self.col_mask = col_mask
        self.col_token_type_ids = col_token_type_ids
        self.column_input = column_input
        self.col_table_ids = col_table_ids


class TableInputFeature(object):
    def __init__(self, tab_ids, table_mask, table_token_type_ids, table_input):
        self.tab_ids = tab_ids
        self.table_mask = table_mask
        self.table_token_type_ids = table_token_type_ids
        self.table_input = table_input


class SparcProcessor(object):

    def get_examples(self, args):
        """See base class.
        :type args: object
        """
        data, table_data = load_dataSets(args)
        # data = sparc_processor._read_json(args.train_path)
        # table_data = sparc_processor._read_json(args.table_path)
        tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer, do_lower_case=False)
        examples = convert_examples_to_features(data, table_data, tokenizer, args.bert_max_src_seq_length,
                                                args.max_action_seq_length, args)

        return examples

    def get_batch(self, args):
        """See base class.
        :type args: object
        """
        data, table_data = load_dataSets(args)
        # data = sparc_processor._read_json(args.train_path)
        # table_data = sparc_processor._read_json(args.table_path)
        tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer, do_lower_case=False)
        examples = convert_examples_to_features(data, table_data, tokenizer, args.bert_max_src_seq_length,
                                                args.max_action_seq_length, args)

        return examples

    @classmethod
    def _read_tsv(cls, input_file, quote_char=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = pd.read_csv(f, sep="\t", index_col=0, quoting=False)

            # lines = []
            # for line in reader:
            #     if sys.version_info[0] == 2:
            #         line = list(unicode(cell, 'utf-8') for cell in line)
            #     lines.append(line)
            return reader

    @classmethod
    def _read_json(cls, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = json.load(f)
            return reader


def convert_examples_to_features(data, table_data, tokenizer, max_seq_length, max_action_seq_length, args):
    """Loads a data file into a list of `InputBatch`s.

    :param max_action_seq_length:
    :param data:
    :param tokenizer:
    :param max_seq_length:
    :param table_data:
    :return:
    """
    logger.info('Processing data:')
    rules = RuleGenerator()
    new_examples = []

    for ex_index in tqdm(range(len(data))):
        entry = data[ex_index]
        db_id = entry['database_id']
        process_tab_dict = process_table(table_data[db_id], entry)
        col_table_dict = get_col_table_dict(process_tab_dict['tab_cols'], process_tab_dict['tab_ids'], entry)
        table_col_name = get_table_colNames(process_tab_dict['tab_ids'], process_tab_dict['col_iter'])
        # convert * to [count, number, many]
        process_tab_dict['col_set_iter'][0] = ['count', 'number', 'many']
        interactions = []


        for turn_level, utterance in enumerate(entry['interaction']):
            utterance_arg_origin = copy.deepcopy(utterance['origin_utterance_arg'])
            actions = create_sql_input(rules, utterance, entry, max_action_seq_length, turn_level)
            # print(target_actions)
            process_dict = process(utterance, entry, turn_level)

            # translate question to fr and zh
            # zh_src = translate(process_dict['utterance_arg'], from_lang='english', to_lang='chinese')
            # fr_src = translate(process_dict['utterance_arg'], from_lang='english', to_lang='french')
            bert_process_dict = []
            if turn_level + 1 >= args.turn_num:


                for u in entry['interaction'][turn_level + 1-args.turn_num:turn_level + 1]:
                    tmp_process_dict = process(u, entry, turn_level)
                    bert_process_dict.append(tmp_process_dict)
                    # utterance_args.extend(u.src_sent)
                    # utterance_args.append([';'])
            else:
                for u in entry['interaction'][:turn_level + 1]:
                    tmp_process_dict = process(u, entry, turn_level)
                    bert_process_dict.append(tmp_process_dict)
            np_features = create_question_np_feature(process_dict, process_tab_dict, max_seq_length, tokenizer, actions)
            bert_features = create_question_np_feature(process_dict, process_tab_dict, max_seq_length, tokenizer, actions)
            pretrained_tar_action = generate_pretrained_target_action(np_features['target_actions'].squeeze(0),
                                                                      np_features['pre_actions'].squeeze(0),
                                                                      np_features['parent_actions'].squeeze(0))
            masked_target_actions = pretrained_tar_action[0]
            masked_pre_actions = pretrained_tar_action[1]
            masked_parent_actions = pretrained_tar_action[2]
            masked_index = pretrained_tar_action[3]
            # print(np_features['target_actions'])
            # print(utterance['query'])
            copy_ast_arg, processed_tree = prune_tree(np_features, interactions, turn_level, max_action_seq_length, utterance, process_dict)
            # print(copy_ast_arg)
            # ner = [stanford_ner(q) for q in process_dict['utterance_arg']]
            ner = None
            if 'group by' in utterance['query'] or 'GROUP BY' in utterance['query']:
                is_group_by = True
            else:
                is_group_by = False
            utterance_feature = UtteranceFeature(
                src_sent=process_dict['utterance_arg'],
                sql=utterance['query'],
                src_sent_origin = utterance_arg_origin,
                one_hot_type=process_dict['one_hot_type'],
                col_hot_type=process_dict['col_set_type'],
                tokenized_src_sent=process_dict['col_set_type'],
                tgt_actions=np_features['target_actions'],
                pre_actions=np_features['pre_actions'],
                parent_actions=np_features['parent_actions'],
                masked_parent_action=masked_parent_actions,
                masked_pre_action=masked_pre_actions,
                masked_target_action=masked_target_actions,
                masked_index=masked_index,
                processed_ast=processed_tree,
                src_sent_ner=ner,
                copy_ast_arg=copy_ast_arg,
                bert_features=bert_features,
                group_by=is_group_by,
                max_seq_length=max_seq_length

                # zh_src=zh_src,
                # fr_src=fr_src
            )
            interactions.append(utterance_feature)

        # new_feature.question_input_feature = np_features
        new_feature = Feature(

            col_num=len(process_tab_dict['col_set_iter']),

            # deduplicate col name iter
            tab_cols=process_tab_dict['col_set_iter'],
            col_set=entry['col_set'],
            # table_name iter
            table_names=process_tab_dict['table_names'],
            origin_table_name=table_data[db_id]['table_names_original'],
            origin_column_name=[e[1] for e in table_data[db_id]['column_names_original']],
            column_name=[e[1] for e in table_data[db_id]['column_names']],
            table_len=len(process_tab_dict['table_names']),
            col_table_dict=col_table_dict,
            # origin cols
            cols=process_tab_dict['tab_cols'],
            # origin tab id
            tab_ids=process_tab_dict['tab_ids'],
            table_col_name=table_col_name,
            table_col_len=len(table_col_name),
            interactions=interactions,
            db_id=db_id)
        new_examples.append(new_feature)

    return new_examples


def convert_batch_to_features(data, table_data, tokenizer, max_seq_length, max_action_seq_length, args):
    """Loads a data file into a list of `InputBatch`s.

    :param max_action_seq_length:
    :param data:
    :param tokenizer:
    :param max_seq_length:
    :param table_data:
    :return:
    """
    logger.info('Processing data:')
    rules = RuleGenerator()
    new_examples = []

    entry = data
    db_id = entry['database_id']
    process_tab_dict = process_table(table_data[db_id], entry)
    col_table_dict = get_col_table_dict(process_tab_dict['tab_cols'], process_tab_dict['tab_ids'], entry)
    table_col_name = get_table_colNames(process_tab_dict['tab_ids'], process_tab_dict['col_iter'])
    # convert * to [count, number, many]
    process_tab_dict['col_set_iter'][0] = ['count', 'number', 'many']
    interactions = []


    for turn_level, utterance in enumerate(entry['interaction']):
        utterance_arg_origin = copy.deepcopy(utterance['origin_utterance_arg'])
        # actions = create_sql_input(rules, utterance, entry, max_action_seq_length, turn_level)
        # print(target_actions)
        process_dict = process(utterance, entry, turn_level)

        # translate question to fr and zh
        # zh_src = translate(process_dict['utterance_arg'], from_lang='english', to_lang='chinese')
        # fr_src = translate(process_dict['utterance_arg'], from_lang='english', to_lang='french')
        bert_process_dict = []
        if turn_level + 1 >= args.turn_num:


            for u in entry['interaction'][turn_level + 1-args.turn_num:turn_level + 1]:
                tmp_process_dict = process(u, entry, turn_level)
                bert_process_dict.append(tmp_process_dict)
                # utterance_args.extend(u.src_sent)
                # utterance_args.append([';'])
        else:
            for u in entry['interaction'][:turn_level + 1]:
                tmp_process_dict = process(u, entry, turn_level)
                bert_process_dict.append(tmp_process_dict)
        np_features = create_question_np_feature(process_dict, process_tab_dict, max_seq_length, tokenizer)
        bert_features = create_question_np_feature(process_dict, process_tab_dict, max_seq_length, tokenizer)
        # pretrained_tar_action = generate_pretrained_target_action(np_features['target_actions'].squeeze(0),
        #                                                           np_features['pre_actions'].squeeze(0),
        #                                                           np_features['parent_actions'].squeeze(0))
        # masked_target_actions = pretrained_tar_action[0]
        # masked_pre_actions = pretrained_tar_action[1]
        # masked_parent_actions = pretrained_tar_action[2]
        # masked_index = pretrained_tar_action[3]
        # print(np_features['target_actions'])
        # print(utterance['query'])
        # copy_ast_arg, processed_tree = prune_tree(np_features, interactions, turn_level, max_action_seq_length, utterance, process_dict)
        # print(copy_ast_arg)
        # ner = [stanford_ner(q) for q in process_dict['utterance_arg']]
        # if 'group by' in utterance['query'] or 'GROUP BY' in utterance['query']:
        #     is_group_by = True
        # else:
        #     is_group_by = False
        utterance_feature = UtteranceFeature(
            src_sent=process_dict['utterance_arg'],
            sql=None,
            src_sent_origin=utterance_arg_origin,
            one_hot_type=process_dict['one_hot_type'],
            col_hot_type=process_dict['col_set_type'],
            tokenized_src_sent=process_dict['col_set_type'],
            tgt_actions=None,
            pre_actions=None,
            parent_actions=None,
            masked_parent_action=None,
            masked_pre_action=None,
            masked_target_action=None,
            masked_index=None,
            processed_ast=None,
            src_sent_ner=None,
            copy_ast_arg=None,
            bert_features=bert_features,
            group_by=None,
            max_seq_length=max_seq_length

            # zh_src=zh_src,
            # fr_src=fr_src
        )
        interactions.append(utterance_feature)

    # new_feature.question_input_feature = np_features
    new_feature = Feature(

        col_num=len(process_tab_dict['col_set_iter']),

        # deduplicate col name iter
        tab_cols=process_tab_dict['col_set_iter'],
        col_set=entry['col_set'],
        # table_name iter
        table_names=process_tab_dict['table_names'],
        origin_table_name=table_data[db_id]['table_names_original'],
        origin_column_name=[e[1] for e in table_data[db_id]['column_names_original']],
        column_name=[e[1] for e in table_data[db_id]['column_names']],
        table_len=len(process_tab_dict['table_names']),
        col_table_dict=col_table_dict,
        # origin cols
        cols=process_tab_dict['tab_cols'],
        # origin tab id
        tab_ids=process_tab_dict['tab_ids'],
        table_col_name=table_col_name,
        table_col_len=len(table_col_name),
        interactions=interactions,
        db_id=db_id)
    new_examples.append(new_feature)
    rule = RuleGenerator()
    batch = BatchExample(new_examples, batch_size=1, grammar=rule.rule_dict, args=args, shuffle=False)
    return batch


def to_batch(examples, batch_size, shuffle=True, drop_last=True):
    batched_example = []
    if shuffle:
        perm = np.random.permutation(len(examples))
    else:
        perm = np.arange(len(examples))
    st = 0
    while st < len(examples):
        ed = st+batch_size if batch_size < len(perm) else len(perm)
        one_batch = []
        for i in range(st, ed):
            example = examples[perm[i]]
            one_batch.append(example)
        batched_example.append(one_batch)
    return batched_example


def convert_feature_to_dict(*features):
    """
    :param features: features list
    :return: dict_feature feature
    """
    dict_feature = {}
    for feature in features:
        for idx, f in enumerate(feature):
            f_dict = f.__dict__
            for k, v in f_dict.items():
                if idx == 0:
                    dict_feature[k] = []
                    # dict_feature[k].append(v)
                dict_feature[k].append(v)
    return dict_feature


def padding_schema_input(input, max_length):
    padded_input = copy.deepcopy(input)
    if len(input) > max_length:
        padded_input= input[0:max_length]
    else:
        padded_input += ['PAD'] * (max_length - len(input))
    return padded_input


def process_question_input(question_tokens, question_one_hot_type, column_names, table_names, max_seq_length, tokenizer):

    """ ************************ create question input ***********************
    The convention in BERT is:
     (a) For sequence pairs:
      tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
      type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
     (b) For single sequences:
      tokens:   [CLS] the dog is hairy . [SEP]
      type_ids: 0   0   0   0  0     0 0

     Where "type_ids" are used to indicate whether this is the first
     sequence or the second sequence. The rule_embedding vectors for `type=0` and
     `type=1` were learned during pre-training and are added to the wordpiece
     rule_embedding vector (and position vector). This is not *strictly* necessary
     since the [SEP] token unambiguously separates the sequences, but it makes
     it easier for the model to learn the concept of sequences.

     For classification tasks, the first vector (corresponding to [CLS]) is
     used as as the "sentence vector". Note that this only makes sense because
     the entire model is fine-tuned.

     Question_tokens:['Give', 'the', 'average', 'number', 'of', ['working',  'horses'], 'on', 'farms']
     Insert ['CLS'] at the beginning of question tokens and '[SEP]' in the end
     After insertion: ['[CLS]' 'Give', 'the', 'average', 'number', 'of', ['working',  'horses'], 'on', 'farms','[SEP]']
    """

    question_one_hot_type = _truncate_seq_pair(question_tokens, column_names, table_names, question_one_hot_type, max_seq_length)
    padding_one_hot = np.zeros([1, question_one_hot_type.shape[1]])
    question_tokens.insert(0, ['[CLS]'])
    question_tokens.append(['[SEP]'])
    question_one_hot_type = np.insert(question_one_hot_type,  0, padding_one_hot, 0)
    question_one_hot_type = np.append(question_one_hot_type, padding_one_hot, 0)

    # Column names: ['*', ['city', 'id'], ['official', 'name'], 'status']
    # Insert [SEP] tokens between every two column names
    # After insertion:
    # ['*', '[SEP]', ['city', 'id'], '[SEP]', ['official', 'name'], '[SEP]', 'status', '[SEP]']
    for idx, t in enumerate((table_names, column_names)):
        type_one_hot = np.eye(question_one_hot_type.shape[1])[idx]
        type = [type_one_hot]*len(t)
        type = np.stack(type)
        for i in range(len(t)):
            t.insert((2 * (i + 1) - 1), ['[SEP]'])
            type = np.insert(type,  (2 * (i + 1) - 1), padding_one_hot, 0)
            # column_id.insert((2 * (i + 1) - 1), 0)
        question_one_hot_type = np.append(question_one_hot_type, type, 0)

    # Concatenate question_tokens with column names
    inputs = copy.deepcopy(question_tokens)
    inputs.extend(column_names)
    inputs.extend(table_names)

    # Convert input tokens to ids
    input_ids = convert_input_to_ids(inputs, tokenizer)

    # create input_mask and padding
    input_mask = [1] * len(input_ids)
    if len(input_ids) < max_seq_length:

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
    if len(inputs) < max_seq_length:
        type_padding = [padding_one_hot.squeeze(0)] * (max_seq_length - len(inputs))
    # question_one_hot_type += padding
    #     print(question_one_hot_type)
        question_one_hot_type = np.append(question_one_hot_type, type_padding, 0)
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(question_one_hot_type) == max_seq_length

    return input_ids, input_mask, question_one_hot_type


def create_schema_input(inputs, input_token_type_ids, input_max_seq_length, tokenizer, col_tab_ids=None):

    _truncate_tab_col_seq_pair(inputs, input_token_type_ids, input_max_seq_length, col_tab_ids)

    # see notes in method declaration
    expand_tokens_with_sep(inputs, input_token_type_ids, col_tab_ids)

    # convert table input to ids
    ids = convert_input_to_ids(inputs, input_token_type_ids, tokenizer, col_tab_ids)
    input_ids = ids[0]
    input_token_type_ids = ids[1]

    # create table input mask and padding
    input_mask = [1] * len(input_token_type_ids)
    table_padding = [0] * (input_max_seq_length - len(input_token_type_ids))
    if col_tab_ids is not None:
        col_tab_ids = ids[2]
        col_tab_ids_padding = [0] * (input_max_seq_length - len(col_tab_ids))
        col_tab_ids += col_tab_ids_padding
        assert len(col_tab_ids) == input_max_seq_length

    # expand the input with padding to max length
    input_ids += table_padding
    input_mask += table_padding
    input_token_type_ids += table_padding

    # assertion
    assert len(input_ids) == input_max_seq_length
    assert len(input_mask) == input_max_seq_length
    assert len(input_token_type_ids) == input_max_seq_length
    if col_tab_ids is None:
        return input_ids, input_mask, input_token_type_ids
    else:
        return input_ids, input_mask, input_token_type_ids, col_tab_ids


def create_sql_input(rules, utterance, entry, max_action_seq_length, turn_level):

    rule_seq = []
    rule_stack = []
    # print(example['sql'])
    # print(utterance['utterance_arg'])
    # ner

    parser = FromSQLParser(utterance['sql'], utterance['origin_utterance_arg'], entry['interaction'], turn_level)
    # rule_seq.append(None)
    # print(utterance['query'])
    rule_seq, rule_stack = parser.parse_sql(root=None,
                                            rules=rules,
                                            column_names=entry['names'],
                                            table_names=entry['table_names'],
                                            col_set=entry['col_set'],
                                            rule_seq=rule_seq,
                                            rule_stack=rule_stack,
                                            table_ids=entry['col_table'],
                                            is_root=True)
    parent_rules = []
    for rule in rule_seq:
        if rule.parent is None:
            parent_rules.append(None)
        else:
            parent_rules.append(rule.parent.rule)

    pre_rules = []
    for rule in rule_seq:
        if rule.pre is None:
            pre_rules.append(None)
        else:
            pre_rules.append(rule.pre.rule)

    target_rules = [rule.rule for rule in rule_seq]

    parent_rules = convert_rules_to_action_seq(parent_rules, rules, max_action_seq_length)
    pre_rules = convert_rules_to_action_seq(pre_rules, rules, max_action_seq_length)
    target_actions = convert_rules_to_action_seq(target_rules, rules, max_action_seq_length)

    return target_actions, pre_rules, parent_rules


def create_question_np_feature(process_dict, process_tab_dict, max_seq_length, tokenizer, actions=None):


    # Tokenize
    # Treat a multi_tokens words as a single word
    # Before: ['Give', 'the', 'average', 'number', 'of', 'working horses', 'on', 'farms']
    # After: ['Give', 'the', 'average', 'number', 'of', ['working', 'horses'], 'on', 'farms']
    # create question input
    if isinstance(process_dict, list):
        utterance_token_copy = []
        one_hot_type_copy = []
        for p in process_dict:
            utterance_token_copy.extend(p['utterance_arg'])
            one_hot_type_copy.append(p['one_hot_type'])
            utterance_token_copy.append(['SEP'])
            one_hot_type_copy.append(np.zeros([1, 6]))
        one_hot_type_copy = np.concatenate(one_hot_type_copy, 0)
    else:
        utterance_token_copy = copy.deepcopy(process_dict['utterance_arg'])
        column_name_copy = copy.deepcopy(process_dict['col_set_iter'])
        table_name_copy = copy.deepcopy(process_tab_dict['table_names'])
        one_hot_type_copy = copy.deepcopy(process_dict['one_hot_type'])
    # *******question input for bert*******
    utterance_input_ids, \
    utterance_input_mask, utterance_type_ids = process_question_input(question_tokens=utterance_token_copy,
                                                                      question_one_hot_type=one_hot_type_copy,
                                                                      column_names=column_name_copy,
                                                                      table_names=table_name_copy,
                                                                      max_seq_length=max_seq_length,
                                                                      tokenizer=tokenizer)

    if actions:
        utterance_input_feature = QuestionInputFeature(utterance_input_ids, utterance_input_mask, utterance_type_ids, *actions)
        np_features = convert_to_question_np_feature(utterance_input_feature)

    else:
        utterance_input_feature = QuestionInputFeature(utterance_input_ids, utterance_input_mask, utterance_type_ids)
        np_features = utterance_input_feature



    return np_features


def convert_to_question_np_feature(utterance_input_feature):
    # convert python object to numpy
    dict_features = convert_feature_to_dict([utterance_input_feature])
    np_features = {}
    for name, obj in dict_features.items():
        # convert Action object to numpy array with shape:[max_seq_action_len, action_type]
        if name == 'target_actions' or name == 'pre_actions' or name == 'parent_actions':
            target_actions = dict_features[name]
            target_action_tmp = {}
            target_seq = []
            for action_seq in target_actions:
                if action_seq:
                    keys = action_seq[0].__dict__.keys()
                    action_matrixs = [[action.__dict__[key] for action in action_seq] for key in keys]
                    matrix_trans = [np.asarray(matrix).reshape((-1, 1)) for matrix in action_matrixs]
                    target_seq.append(np.concatenate(matrix_trans, axis=1))
            # shape: [max_seq_action_len, action_type]
            np_features[name] = np.asarray(target_seq)
        else:
            np_features[name] = np.asarray(obj)
    return np_features

def flatten_ids_list(ls):
    """ Flatten the ids list
    Example: [1, 1, [1,1], 1] =>[1, 1, 1, 1, 1]
    :param ls: a list which likes [1, 1, [1,1], 1]
    :return: flattened list which contain none list type object
    """
    question_type_ids_new = []
    for i in ls:
        if isinstance(i, list):
            question_type_ids_new.extend(i)
        else:
            question_type_ids_new.append(i)
    return question_type_ids_new




def expand_tokens_with_sep(tokens, token_types_ids, col_tab_id=None):
    """ Expand tokens
    Expand tokens with '[SEP]' and '[CLS]' and expand
    token type ids with 0 for token '[SEP]' and '[CLS]'
    :param col_tab_id: table id corresponding to column id
    :param tokens: tokens
    :param token_types_ids: token type ids for tokens
    :return: None
    """

    for i in range(len(tokens)):
        tokens.insert((2 * (i + 1) - 1), '[SEP]')
    tokens.insert(0, '[CLS]')

    for i in range(int((len(tokens) + 1) / 2)):
        token_types_ids.insert((2 * i), 0)
    # padding the col_tab_id with -1
    if col_tab_id is not None:
        for i in range(int((len(tokens) + 1) / 2)):
            col_tab_id.insert((2 * i), -1)


# def convert_input_to_ids(inputs, input_type_ids, tokenizer, col_tab_ids=None):
#     input_ids = []
#     for index, token in enumerate(inputs):
#         if isinstance(token, list):
#             ids = tokenizer.convert_tokens_to_ids(token)
#             input_ids.extend(ids)
#             input_type_ids[index] = [input_type_ids[index]] * len(ids)
#             if col_tab_ids is not None:
#                 col_tab_ids[index] = [col_tab_ids[index]] * len(ids)
#
#         else:
#             ids = tokenizer.convert_tokens_to_ids(token.split(' '))
#             input_ids.extend(ids)
#     input_type_ids = flatten_ids_list(input_type_ids)
#     if col_tab_ids is not None:
#         col_tab_id = flatten_ids_list(col_tab_ids)
#         return input_ids, input_type_ids, col_tab_id
#     else:
#         return input_ids, input_type_ids

def convert_input_to_ids(inputs,  tokenizer):
    input_ids = []
    for index, token in enumerate(inputs):
        ids = tokenizer.convert_tokens_to_ids(token)
        input_ids.extend(ids)
    return input_ids


def _truncate_seq_pair(tokens_a, tokens_b, tokens_c, token_a_type, max_length):
    """Truncates a sequence pair in place to the maximum length.
    This is a simple heuristic which will always truncate the longer sequence
    one token at a time. This makes more sense than truncating an equal percent
    of tokens from each, since if one sequence is very short then each token
    that's truncated likely contains more information than a longer sequence.
    """

    while True:
        len_tok_a = len(flatten_ids_list(tokens_a))
        len_tok_b = len(flatten_ids_list(tokens_b))
        len_tok_c = len(flatten_ids_list(tokens_c))
        total_length = len_tok_a + (len_tok_b+len(tokens_b))+(len_tok_c+len(tokens_c)) + 2
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
            token_a_type = np.delete(token_a_type, -1, 0)
        else:
            tokens_b.pop()
    return token_a_type


def _truncate_tab_col_seq_pair(tokens_a, token_type, token_max_length, col_tab_ids=None):
    """Truncates a sequence pair in place to the maximum length.
    This is a simple heuristic which will always truncate the longer sequence
    one token at a time. This makes more sense than truncating an equal percent
    of tokens from each, since if one sequence is very short then each token
    that's truncated likely contains more information than a longer sequence.
    """

    while True:
        len_tok_a = len(flatten_ids_list(tokens_a))
        token_total_length = (len_tok_a+len(tokens_a)) + 1
        if token_total_length <= token_max_length:
            break
        tokens_a.pop()
        token_type.pop()
        if col_tab_ids is not None:
            col_tab_ids.pop()


def convert_rules_to_action_seq(rules: list, rules_gen: RuleGenerator, max_action_seq_length):
    productions = rules_gen.grammar.productions()
    rule_dict = rules_gen.get_rule_ids()
    rule_dict = {v:k for k, v in rule_dict.items()}
    c = Production(lhs=Nonterminal('C'), rhs=[Nonterminal('column')])
    t = Production(lhs=Nonterminal('T'), rhs=[Nonterminal('t')])
    y = Production(lhs=Nonterminal('Y'), rhs=[Nonterminal('val')])
    c_id = rules_gen.get_column_rule_id()
    t_id = rules_gen.get_table_rule_id()
    y_id = rules_gen.get_value_rule_id()

    actions = []
    sos = Action(act_type=RULE_TYPE, rule_id=SOS_id, table_id=PAD_id, column_id=PAD_id, action_seq_mask=1)
    actions.append(sos)
    for rule in rules:
        if rule is None:
            act = Action(act_type=RULE_TYPE, rule_id=SOS_id, table_id=PAD_id, column_id=PAD_id, action_seq_mask=1)
            # ids.append(ActionId(SOS_id, RULE_TYPE))
        elif rule.lhs() == Nonterminal('C'):
            col_id = rule.rhs()[0]
            act = Action(act_type=COL_TYPE,  rule_id=c_id, table_id=PAD_id, column_id=col_id, action_seq_mask=1)
            # ids.append(ActionId(c_id, COL_TYPE, str(rule.rhs())))
        elif rule.lhs() == Nonterminal('T'):
            tab_id = rule.rhs()[0]
            act = Action(act_type=TAB_TYPE, rule_id=t_id, table_id=tab_id, column_id=PAD_id, action_seq_mask=1)
            # ids.append(ActionId(t_id, TAB_TYPE, str(rule.rhs())))
        elif rule.lhs() == Nonterminal('Y'):
            # rule.rule_id = y_id
            # rule.rule_type = TOKEN_TYPE
            # rule.data = str(rule.rhs())
            # val_id = rule.rhs()[0]
            val_id = eval(rule.rhs()[0])
            # act = Action(act_type=TOKEN_TYPE, rule_id=y_id, table_id=PAD_id, column_id=PAD_id, action_seq_mask=1)
            act = Action(act_type=VAL_TYPE, rule_id=y_id, table_id=PAD_id, column_id=PAD_id, action_seq_mask=1,
                         val_id_start=val_id[0], val_id_end=val_id[1])

            # TODO copy type
            # ids.append(ActionId(y_id, VAL_TYPE, str(rule.rhs())))
        else:
            rule_id = rule_dict[rule]
            rule.rule_type = RULE_TYPE
            act = Action(act_type=RULE_TYPE, rule_id=rule_id, table_id=PAD_id, column_id=PAD_id, action_seq_mask=1)
            # ids.append(ActionId(rule_dict[rule],RULE_TYPE)
        actions.append(act)
    act = Action(act_type=RULE_TYPE, rule_id=EOS_id, table_id=PAD_id, column_id=PAD_id, action_seq_mask=1)
    actions.append(act)
    # expand action sequence to max action sequence length
    if len(actions) > max_action_seq_length:
        actions = actions[0:max_action_seq_length]
    else:
        while len(actions) < max_action_seq_length:
            act = Action(act_type=NONE_TYPE, rule_id=PAD_id, table_id=PAD_id, column_id=PAD_id, action_seq_mask=PAD_id)
            actions.append(act)
    return actions


def convert_action_seq_to_str(actions):
    rule_seq = [a[1] for a in actions]
    rule_seq_str = [str(s) for s in rule_seq]
    rule_seq_str = ','.join(rule_seq_str)
    return rule_seq_str


def read_train_json(input_file):

    with open(input_file, 'r', encoding="utf-8") as f:
        data = json.load(f)
        return data


# def convert_db_name_to_id(db_name):
#     db_id_dict = read_db_id_from_file("../data/db_ids.json")
#     dict = {v:k for k, v in db_id_dict.items()}
#     id = dict[db_name]
#     return int(id)


# def convert_id_to_db_name(db_id):
#     db_id_dict = read_db_id_from_file("../data/db_ids.json")
#     db_name = db_id_dict[str(db_id)]
#     return db_name


def generate_pretrained_target_action(target_actions, pre_actions, parent_actions):
    masked_target_actions = copy.deepcopy(target_actions)
    masked_pre_actions = copy.deepcopy(pre_actions)
    masked_parent_actions = copy.deepcopy(parent_actions)
    masked_fill = np.zeros(7)
    mask = target_actions[:, 4]
    target_action_len = np.sum(mask)
    masked_index = random.randint(1, target_action_len)
    masked_target_actions[masked_index, :] = masked_fill
    masked_pre_actions[masked_index, :] = masked_fill
    masked_parent_actions[masked_index, :] = masked_fill

    return masked_target_actions, masked_pre_actions, masked_parent_actions, masked_index


def translate(sents, from_lang, to_lang):

    translator = Translator(from_lang=from_lang, to_lang=to_lang)

    trs = [[translator.translate(w)
            for w in arg]
            for arg in sents]
    return trs


# def stanford_ner(sents):
#     parser = CoreNLPParser(url='http://localhost:9000', tagtype='ner')
#     ner = parser.tag(sents)
#     return ner


def prune_tree(np_features, interactions, turn_level, max_action_seq_length, utterance, process_dict):
    ast = build_tree_from_action_seq(np_features['target_actions'].squeeze(0), np_features['pre_actions'].squeeze(0),
                                     np_features['parent_actions'].squeeze(0), max_action_seq_length)
    # ast.pretty_print()
    processed_tree = process_ast(ast)

    copy_dict = {}
    if turn_level > 0:
        for t, u in enumerate(interactions[:turn_level]):
            query_tree = [pt[1][0] for pt in processed_tree]
            # print(utterance['query'])
            # print(process_dict['utterance_arg'])
            value_tree = [pt[1][0] for pt in u.processed_ast]
            # print(u.sql)
            # print(u.src_sent)
            copy_list = is_copy_from_pre(query_tree, value_tree)
            if len(copy_list) > 0:
                copy_list = list(copy_list)
                # # only keep longest matched tree
                # max_index = max((len(value_tree[pt]), i) for i, pt in enumerate(copy_list))
                # copy_dict[t] = copy_list[max_index[1]]
                copy_dict[t] = copy_list

    copy_ast_arg = []
    for k, c_l in copy_dict.items():
        for v in c_l:

            copy_pos = {}
            origin_actions = copy.deepcopy(np_features['target_actions'].squeeze(0))
            # print(origin_actions)
            origin_actions_str = convert_action_seq_to_str(origin_actions.tolist())
            # print(origin_actions_str)

            ps = copy.deepcopy(interactions[k].processed_ast)
            query_actions = [a[1][0] for a in ps]

            query_actions_str = convert_action_seq_to_str(query_actions[v])

            start = origin_actions_str.find(query_actions_str)

            if start != -1:
                # print(query_actions_str)
                # print(origin_actions_str)
                start = len(origin_actions_str[:start-1].split(','))
                end = start + len(query_actions_str.split(','))
                copy_pos['start'] = start
                # same to python boundary rules, except right
                copy_pos['end'] = end
                # the position of ast be copied [turn_level, id]
                copy_pos['arg'] = (k, v)
                copy_pos['seq'] = query_actions[v]
                # copy_pos['seq'] =
                copy_ast_arg.append(copy_pos)
    copy_args = filter_copy_args_with_seq(copy_ast_arg)
    # only keep most recently turn that copied from
    # if len(copy_ast_arg) > 0:
    #     copy_ast_arg = max(((i['arg'][0]), i) for i in copy_ast_arg)[1]
    # else:
    #     copy_ast_arg = None
    return copy_args, processed_tree


def filter_copy_args_with_seq(copy_args):
    copy_args_sorted_by_len = sorted(copy_args, key=lambda x: len(x['seq']))
    intervals = []
    filtered_copy_args = []
    for arg in copy_args_sorted_by_len:
        start_pos = arg['start']
        end_pos = arg['end']
        intvl = Interval(start_pos, end_pos, lower_closed = True, upper_closed=False)
        if not is_overlaps(intervals, intvl):
            intervals.append(intvl)
            filtered_copy_args.append(arg)

    # for idx,  arg in enumerate(filtered_copy_args):
    #     arg['arg'] = idx

    return filtered_copy_args

def is_overlaps(intervals, interval):
    # result = False
    if len(intervals) > 0:
        for i in intervals:
            if i.overlaps(interval):
                return True
    return False


def build_tree_from_action_seq(target_actions, parent_actions, pre_actions, max_output_length):
    r = RuleGenerator()
    asts = TreeWithPara('Z', [])
    cur_node = asts
    for i in range(max_output_length):

        i_cur_node = cur_node

        if target_actions[i+1, 1] == EOS_id:
            break

        else:
            if i_cur_node.label() == 'T':
                tab_id = target_actions[i+1, 2]
                i_cur_node.append(tab_id)
                i_cur_node.visited = True
                i_cur_node.set_rule_id(r.get_table_rule_id())
                i_cur_node.target_action = target_actions[i+1, :]
                i_cur_node.pre_action = pre_actions[i+1, :]
                i_cur_node.parent_action = parent_actions[i+1, :]
                i_cur_node = TreeWithPara.next_unvisited_b(i_cur_node)

            elif i_cur_node.label() == 'C':

                col_id = target_actions[i+1, 3]
                i_cur_node.append(col_id)
                i_cur_node.visited = True
                i_cur_node.set_rule_id(r.get_column_rule_id())
                i_cur_node.target_action = target_actions[i+1, :]
                i_cur_node.pre_action = pre_actions[i+1, :]
                i_cur_node.parent_action = parent_actions[i+1, :]
                i_cur_node = TreeWithPara.next_unvisited_b(i_cur_node)

            # elif i_cur_node.label() == 'Y':
            #
            #     rule_id = target_actions[i+1, 1]
            #     i_cur_node.append('val')
            #     i_cur_node.visited = True
            #     i_cur_node.set_rule_id(rule_id)
            #     i_cur_node.target_action = target_actions[i+1, :]
            #     i_cur_node.pre_action = pre_actions[i+1, :]
            #     i_cur_node.parent_action = parent_actions[i+1, :]
            #     i_cur_node = TreeWithPara.next_unvisited(i_cur_node)

            else:
                rule_id = target_actions[i+1, 1]
                rule = r.get_rule_by_index(rule_id)
                # print(rule)
                i_cur_node.set_rule_id(rule_id)
                i_cur_node.target_action = target_actions[i+1, :]
                i_cur_node.pre_action = pre_actions[i + 1, :]
                i_cur_node.parent_action = parent_actions[i + 1, :]
                if i == max_output_length - 1:
                    asts = None
                else:
                    expand_tree(i_cur_node, rule, r.non_terminals)

                if i_cur_node.is_all_visited():
                    i_cur_node = TreeWithPara.next_unvisited_b(i_cur_node)

                else:
                    left_most_node = i_cur_node.left_most_child_unvisited()
                    i_cur_node = left_most_node

        cur_node = i_cur_node

    return asts


def expand_tree(node, rule, non_terminals):
    rhs = list(rule.rhs())
    for element in rhs:
        if element in non_terminals:
            node.append(TreeWithPara(str(element), []))
        else:
            node.append(str(element))
    node.visited = True


def process_ast(ast):
    # print(ast)
    subtrees = list(ast.subtrees(lambda t: t.label() == 'Select' or t.label() == 'Filter' or t.label() == 'R'
                                           or t.label() == 'A' or t.label() == 'V' or t.label() == 'X'))
    # print(subtrees)
    prune_result = []
    for t in subtrees:
        # t.pretty_print()
        actions = trans_ast_to_action_seq(t)
        # print(actions)
        column_alignment, table_alignment = schema_alignment(actions[0])
        prune_result.append((t, actions, column_alignment, table_alignment))
    deduplicated_result = []
    for i in prune_result:
        if not contains_nparray(deduplicated_result, i):
            deduplicated_result.append(i)
        # t.pretty_print(maxwidth=10)
    return deduplicated_result

def contains_nparray(array, item):
    if len(array) > 0:
        for i in array:
            # print(i[1][0].shape)
            # print(item[1][0].shape)
            if i[1][0].shape ==item[1][0].shape:
                if (i[1][0]==item[1][0]).all():
                    return True
            # else:
            #     print(i[1][0].shape)
            #     print(item[1][0].shape)
            #     if i[1][0] == item[1][0]:
            #         return True
    return False
def trans_ast_to_action_seq(ast):
    # ast.pretty_print()
    target_actions = []
    pre_actions = []
    parent_actions = []
    cur_node = ast
    ast.set_tree_to_default()
    # print(ast.is_all_visited())
    while True:
        if ast.is_all_visited_b():
            break
        if cur_node.label() == 'T':
            cur_node.visited = True
            target_actions.append(cur_node.target_action)
            pre_actions.append(cur_node.pre_action)
            parent_actions.append(cur_node.parent_action)
            cur_node = TreeWithPara.next_unvisited_b(cur_node)
        elif cur_node.label() == 'C':
            cur_node.visited = True
            target_actions.append(cur_node.target_action)
            pre_actions.append(cur_node.pre_action)
            parent_actions.append(cur_node.parent_action)
            cur_node = TreeWithPara.next_unvisited_b(cur_node)
        else:
            if cur_node.is_all_visited_b():
                cur_node = TreeWithPara.next_unvisited_b(cur_node)
                cur_node.visited = True
                target_actions.append(cur_node.target_action)
                pre_actions.append(cur_node.pre_action)
                parent_actions.append(cur_node.parent_action)
            else:
                left_most_node = cur_node.left_most_child_unvisited_b()
                cur_node = left_most_node
                cur_node.visited = True
                # print(cur_node.label())
                # print(cur_node.target_action)
                if cur_node.label() == 'T' or cur_node.label() == 'C':
                    pass
                else:
                    target_actions.append(cur_node.target_action)
                    pre_actions.append(cur_node.pre_action)
                    parent_actions.append(cur_node.parent_action)
    target_actions = np.asarray(target_actions)
    pre_actions = np.asarray(pre_actions)
    parent_actions = np.asarray(parent_actions)
    return target_actions, pre_actions, parent_actions


def schema_alignment(actions):
    alignment_column = []
    alignment_table = []
    # print(actions)
    r = RuleGenerator()
    # print(actions)
    for i in range(len(actions)):
        # print(actions)
        if actions[i, 1] == r.get_column_rule_id():
            alignment_column.append(actions[i, 3])
        elif actions[i, 1] == r.get_table_rule_id():
            alignment_table.append(actions[i, 2])

    return alignment_column, alignment_table


# def stanford_ner(sents):
#     parser = CoreNLPParser(url='http://10.249.149.2:9000', tagtype='ner')
#     ner = parser.tag(sents)
#     ner_tag = [n[1] for n in ner]
#     return ner_tag


# def coref_resolution(sents):
#
#     nlp = spacy.load('en')
#     neuralcoref.add_to_pipe(nlp, greedyness=0.55)
#     doc = nlp(sents)
#     if(doc._.has_coref):
#         print(sents)
#         print(doc._.coref_clusters)
#         print(doc._.coref_resolved)


def is_copy_from_pre(query, value):
    copy_list = set()
    # print('value:')
    # print(value)
    # print('query:')
    # print(query)
    for i, f in enumerate(value):
        for j, t in enumerate(query):
            if f.shape == t.shape:
                if ((f == t).all()):
                    copy_list.add(i)
    # print(copy_list)
    return copy_list


if __name__ == '__main__':
    arg_parser = init_arg_parser()
    arg_parser.add_argument('--data_path', default='../data/dev.json', type=str, help='dataset')
    # arg_parser.add_argument('--train_path', default='../data/train.json', type=str, help='dataset')
    arg_parser.add_argument('--table_path', default='../data/tables.json', type=str, help='table dataset')
    # arg_parser.add_argument('--dev_output', default='../data/dev.pkl', type=str, help='output data')
    arg_parser.add_argument('--output', default='../data/dev.pkl', type=str, help='output data')
    args = arg_parser.parse_args()
    print('loading')
    data = SparcProcessor().get_examples(args)
    print('loaded')
    rule = RuleGenerator()
    dev_data_batch = BatchExample(data, batch_size=args.batch_size, grammar=rule.rule_dict, args=args, shuffle=False)
    # print(b.batches[0].col_set)
    # print(b.batches[0].tgt_actions.cpu().tolist())
    # print(b.batches[0].col_table_dict)
    with open(args.output, 'wb') as f:
        pickle.dump(dev_data_batch, f)

    # coref_resolution(['What are all the airlines .', 'Of it , which is Jetblue Airways ?'])






