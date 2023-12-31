import random
import argparse
import torch
import numpy as np

def init_arg_parser():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--seed', default=5783287, type=int, help='random seed')
    arg_parser.add_argument('--cuda', action='store_true', help='use gpu')
    arg_parser.add_argument('--no_sl', action='store_true', help='dataset')
    arg_parser.add_argument('--enable_copy', action='store_true', help='use copy mechanism')
    arg_parser.add_argument('--use_att', action='store_true', help='use att mechanism')
    arg_parser.add_argument('--use_bert', action='store_true', default=True)
    arg_parser.add_argument('--use_schema_lstm', action='store_true')
    arg_parser.add_argument('--use_turn_pos_enc', action='store_true')
    arg_parser.add_argument('--use_init_decoder_att', action='store_true')
    arg_parser.add_argument('--use_utter_schema_att', action='store_true', default=True)
    arg_parser.add_argument('--use_lstm_encoder', action='store_true')
    arg_parser.add_argument('--use_utterance_attention', action='store_true')
    arg_parser.add_argument('--use_level_att', action='store_true')
    arg_parser.add_argument('--use_utter_self_attention', action='store_true')
    arg_parser.add_argument('--discourse_level_lstm', action='store_true')
    arg_parser.add_argument('--decoder_use_schema', action='store_true')
    arg_parser.add_argument('--decoder_use_query', action='store_true')
    arg_parser.add_argument('--decoder_att_over_all', action='store_true')
    arg_parser.add_argument('--use_col_type', action='store_true', default=True)
    arg_parser.add_argument('--combine_encoding', action='store_true', default=True)
    arg_parser.add_argument('--use_diff_decoder_att', action='store_true')
    arg_parser.add_argument('--use_encoding_linear', action='store_true')
    arg_parser.add_argument('--tar_cnn_input', action='store_true')
    arg_parser.add_argument('--use_query_lstm', action='store_true')
    arg_parser.add_argument('--use_copy_query_lstm', action='store_true')
    arg_parser.add_argument('--use_pre_tree_hs', action='store_true')
    arg_parser.add_argument('--batch_loss', action='store_true', default=True)
    arg_parser.add_argument('--inter_loss', action='store_true')
    arg_parser.add_argument('--utter_loss', action='store_true')
    arg_parser.add_argument('--col_gate_utter', action='store_true', default=True)
    arg_parser.add_argument('--cuda_device_num', default=2, help='gpu rank', type=int)
    arg_parser.add_argument('--turn_num', default=1, help='turn numdd', type=int)
    arg_parser.add_argument('--pretrained_bert', default="./bert_model", type=str)
    arg_parser.add_argument('--lr_scheduler', action='store_true', default=True, help='use learning rate scheduler')
    arg_parser.add_argument('--scheduler_gamma', default=0.8, type=float, help='decay rate of learning rate scheduler')
    arg_parser.add_argument('--column_pointer', action='store_true', default=True, help='use column pointer')
    arg_parser.add_argument('--loss_epoch_threshold', default=100, type=int, help='loss epoch threshold')
    arg_parser.add_argument('--sketch_loss_coefficient', default=1, type=float, help='sketch loss coefficient')
    arg_parser.add_argument('--sentence_features', action='store_true', help='use sentence features')
    arg_parser.add_argument('--model_name', choices=['transformer', 'rnn', 'table', 'sketch'], default='rnn', help='model name')
    arg_parser.add_argument('--lstm', choices=['lstm', 'lstm_with_dropout', 'parent_feed'], default='lstm')
    arg_parser.add_argument('--load_model', default=None, type=str, help='load a pre-trained model')
    arg_parser.add_argument('--glove_embed_path', default="./word_embed_large.pk", type=str)
    arg_parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    arg_parser.add_argument('--beam_size', default=1, type=int, help='beam size for beam search')
    arg_parser.add_argument('--embed_size', default=300, type=int, help='size of word embeddings')
    arg_parser.add_argument('--col_embed_size', default=300, type=int, help='size of word embeddings')
    arg_parser.add_argument('--rule_embedding_size', default=128, type=int)
    arg_parser.add_argument('--action_embed_size', default=128, type=int)
    arg_parser.add_argument('--type_embed_size', default=128, type=int)
    arg_parser.add_argument('--turn_pos_enc_size', default=50, type=int)
    arg_parser.add_argument('--hidden_size', default=300, type=int, help='size of LSTM hidden states')
    arg_parser.add_argument('--att_vec_size', default=300, type=int, help='size of attentional vector')
    arg_parser.add_argument('--dropout', default=0.3, type=float, help='dropout rate')
    arg_parser.add_argument('--word_dropout', default=0.2, type=float, help='word dropout rate')
    arg_parser.add_argument('--no_query_vec_to_action_map', default=False, action='store_true')
    arg_parser.add_argument('--readout', default='identity', choices=['identity', 'non_linear'])
    arg_parser.add_argument('--query_vec_to_action_diff_map', default=False, action='store_true')
    arg_parser.add_argument('--column_att', choices=['dot_prod', 'affine'], default='affine')
    arg_parser.add_argument('--decode_max_time_step', default=40, type=int, help='maximum number of time steps used in decoding and sampling')
    arg_parser.add_argument('--save_to', default='model', type=str, help='save trained model to')
    arg_parser.add_argument('--toy', action='store_true', help='If set, use small data; used for fast debugging.')
    arg_parser.add_argument('--clip_grad', default=5., type=float, help='clip gradients')
    arg_parser.add_argument('--max_epoch', default=-1, type=int, help='maximum number of training epoches')
    arg_parser.add_argument('--optimizer', default='Adam', type=str, help='optimizer')
    arg_parser.add_argument('--lr_base', default=1e-3, type=float)
    arg_parser.add_argument('--lr_connection', default=1e-4, type=float)
    arg_parser.add_argument('--lr_transformer', default=1e-5, type=float)
    arg_parser.add_argument('--lr_copy', default=1e-3, type=float)
    arg_parser.add_argument('--device_num', default=0, type=int, help='select gpu')
    arg_parser.add_argument('--dataset', default="./data", type=str)
    arg_parser.add_argument('--epoch', default=100, type=int, help='Maximum Epoch')
    arg_parser.add_argument('--save', default='./', type=str, help="Path to save the checkpoint and logs of epoch")
    arg_parser.add_argument('--bert_tokenizer', default="bert-base-uncased", type=str)
    arg_parser.add_argument('--nhead', default=10, type=int)
    arg_parser.add_argument('--dim_feedforward', default=768, type=int)
    arg_parser.add_argument('--nlayers', default=6, type=int)
    arg_parser.add_argument('--rule_type_num', default=5, type=int)
    arg_parser.add_argument('--multi_head_num', default=1, type=int)
    arg_parser.add_argument('--max_action_seq_length', default=128, type=int)
    arg_parser.add_argument('--max_src_seq_length', default=512, type=int)
    arg_parser.add_argument('--bert_max_src_seq_length', default=256, type=int)
    arg_parser.add_argument('--max_col_length', default=256, type=int)
    arg_parser.add_argument('--max_table_length', default=256, type=int)
    arg_parser.add_argument('--max_align_length', default=56, type=int)
    arg_parser.add_argument('--train_data', default='../data/train.pkl', type=str)
    arg_parser.add_argument('--dev_data', default='../data/dev.pkl', type=str)
    arg_parser.add_argument('--gold', default='./data/dev_gold.txt', dest='gold', type=str)
    arg_parser.add_argument('--pred', default='./predicted_sql.txt', dest='pred', type=str)
    arg_parser.add_argument('--db', default='./data/database', dest='db', type=str)
    arg_parser.add_argument('--table', default='./data/tables.json', dest='table', type=str)
    arg_parser.add_argument('--etype', default='match', dest='etype', type=str)

    return arg_parser

def init_config(arg_parser):
    args = arg_parser.parse_args()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    # print(args.only_copy)
    np.random.seed(int(args.seed * 13 / 7))
    random.seed(int(args.seed))
    return args

