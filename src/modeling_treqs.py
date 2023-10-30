# noinspection PyCallingNonCallable
import math
import operator
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from encoder import *
import json
import copy
import numpy as np
from queue import PriorityQueue
from tree import *
from RuleGenerator import RuleGenerator
from shared_variables import COL_TYPE, TAB_TYPE, RULE_TYPE, NONE_TYPE, PAD_id, EOS_id, SOS_id, VAL_TYPE
from run_sparc_dataset_utils import *
import torch
from attention import *
from transformers import *
import torch_utils


def dim1to2(tensors):
    for t in tensors:
        if t.dim() == 1:
            t = t.view(1, -1)
    return tensors


def dim2to3(tensors):
    for t in tensors:
        if t.dim() == 2:
            t = t.unsqueeze(0)


class PointerNet(nn.Module):
    def __init__(self, query_vec_size, src_encoding_size, attention_type='affine'):
        super(PointerNet, self).__init__()

        assert attention_type in ('affine', 'dot_prod')
        if attention_type == 'affine':
            self.src_encoding_linear = nn.Linear(src_encoding_size, query_vec_size, bias=False)

        self.attention_type = attention_type
        self.input_linear = nn.Linear(query_vec_size, query_vec_size)
        self.type_linear = nn.Linear(32, query_vec_size)
        self.V = Parameter(torch.FloatTensor(query_vec_size), requires_grad=True)
        self.tanh = nn.Tanh()
        self.context_linear = nn.Conv1d(src_encoding_size, query_vec_size, 1, 1)
        self.coverage_linear = nn.Conv1d(1, query_vec_size, 1, 1)

        nn.init.uniform_(self.V, -1, 1)

    def forward(self, src_encodings, src_token_mask, query_vec):
        """
        :param src_encodings: Variable(batch_size, src_sent_len, hidden_size * 2)
        :param src_token_mask: Variable(batch_size, src_sent_len)
        :param query_vec: Variable(tgt_action_num, batch_size, query_vec_size)
        :return: Variable(tgt_action_num, batch_size, src_sent_len)
        """

        # (batch_size, 1, src_sent_len, query_vec_size)

        if self.attention_type == 'affine':
            src_encodings = self.src_encoding_linear(src_encodings)
        src_encodings = src_encodings.unsqueeze(1)

        # (batch_size, tgt_action_num, query_vec_size, 1)
        q = query_vec.permute(1, 0, 2).unsqueeze(3)

        weights = torch.matmul(src_encodings, q).squeeze(3)

        weights = weights.permute(1, 0, 2)

        if src_token_mask is not None:
            src_token_mask = src_token_mask.unsqueeze(0).expand_as(weights)
            weights.data.masked_fill_(src_token_mask.to(torch.bool), -float('inf'))

        return weights


class InteractionLayerInput(object):
    def __init__(self, interactions=None, table_input=None, table_mask=None, column_input=None, col_mask=None,
                 col_table_dict=None):

        self.interactions = interactions
        self.table_input = table_input
        self.table_mask = table_mask
        self.column_input = column_input
        self.col_mask = col_mask
        self.col_table_dict = col_table_dict

    def init_with_batch_index(self, batch, i):
        self.interactions = batch.interactions[i]
        self.table_input = batch.table_names[i]
        self.table_mask = batch.table_mask[i]
        self.column_input = batch.table_sents[i]
        self.col_mask = batch.col_set_mask[i]
        self.col_table_dict = batch.col_table_dict[i]


class DecoderInput(object):
    def __init__(self, encoder_outputs, final_states, tgt_actions, interaction_input,
                 group_by, col_embedding, table_embedding, inter_mem):
        self.encoder_outputs = encoder_outputs
        self.col_embedding = col_embedding
        self.table_embedding = table_embedding
        self.final_states = final_states
        self.inter_mem = inter_mem
        self.tgt_actions = tgt_actions
        self.table_mask = interaction_input.table_mask
        self.table_input = interaction_input.table_input
        self.col_mask = interaction_input.col_mask
        self.column_input = interaction_input.column_input
        self.col_table_dict = interaction_input.col_table_dict
        self.is_group_by = group_by


class NL2SQLTransformer(nn.Module):
    """
    IRNet for training process, during training model, IRNet compute the total loss of
    generated sequence.
    """

    def __init__(self, args, word_emb=None):
        super(NL2SQLTransformer, self).__init__()
        if args.cuda:
            self.device = torch.device('cuda', args.cuda_device_num)
        else:
            self.device = torch.device('cpu')
        self.args = args
        self.word_emb = word_emb
        rule = RuleGenerator()
        self.rule_size = len(rule.rule_dict)
        self.max_input_length = args.max_src_seq_length
        self.rule_embedding = nn.Embedding(self.rule_size, args.rule_embedding_size, padding_idx=PAD_id)
        self.rule_type_embedding = nn.Embedding(args.rule_type_num, args.type_embed_size, padding_idx=PAD_id)
        self.positional_embedder = nn.Embedding(args.turn_num, self.args.turn_pos_enc_size)
        self.col_type = nn.Linear(4, args.col_embed_size)
        self.prob_att = nn.Linear(args.att_vec_size, 1)

        self.rule_output_act = F.tanh if args.readout == 'non_linear' else nn
        # self.rule_output = nn.Linear(args.rule_embedding_size, args.rule_embedding_size)
        self.schema_input = nn.Linear(args.col_embed_size, args.rule_embedding_size)

        self.copy_output = nn.Linear(args.hidden_size, 2)

        self.max_col_input_length = args.max_col_length
        self.max_tab_input_length = args.max_table_length
        self.query_encoder_lstm = nn.LSTM(args.rule_embedding_size, args.rule_embedding_size // 2, bidirectional=True,
                                          batch_first=True)
        self.prev_query_encoder_lstm = nn.LSTM(args.rule_embedding_size, args.rule_embedding_size // 2,
                                               bidirectional=True,
                                               batch_first=True)
        self.encoder = TransformerEncoder(args.bert_tokenizer,
                                          self.device, args.max_src_seq_length,
                                          args.hidden_size,
                                          args.hidden_size,
                                          self.args)
        self.input_dim = args.rule_embedding_size + \
                         args.att_vec_size + \
                         args.type_embed_size
        # self.query_encoder_lstm = nn.LSTM(args.embed_size, args.hidden_size // 2, bidirectional=True,
        #                                   batch_first=True)
        self.decoder_lstm = nn.LSTMCell(self.input_dim, args.hidden_size)
        # pointer network
        encoder_hidden_size = self.args.hidden_size if self.args.use_lstm_encoder else self.encoder.encoder_hidden_size

        utter_embedding_size = self.args.hidden_size if self.args.use_utterance_attention else self.args.hidden_size

        if args.discourse_level_lstm:
            self.discourse_lstms = torch_utils.create_multilayer_lstm_params(1, args.hidden_size, args.hidden_size / 2, "LSTM-t")
            self.initial_discourse_state = torch_utils.add_params(tuple([args.hidden_size / 2]), "V-turn-state-0")
            utter_embedding_size += int(args.hidden_size / 2)
        if self.args.use_turn_pos_enc:
            utter_embedding_size += self.args.turn_pos_enc_size

        schema_embedding_size = args.hidden_size if self.args.use_utter_schema_att else args.hidden_size
        decoder_att_size = self.args.hidden_size + self.args.hidden_size
        decoder_att_size_copy = self.args.hidden_size + self.args.hidden_size

        if self.args.decoder_use_schema:
            decoder_att_size += self.args.col_embed_size
            decoder_att_size_copy += self.args.col_embed_size
        if self.args.decoder_use_query:
            decoder_att_size += self.args.hidden_size
        decoder_att_size_copy += self.args.hidden_size

        self.column_pointer_net = PointerNet(args.hidden_size, schema_embedding_size, attention_type=args.column_att)
        self.table_pointer_net = PointerNet(args.hidden_size, schema_embedding_size, attention_type=args.column_att)
        self.val_pointer_net_start = PointerNet(args.hidden_size, args.hidden_size, attention_type=args.column_att)
        self.val_pointer_net_end = PointerNet(args.hidden_size, args.hidden_size, attention_type=args.column_att)

        if self.args.enable_copy:
            self.copy_ast_pointer_net = PointerNet(args.hidden_size, args.rule_embedding_size,
                                                   attention_type=args.column_att)
            self.sketch_copy_ast_pointer_net = PointerNet(args.hidden_size, args.rule_embedding_size,
                                                          attention_type=args.column_att)

        # self.val_pointer_network = PointerNet(utter_embedding_size, self.args.hidden_size, attention_type=args.column_att)
        # self.sketch_att_vec_linear = nn.Linear(args.hidden_size, args.att_vec_size, bias=False)
        self.dropout = nn.Dropout(args.dropout)
        self.query_vec_to_action_embed = nn.Linear(args.att_vec_size, args.rule_embedding_size,
                                                   bias=args.readout == 'non_linear')
        self.prev_query_vec_to_action_embed = nn.Linear(args.att_vec_size, args.rule_embedding_size,
                                                        bias=args.readout == 'non_linear')
        self.prev_query_vec_to_col_embed = nn.Linear(args.att_vec_size, args.col_embed_size,
                                                     bias=args.readout == 'non_linear')
        self.prev_query_vec_to_tab_embed = nn.Linear(args.att_vec_size, args.col_embed_size,
                                                     bias=args.readout == 'non_linear')
        self.copy_vec_to_action_embed = nn.Linear(args.hidden_size, args.rule_embedding_size,
                                                  bias=args.readout == 'non_linear')

        self.decoder_src_attention = nn.MultiheadAttention(self.args.hidden_size, 1, kdim=utter_embedding_size,
                                                           vdim=utter_embedding_size)
        self.sketch_decoder_src_attention = nn.MultiheadAttention(self.args.hidden_size, 1, kdim=utter_embedding_size,
                                                                  vdim=utter_embedding_size)

        self.decoder_query_attention = nn.MultiheadAttention(self.args.hidden_size, 1,
                                                             kdim=self.args.rule_embedding_size,
                                                             vdim=self.args.rule_embedding_size)
        self.decoder_query_attention_copy = nn.MultiheadAttention(self.args.hidden_size, 1,
                                                             kdim=self.args.rule_embedding_size,
                                                             vdim=self.args.rule_embedding_size)
        self.sketch_copy_query_attention = nn.MultiheadAttention(self.args.hidden_size, 1,
                                                                 kdim=self.args.rule_embedding_size,
                                                                 vdim=self.args.rule_embedding_size)
        self.copy_query_attention = nn.MultiheadAttention(self.args.hidden_size, 1,
                                                          kdim=self.args.rule_embedding_size,
                                                          vdim=self.args.rule_embedding_size)
        self.sketch_decoder_query_attention = nn.MultiheadAttention(self.args.hidden_size, 1,
                                                                    kdim=self.args.rule_embedding_size,
                                                                    vdim=self.args.rule_embedding_size)
        self.decoder_schema_attention = nn.MultiheadAttention(self.args.hidden_size, 1, kdim=schema_embedding_size,
                                                              vdim=schema_embedding_size)
        self.sketch_decoder_schema_attention = nn.MultiheadAttention(self.args.hidden_size, 1, kdim=schema_embedding_size,
                                                                     vdim=schema_embedding_size)

        self.att_vec_linear = nn.Linear(decoder_att_size, args.hidden_size, bias=False)
        self.att_sketch_linear = nn.Linear(decoder_att_size, args.hidden_size, bias=False)
        self.att_copy_linear = nn.Linear(decoder_att_size_copy, 1, bias=False)
        self.att_copy_sketch_linear = nn.Linear(decoder_att_size_copy, 1, bias=False)
        self.sketch_att_vec_linear = nn.Linear(utter_embedding_size, utter_embedding_size, bias=False)
        self.lf_att_vec_linear = nn.Linear(utter_embedding_size, utter_embedding_size, bias=False)
        # # not used
        self.copy_witch_linear = nn.Linear(self.args.hidden_size, 1, bias=False)
        self.sketch_copy_witch_linear = nn.Linear(self.args.hidden_size, 1, bias=False)

        self.interaction_attention = nn.MultiheadAttention(encoder_hidden_size, self.args.multi_head_num,
                                                           kdim=encoder_hidden_size, vdim=encoder_hidden_size)
        self.utterance_attention = nn.MultiheadAttention(utter_embedding_size, self.args.multi_head_num,
                                                         kdim=args.hidden_size, vdim=args.hidden_size)
        self.utterance_attention_linear = nn.Linear(utter_embedding_size * 2, utter_embedding_size, bias=False)
        self.decoder_cell_init = nn.Linear(encoder_hidden_size, args.hidden_size)
        self.copy_out = nn.Linear(encoder_hidden_size, 2)
        self.group_output = nn.Linear(encoder_hidden_size, 2)

        self.schema_attention = nn.MultiheadAttention(schema_embedding_size, self.args.multi_head_num,
                                                      kdim=utter_embedding_size, vdim=utter_embedding_size)
        # self.schema_att_linear = nn.Linear(args.hidden_size*2, args.hidden_size, bias=False)
        self.column_rnn_input = nn.Linear(schema_embedding_size, args.rule_embedding_size, bias=False)
        self.table_rnn_input = nn.Linear(schema_embedding_size, args.rule_embedding_size, bias=False)
        self.start_query_attention_vector = torch.nn.Parameter(
            torch.empty(self.args.rule_embedding_size).uniform_(-0.1, 0.1))
        self.act = nn.LeakyReLU(0.1)
        self.utter_muiltiheadatt = nn.MultiheadAttention(args.hidden_size, 1)
        # init the embedding layer
        nn.init.xavier_normal_(self.rule_embedding.weight.data)
        nn.init.xavier_normal_(self.rule_type_embedding.weight.data)
        nn.init.xavier_normal_(self.positional_embedder.weight.data)

    def forward(self, batch, device, optimizer, scheduler, batch_id, dirty_data):
        batch_loss = 0
        for i in range(len(batch.interactions)):
            if [batch_id, i] in dirty_data:

                continue
            interaction_layer_input = InteractionLayerInput()
            interaction_layer_input.init_with_batch_index(batch, i)
            t_loss = self.interactions_layer(interaction_layer_input, device, optimizer)

            if torch.isnan(t_loss):
                print("dirty uid:", i)

            batch_loss += t_loss
            # if total_loss != 0:
            #     optimizer.zero_grad()
            #     total_loss.backward(retain_graph=True)
            #     # group_loss.backward()
            #     # print(group_loss)
            #     if self.args.clip_grad > 0.:
            #         grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.clip_grad)
            #     optimizer.step()
            #     if self.args.lr_scheduler:
            #         scheduler.step()

        return batch_loss / len(batch.interactions)

    def interactions_layer(self, inputs, device, optimizer):

        inter_loss = 0
        pre_queries = []
        previous_query_states = []
        for turn_level, utter in enumerate(inputs.interactions):

            decoder_input = self.get_decoder_input(turn_level, utter, inputs)

            final_utterance_state = self.get_decoder_init_vec(decoder_input, turn_level)
            u_loss, u_sketch_loss, group_loss, query_history = self.decode(decoder_input, turn_level,
                                                                           previous_query_states,
                                                                           pre_queries,
                                                                           final_utterance_state, device)

            previous_query_states.append(torch.stack(query_history, 0))
            pre_queries.append(utter.tgt_actions)
            utter_loss = u_loss + u_sketch_loss
            if torch.isnan(utter_loss):
                print(decoder_input.encoder_outputs)
            # print(u_loss)
            if utter_loss != 0 and self.args.utter_loss:
                optimizer.zero_grad()
                utter_loss.backward(retain_graph=True)
                if self.args.clip_grad > 0.:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.clip_grad)
                optimizer.step()
            inter_loss += utter_loss

        if self.args.inter_loss:
            if inter_loss != 0:

                optimizer.zero_grad()
                inter_loss.backward()
                if self.args.clip_grad > 0.:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.clip_grad)
                optimizer.step()

        if self.args.batch_loss:
            if inter_loss != 0:
                return inter_loss / len(inputs.interactions)
            else:
                return inter_loss
        else:
            return inter_loss

    def interactions_layer2(self, inputs, device, optimizer):

        inter_loss = 0
        pre_queries = []
        previous_query_states = []
        input_hidden_states = []
        # final_states = []
        final_states = []
        src_lens = []
        if self.args.discourse_level_lstm:
            discourse_state, discourse_lstm_states = self._initialize_discourse_states()
        for turn_level, utter in enumerate(inputs.interactions):
            num_utterances_to_keep = min(self.args.turn_num, turn_level + 1)
            decoder_input = self.get_decoder_input2(turn_level, utter, inputs, discourse_state,
                                                    input_hidden_states, final_states, src_lens)

            # print(decoder_input.encoder_outputs)

            final_utterance_state = self.get_decoder_init_vec(decoder_input, turn_level)
            # final_utterance_states_c, final_utterance_states_h, final_utterance_state = \
            #     self.get_utterance_attention(final_utterance_states_c, final_utterance_states_h,
            #                                  final_utterance_state, num_utterances_to_keep)
            u_loss, u_sketch_loss, group_loss, query_history = self.decode(decoder_input, turn_level,
                                                                           previous_query_states,
                                                                           pre_queries,
                                                                           final_utterance_state, device)

            # print(final_utterance_state[0].size())
            if self.args.discourse_level_lstm:
                _, discourse_state, discourse_lstm_states = torch_utils.forward_one_multilayer(
                    self.discourse_lstms, final_utterance_state[0].squeeze(), discourse_lstm_states, 0.)

            previous_query_states, pre_queries = self.get_previous_query(previous_query_states, pre_queries, decoder_input.tgt_actions)
            previous_query_states = previous_query_states[-num_utterances_to_keep:]
            pre_queries = pre_queries[-num_utterances_to_keep:]

            # previous_query_embedd.append(torch.stack(query_history, 0))
            # pre_target_actions.append(utter.tgt_actions)
            utter_loss = u_loss + u_sketch_loss

            if utter_loss != 0 and self.args.utter_loss:
                optimizer.zero_grad()
                utter_loss.backward(retain_graph=False)
                if self.args.clip_grad > 0.:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.clip_grad)
                optimizer.step()
            inter_loss += utter_loss

        if self.args.inter_loss:
            if inter_loss != 0:

                optimizer.zero_grad()
                inter_loss.backward()
                if self.args.clip_grad > 0.:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.clip_grad)
                optimizer.step()

        if self.args.batch_loss:
            if inter_loss != 0:
                return inter_loss / len(inputs.interactions)
            else:
                return inter_loss
        else:
            return inter_loss

    def score_previous_query(self, previous_query_embedd, decoder_state):
        previous_query_embedd_cat = torch.cat(previous_query_embedd, 0)
        query_logits = torch.matmul(previous_query_embedd_cat,
                                    self.prev_query_vec_to_action_embed(decoder_state).transpose(0, 1))
        query_probs = F.softmax(query_logits.transpose(0, 1), dim=1)
        return query_probs.unsqueeze(0)

    def get_previous_query(self, previous_query_states, previous_queries, previous_query):
        actions = previous_query[:,1]
        query_len = previous_query[:, 4].sum().item()
        query_embedd = self.rule_embedding(actions[:query_len-1])

        query_states, _ = self.lstm_encode(query_embedd.unsqueeze(0), query_len-1, self.query_encoder_lstm)
        previous_query_states.append(query_states.squeeze(0))
        previous_queries.append(previous_query)
        return previous_query_states, previous_queries

    def decode(self, inputs, turn_level, previous_query_embedd, pre_target_actions, final_utterance_state, device):
        """

        :param inputs:
        :param turn_level:
        :param previous_query_embedd:
        :param pre_target_actions:
        :param device:
        :return:
        """
        noise = 0.00000001
        # print(inputs.tgt_actions)
        loss = 0
        sketch_loss = 0
        if self.args.use_encoding_linear:

            utterance_encodings_sketch_linear = self.sketch_att_vec_linear(inputs.encoder_outputs)
            utterance_encodings_lf_linear = self.lf_att_vec_linear(inputs.encoder_outputs)
        else:
            utterance_encodings_sketch_linear = inputs.encoder_outputs
            utterance_encodings_lf_linear = inputs.encoder_outputs

        if self.args.col_gate_utter:
            col_appear_mask = np.zeros((1, len(inputs.column_input)), dtype=np.float32)

        dec_init_vec = final_utterance_state
        if self.args.enable_copy and turn_level > 0:
            previous_query_actions = torch.cat([a[:(a[:, 4].sum() - 1)] for a in pre_target_actions], 0)
            prev_col_ids, prev_table_ids = self.get_schema_ids(previous_query_actions)

        # >>>>>>>>>>>>>>>>>>>>>>>>>>> decode only skeleton <<<<<<<<<<<<<<<<<<<<<<<<<<
        h_tm1 = dec_init_vec
        att_tm1 = torch.zeros([self.args.hidden_size], device=self.device, requires_grad=False)
        #
        # group_loss = self.group_by_layer(inputs.is_group_by, inputs.inter_mem[-1].unsqueeze(0))
        # loss += group_loss
        group_loss = 0
        # generate target action sequence
        rule = RuleGenerator()
        target_action_embeddings = self.rule_embedding(inputs.tgt_actions[:, 1]).squeeze(1)
        current_frontier_node_type = inputs.tgt_actions[:, 0]
        current_frontier_type_embeddings = self.rule_type_embedding(current_frontier_node_type).squeeze(1)

        tar_idx = 0
        while tar_idx < self.args.max_action_seq_length - 1:
            if inputs.tgt_actions[tar_idx, 1] == EOS_id:
                # sketch_loss = sketch_loss / tar_idx
                break

            if tar_idx == 0:
                target_action = inputs.tgt_actions[tar_idx + 1, :]
                combined_embedding = torch.zeros(self.input_dim, requires_grad=False, device=device)
            else:
                # shape: [batch_size, hidden_size]

                decoder_input_embedd = target_action_embeddings[tar_idx, :]
                current_frontier_type_embedd = current_frontier_type_embeddings[tar_idx, :]
                target_action = inputs.tgt_actions[tar_idx + 1, :]

                combined_embedding = torch.cat((decoder_input_embedd, att_tm1.squeeze(), current_frontier_type_embedd))

            schema_embedding = torch.cat([inputs.col_embedding, inputs.table_embedding], 1)

            if self.args.use_diff_decoder_att:
                (h_t, cell_t), att_t, aw, alpha_q, copy_switch = self.lstm_decode(combined_embedding.unsqueeze(0),
                                                                                  h_tm1, turn_level,
                                                                                  utterance_encodings_sketch_linear,
                                                                                  schema_embedding,
                                                                                  previous_query_embedd,
                                                                                  self.decoder_lstm,
                                                                                  self.sketch_decoder_src_attention,
                                                                                  self.sketch_decoder_schema_attention,
                                                                                  self.decoder_query_attention,
                                                                                  self.decoder_query_attention_copy,
                                                                                  self.att_sketch_linear,
                                                                                  self.att_copy_sketch_linear)
            else:
                (h_t, cell_t), att_t, aw, alpha_q, copy_switch = self.lstm_decode(combined_embedding.unsqueeze(0),
                                                                                  h_tm1, turn_level,
                                                                                  utterance_encodings_sketch_linear,
                                                                                  schema_embedding,
                                                                                  previous_query_embedd,
                                                                                  self.decoder_lstm,
                                                                                  self.decoder_src_attention,
                                                                                  self.decoder_schema_attention,
                                                                                  self.decoder_query_attention_copy,
                                                                                  self.decoder_query_attention,
                                                                                  self.att_vec_linear,
                                                                                  self.att_copy_linear)

            query_probs = self.score_previous_query(previous_query_embedd, att_t) if turn_level > 0 else None
            # query_probs = alpha_q
            # print(query_probs.size())
            # print(alpha_q.size())
            loss_fct = nn.NLLLoss()
            # shape:[max_col_input_len, hidden_size]
            action_type = rule.determine_type_of_node(int(target_action[1]))
            if action_type == RULE_TYPE:

                rule_logits = torch.matmul(self.rule_embedding.weight,
                                           self.query_vec_to_action_embed(att_t).transpose(0, 1))
                rule_probs = F.softmax(rule_logits.transpose(0, 1), dim=1)
                # print(rule_probs.size())
                if self.args.enable_copy and turn_level > 0:
                    # print(query_probs.size())
                    indices = []
                    for i in list(range(self.rule_size)):
                        indices.append(torch.where(previous_query_actions[:, 1] == i)[0])
                    copy_probs = []
                    for idx, i in enumerate(indices):
                        copy_probs.append(torch.index_select(query_probs, 2, i).sum())

                    copy_probs = torch.stack(copy_probs)
                    rule_probs = torch.add(rule_probs * (1 - copy_switch), copy_probs.unsqueeze(0) * copy_switch)
                    # rule_probs += noise
                    rule_probs = torch.log(rule_probs)

                    rule_loss = loss_fct(rule_probs,
                                         torch.tensor([target_action[1].item()], dtype=torch.long, device=device))

                else:
                    # rule_probs = torch.log(rule_probs)
                    rule_probs = F.log_softmax(rule_logits.transpose(0, 1), dim=1)
                    rule_loss = loss_fct(rule_probs,
                                         torch.tensor([target_action[1].item()], dtype=torch.long, device=device))
                sketch_loss += rule_loss

                # if torch.isnan(sketch_loss):
                #     print(inputs.tgt_actions)
            tar_idx += 1

            h_tm1 = (h_t, cell_t)
            att_tm1 = att_t

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>> decode with schema <<<<<<<<<<<<<<<<<<<<<<<<<<<<
        h_tm1 = dec_init_vec
        att_tm1 = torch.zeros([self.args.hidden_size], device=self.device, requires_grad=False)
        table_enable = []

        tree_seq_embed = []
        tar_idx = 0
        while tar_idx < self.args.max_action_seq_length - 1:

            if inputs.tgt_actions[tar_idx, 1] == EOS_id:
                # loss = loss / tar_idx
                break

            if tar_idx == 0:
                target_action = inputs.tgt_actions[tar_idx + 1, :]
                combined_embedding = torch.zeros(self.input_dim, requires_grad=False, device=device)
                tree_seq_embed.append(torch.zeros(self.args.rule_embedding_size, device=self.device))
            else:
                # shape: [batch_size, hidden_size]
                current_frontier_type_embedd = current_frontier_type_embeddings[tar_idx, :]
                target_action = inputs.tgt_actions[tar_idx + 1, :]
                pre_action = inputs.tgt_actions[tar_idx, :]
                action_type = rule.determine_type_of_node(int(pre_action[1]))

                if self.args.tar_cnn_input:
                    if action_type == TAB_TYPE:
                        table_id = pre_action[2]
                        decoder_input_embedd = self.table_rnn_input(inputs.table_embedding[:, table_id, :].squeeze())
                    elif action_type == COL_TYPE:
                        col_id = pre_action[3]
                        decoder_input_embedd = self.column_rnn_input(inputs.col_embedding[:, col_id, :].squeeze())
                    else:
                        decoder_input_embedd = target_action_embeddings[tar_idx, :]
                else:
                    decoder_input_embedd = target_action_embeddings[tar_idx, :]
                tree_seq_embed.append(decoder_input_embedd)
                combined_embedding = torch.cat((decoder_input_embedd, att_tm1.squeeze(), current_frontier_type_embedd))
            schema_embedding = torch.cat([inputs.col_embedding, inputs.table_embedding], 1)

            (h_t, cell_t), att_t, aw, alpha_q, copy_switch = self.lstm_decode(combined_embedding.unsqueeze(0),
                                                                                 h_tm1, turn_level,
                                                                                 utterance_encodings_lf_linear,
                                                                                 schema_embedding,
                                                                                 previous_query_embedd,
                                                                                 self.decoder_lstm,
                                                                                 self.decoder_src_attention,
                                                                                 self.decoder_schema_attention,
                                                                                 self.decoder_query_attention,
                                                                                 self.decoder_query_attention_copy,
                                                                                 self.att_vec_linear,
                                                                                 self.att_copy_linear)

            loss_fct = nn.NLLLoss()

            action_type = rule.determine_type_of_node(int(target_action[1]))

            query_probs = self.score_previous_query(previous_query_embedd, att_t) if turn_level > 0 else None
            # query_probs = alpha_q
            if action_type == RULE_TYPE:
                rule_logits = torch.matmul(self.rule_embedding.weight,
                                           self.query_vec_to_action_embed(att_t).transpose(0, 1))
                rule_probs = F.softmax(rule_logits.transpose(0, 1), dim=1)

                if self.args.enable_copy and turn_level > 0:
                    indices = []
                    for i in list(range(self.rule_size)):
                        indices.append(torch.where(previous_query_actions[:, 1] == i)[0])
                    copy_probs = []
                    for idx, i in enumerate(indices):
                        copy_probs.append(torch.index_select(query_probs, 2, i).sum())

                    copy_probs = torch.stack(copy_probs)
                    rule_probs = torch.add(rule_probs * (1 - copy_switch), copy_probs.unsqueeze(0) * copy_switch)
                    # rule_probs += noise
                    rule_probs = torch.log(rule_probs)
                    rule_loss = loss_fct(rule_probs,
                                         torch.tensor([target_action[1].item()], dtype=torch.long, device=device))

                else:
                    rule_probs = F.log_softmax(rule_logits.transpose(0, 1), dim=1)
                    rule_loss = loss_fct(rule_probs,
                                         torch.tensor([target_action[1].item()], dtype=torch.long, device=device))
                loss += rule_loss

            elif action_type == COL_TYPE:
                col_appear_mask_val = torch.from_numpy(col_appear_mask)
                if self.cuda:
                    col_appear_mask_val = col_appear_mask_val.to(self.device)
                if self.args.column_pointer:
                    gate = torch.sigmoid(self.prob_att(att_t))
                    # this equation can be found in the IRNet-Paper, at the end of chapter 2.
                    # See the comments in the paper.
                    # noinspection PyCallingNonCallable
                    weights = self.column_pointer_net(src_encodings=inputs.col_embedding,
                                                      query_vec=att_t.unsqueeze(0),
                                                      src_token_mask=None) * col_appear_mask_val * gate + \
                              self.column_pointer_net(src_encodings=inputs.col_embedding,
                                                      query_vec=att_t.unsqueeze(0),
                                                      src_token_mask=None) * (1 - col_appear_mask_val) * (
                                      1 - gate)
                else:
                    # remember: a pointer network basically just selecting a column from "table_embedding".
                    # It is a simplified attention mechanism
                    # noinspection PyCallingNonCallable
                    weights = self.column_pointer_net(src_encodings=inputs.col_embedding,
                                                      query_vec=att_t.unsqueeze(0),
                                                      src_token_mask=None)
                    # weights.data.masked_fill_(col_padding_mask.to(torch.bool), -float('inf'))

                col_probs = F.softmax(weights.squeeze(1), dim=-1)
                if self.args.enable_copy and turn_level > 0:
                    indices = []
                    # prev_col_ids = torch.tensor(prev_col_ids)
                    for i in list(range(inputs.col_embedding.size()[1])):
                        indices.append(torch.where(prev_col_ids == i)[0])
                    copy_probs = []
                    for idx, i in enumerate(indices):
                        copy_probs.append(torch.index_select(query_probs, 2, i).sum())

                    copy_probs = torch.stack(copy_probs)
                    probs = torch.add(col_probs * (1 - copy_switch), copy_probs.unsqueeze(0) * copy_switch)
                    # probs += noise
                    probs = torch.log(probs + 1e-6)
                    col_loss = loss_fct(probs,
                                        torch.tensor([target_action[3]], dtype=torch.long, device=device))
                    # col_loss = loss_fct(probs, torch.tensor(target_col_indexes, dtype=torch.long, device=device))
                else:
                    col_probs = F.log_softmax(weights.squeeze(1), dim=-1)
                    col_loss = loss_fct(col_probs,
                                        torch.tensor([target_action[3]], dtype=torch.long, device=device))

                loss += col_loss
                col_appear_mask[0, target_action[3]] = 1
                table_enable = inputs.col_table_dict[target_action[3].item()]

            elif action_type == TAB_TYPE:
                table_mask_copy = torch.ones_like(inputs.table_mask)
                # table_mask_copy = table_padding_mask.clone()
                for i in table_enable:
                    table_mask_copy[i] = 0
                # noinspection PyCallingNonCallable
                table_weights = self.table_pointer_net(src_encodings=inputs.table_embedding,
                                                       query_vec=att_t.unsqueeze(0),
                                                       src_token_mask=None)
                tab_probs = F.softmax(table_weights.squeeze(1), dim=-1)
                # print(att_t)
                if self.args.enable_copy and turn_level > 0 and len(prev_table_ids) > 0:
                    indices = []
                    # prev_table_ids = torch.tensor(prev_table_ids)
                    for i in list(range(inputs.table_embedding.size()[1])):
                        indices.append(torch.where(prev_table_ids == i)[0])
                    copy_probs = []
                    for idx, i in enumerate(indices):
                        copy_probs.append(torch.index_select(query_probs, 2, i).sum())

                    copy_probs = torch.stack(copy_probs)
                    probs = torch.add(tab_probs * (1 - copy_switch), copy_probs.unsqueeze(0) * copy_switch)
                    # probs += noise
                    probs = torch.log(probs)
                    if target_action[2] == -1:
                        tab_loss = 0
                    else:
                        tab_loss = loss_fct(probs,
                                            torch.tensor([target_action[2]], dtype=torch.long, device=device))

                else:
                    tab_probs = F.log_softmax(table_weights.squeeze(1), dim=-1)

                    if target_action[2] == -1:
                        tab_loss = 0
                    else:
                        tab_loss = loss_fct(tab_probs,
                                            torch.tensor([target_action[2]], dtype=torch.long, device=device))

                loss += tab_loss
            elif action_type == VAL_TYPE:
                # modify turn
                # last 2 turn

                # all turn
                weights_start = self.val_pointer_net_start(src_encodings=inputs.encoder_outputs.transpose(0, 1),
                                                     query_vec=att_t.unsqueeze(0),
                                                     src_token_mask=None)
                # weights_start.data.masked_fill_(src_padding_mask.to(torch.bool), -float('inf'))
                val_start_probs = F.log_softmax(weights_start.squeeze(-1), dim=-1)

                weights_end = self.val_pointer_net_end(src_encodings=inputs.encoder_outputs.transpose(0, 1),
                                                   query_vec=att_t.unsqueeze(0),
                                                   src_token_mask=None)
                # weights_end.data.masked_fill_(src_padding_mask.to(torch.bool), -float('inf'))
                val_end_probs = F.log_softmax(weights_end.squeeze(-1), dim=-1)
                # col_padding_mask[i_batch, target_action[i_batch, 3]] = 1
                # print(val_start_probs.size(), target_action[5])
                if target_action[5] != -1 and target_action[6] != -1:
                    copy_start_loss = loss_fct(val_start_probs, torch.tensor([target_action[5]], dtype=torch.long, device=device))
                    # print(val_end_probs.size(), target_action[6])
                    copy_end_loss = loss_fct(val_end_probs, torch.tensor([target_action[6]-1], dtype=torch.long, device=device))

                    loss += copy_start_loss
                    loss += copy_end_loss
                # table_enable = col_table_dict[target_action[3].item()]
            tar_idx += 1
            h_tm1 = (h_t, cell_t)
            att_tm1 = att_t

        return loss, sketch_loss, group_loss, tree_seq_embed

    def predict(self, batch, device):
        asts = []
        losses = []
        for i in range(len(batch.interactions)):
            interaction_layer_input = InteractionLayerInput()
            interaction_layer_input.init_with_batch_index(batch, i)
            ast, loss = self.predict_interaction_layer(interaction_layer_input, device)
            asts.append(ast)
            losses.append(loss)
        return asts, losses

    def predict_interaction_layer(self, inputs, device):
        asts = []
        processed_trees = []
        loss = 0
        previous_query_embedd = []
        previous_rule_seq = []
        for turn_level, utter in enumerate(inputs.interactions):
            decoder_input = self.get_decoder_input(turn_level, utter, inputs)
            final_utterance_state = self.get_decoder_init_vec(decoder_input, turn_level)
            ast, loss_u, is_group, query_history, rule_seq = self.decode_pred(decoder_input, turn_level,
                                                                              processed_trees,
                                                                              previous_query_embedd,
                                                                              previous_rule_seq,
                                                                              final_utterance_state, device)

            previous_query_embedd.append(torch.stack(query_history, 0))
            previous_rule_seq.append(rule_seq)
            loss += loss_u

            if ast is not None:
                asts.append((ast, is_group))
                # ast.pretty_print()
                print(ast)
                processed_trees.append(process_ast(ast))

        return asts, loss

    def predict_interaction_layer2(self, inputs, device):
        asts = []
        processed_trees = []
        loss = 0
        previous_query_states = []
        pre_queries = []
        input_hidden_states = []
        final_states = []
        src_lens = []
        if self.args.discourse_level_lstm:
            discourse_state, discourse_lstm_states = self._initialize_discourse_states()

        for turn_level, utter in enumerate(inputs.interactions):
            num_utterances_to_keep = min(self.args.turn_num, turn_level + 1)

            decoder_input = self.get_decoder_input2(turn_level, utter, inputs, discourse_state,
                                                    input_hidden_states, final_states, src_lens)

            final_utterance_state = self.get_decoder_init_vec(decoder_input, turn_level)

            ast, loss_u, is_group, query_history, rule_seq = self.decode_pred(decoder_input, turn_level,
                                                                              processed_trees,
                                                                              previous_query_states,
                                                                              pre_queries,
                                                                              final_utterance_state, device)

            if self.args.discourse_level_lstm:
                _, discourse_state, discourse_lstm_states = torch_utils.forward_one_multilayer(
                    self.discourse_lstms, final_utterance_state[0].squeeze(), discourse_lstm_states, 0.)

            previous_query_states, pre_queries = self.get_previous_query(previous_query_states, pre_queries,
                                                                         rule_seq)
            previous_query_states = previous_query_states[-num_utterances_to_keep:]
            pre_queries = pre_queries[-num_utterances_to_keep:]

            # previous_query_states.append(torch.stack(query_history, 0))
            # pre_queries.append(rule_seq)
            loss += loss_u

            if ast is not None:
                asts.append((ast, is_group))
                # ast.pretty_print()
                print(ast)
                processed_trees.append(process_ast(ast))

        return asts, loss

    # noinspection PyCallingNonCallable
    def decode_pred(self, inputs, turn_level, processed_trees, previous_query_embedd, previous_rule_seq, final_utterance_state, device):
        table_padding_mask = self._generate_padding_mask(inputs.table_mask)
        if self.args.use_encoding_linear:
            utterance_encodings_lf_linear = self.lf_att_vec_linear(inputs.encoder_outputs)
        else:
            utterance_encodings_lf_linear = inputs.encoder_outputs

        if self.args.col_gate_utter:
            col_appear_mask = np.zeros((1, len(inputs.column_input)), dtype=np.float32)

        dec_init_vec = final_utterance_state

        h_tm1 = dec_init_vec
        att_tm1 = torch.zeros([self.args.hidden_size], device=self.device, requires_grad=False)
        pre_action_embedd = torch.zeros(self.args.action_embed_size, device=device)
        cur_node_embed = torch.zeros(self.args.type_embed_size, device=device)

        group_logits = self.group_output(self.act(inputs.inter_mem[-1].unsqueeze(0)))
        group_prob = F.log_softmax(group_logits, dim=1)
        group_label = torch.argmax(group_prob, dim=1)
        is_group = True if group_label.item() == 1 else False

        r = RuleGenerator()
        rule_seq = []
        table_enable = []
        tree_seq_embed = []
        r_flag = False

        parent_action = [SOS_id]
        ast = TreeWithPara('Z', [])
        cur_node = ast
        cur_node.target_action = [RULE_TYPE, SOS_id, 0, 0, 1, 0, 0]
        rule_seq.append(cur_node.target_action)
        cur_node.parent_action = parent_action

        if self.args.enable_copy and turn_level > 0:
            previous_query_actions = []
            for a in previous_rule_seq:
                previous_query_actions.extend(a)
            previous_query_actions = torch.tensor(np.asarray(previous_query_actions).squeeze(), dtype=torch.long,
                                                  device=self.device)
            prev_col_ids, prev_tab_ids = self.get_schema_ids(previous_query_actions)

        tar_idx = 0
        while tar_idx < self.args.max_action_seq_length:
            tar_idx += 1
            apply_action = True
            combined_embedding = torch.cat((pre_action_embedd, att_tm1.squeeze(), cur_node_embed))
            tree_seq_embed.append(pre_action_embedd)
            if parent_action == EOS_id:
                rule_seq.append([RULE_TYPE, EOS_id, 0, 0, 1, 0, 0])
                break

            schema_embedding = torch.cat([inputs.col_embedding, inputs.table_embedding], 1)

            (h_t, cell_t), att_t, aw, alpha_q, copy_switch = self.lstm_decode(combined_embedding.unsqueeze(0),
                                                                              h_tm1, turn_level,
                                                                              utterance_encodings_lf_linear,
                                                                              schema_embedding,
                                                                              previous_query_embedd,
                                                                              self.decoder_lstm,
                                                                              self.decoder_src_attention,
                                                                              self.decoder_schema_attention,
                                                                              self.decoder_query_attention,
                                                                              self.decoder_query_attention_copy,
                                                                              self.att_vec_linear,
                                                                              self.att_copy_linear)
            # query_probs = self.score_previous_query(previous_query_embedd, att_t) if turn_level > 0 else None
            query_probs = alpha_q

            if apply_action:
                if parent_action == EOS_id:
                    rule_seq.append([RULE_TYPE, EOS_id, 0, 0, 1, 0, 0])
                    break
                else:
                    if cur_node.label() == 'T' or cur_node.label() == 'C':

                        if cur_node.label() == 'T':
                            mask_copy = torch.ones_like(table_padding_mask)
                            for i in table_enable:
                                mask_copy[i] = 0
                            weights = self.table_pointer_net(src_encodings=inputs.table_embedding,
                                                             query_vec=att_t.unsqueeze(0),
                                                             src_token_mask=mask_copy[:inputs.table_embedding.size()[1]].view(
                                                                 1, -1))
                            # probs = F.softmax(weights.squeeze(1), dim=-1)

                        elif cur_node.label() == 'C':
                            col_appear_mask_val = torch.from_numpy(col_appear_mask)
                            if self.cuda:
                                col_appear_mask_val = col_appear_mask_val.to(self.device)
                            if self.args.column_pointer:
                                gate = torch.sigmoid(self.prob_att(att_t))

                                # equation can be found in the IRNet-Paper, at the end of chapter 2. See the comments in the paper.
                                weights = self.column_pointer_net(src_encodings=inputs.col_embedding,
                                                                  query_vec=att_t.unsqueeze(0),
                                                                  src_token_mask=None) * col_appear_mask_val * gate + \
                                          self.column_pointer_net(src_encodings=inputs.col_embedding,
                                                                  query_vec=att_t.unsqueeze(0),
                                                                  src_token_mask=None) * (
                                                  1 - col_appear_mask_val) * (
                                                  1 - gate)
                            else:
                                # remember: a pointer network basically just selecting a column from "table_embedding". It is a simplified attention mechanism
                                weights = self.column_pointer_net(src_encodings=inputs.col_embedding,
                                                                  query_vec=att_t.unsqueeze(0),
                                                                  src_token_mask=None)
                        probs = F.softmax(weights.squeeze(1), dim=-1)

                        if self.args.enable_copy and turn_level > 0:
                            if cur_node.label() == 'C':
                                indices = []
                                # prev_col_ids = torch.tensor(prev_col_ids)
                                for i in list(range(inputs.col_embedding.size()[1])):
                                    indices.append(torch.where(prev_col_ids == i)[0])
                                copy_probs = []
                                for idx, i in enumerate(indices):
                                    copy_probs.append(torch.index_select(query_probs, 2, i).sum())

                                copy_probs = torch.stack(copy_probs)
                                probs = torch.add(probs * (1 - copy_switch), copy_probs.unsqueeze(0) * copy_switch)
                                # probs = torch.log(probs)
                                schema_id = torch.argmax(probs, dim=1)
                                schema_id = schema_id.item()
                            elif cur_node.label() == 'T' and len(prev_tab_ids) > 0:
                                indices = []
                                for i in list(range(inputs.table_embedding.size()[1])):
                                    if i in table_enable:
                                        indices.append(torch.where(prev_tab_ids == i)[0])
                                    else:
                                        indices.append(torch.tensor([], device=self.device, dtype=torch.long))

                                copy_probs = []
                                for idx, i in enumerate(indices):
                                    copy_probs.append(torch.index_select(query_probs, 2, i).sum())

                                copy_probs = torch.stack(copy_probs)
                                probs = torch.add(probs * (1 - copy_switch), copy_probs.unsqueeze(0) * copy_switch)
                                schema_id = torch.argmax(probs, dim=1)
                                schema_id = schema_id.item()
                        else:

                            probs = torch.log(probs)
                            schema_id = torch.argmax(probs, dim=1)
                            schema_id = schema_id.item()

                        if cur_node.label() == 'C':
                            col_appear_mask[0, schema_id] = 1
                            table_enable = inputs.col_table_dict[schema_id]

                        cur_node.append(schema_id)
                        # if self.args.tar_cnn_input:
                        #     if cur_node.label() == 'T':
                        #
                        #         pre_action_embedd = self.table_rnn_input(table_embedding[:, schema_id, :].squeeze())
                        #     else:
                        #         # col_id = pre_action[pre_action3]
                        #         pre_action_embedd = self.column_rnn_input(col_embedding[:, schema_id, :].squeeze())
                        # print(tmp_cur_node)
                        cur_node.visited = True
                        cur_rule_id = r.get_table_rule_id() if cur_node.label() == 'T' else r.get_column_rule_id()
                        cur_node_type = 3 if cur_node.label() == 'T' else 2
                        cur_node.set_rule_id(cur_rule_id)
                        pre_action = cur_node.rule_id
                        if self.args.tar_cnn_input:
                            if cur_node.label() == 'T':

                                pre_action_embedd = self.table_rnn_input(inputs.table_embedding[:, schema_id, :].squeeze())
                            else:
                                pre_action_embedd = self.column_rnn_input(inputs.col_embedding[:, schema_id, :].squeeze())
                        else:
                            pre_action_embedd = self.rule_embedding(torch.tensor(pre_action, device=self.device))

                        cur_node.target_action = [cur_node_type, pre_action, schema_id, schema_id, 1, -1, -1]
                        rule_seq.append(cur_node.target_action)
                        cur_node.parent_action = get_parent(ast, cur_node).rule_id \
                            if get_parent(ast, cur_node) is not None \
                            else EOS_id

                        if cur_node.is_all_visited():
                            cur_node = TreeWithPara.next_unvisited(ast, cur_node)

                        else:
                            left_most_node = cur_node.left_most_child_unvisited()
                            cur_node = left_most_node
                        if get_parent(ast, cur_node) is not None:
                            parent_action = get_parent(ast, cur_node).rule_id

                        else:
                            parent_action = EOS_id

                        current_frontier_node_type = r.determine_type_of_node(pre_action)
                        cur_node_embed = self.rule_type_embedding(
                            torch.tensor(current_frontier_node_type, device=self.device)).squeeze()

                    elif cur_node.label() == 'Y':

                        # weights_start = self.val_pointer_network(
                        #     src_encodings=utterance_encodings_lf_linear,
                        #     query_vec=att_t.unsqueeze(0),
                        #     src_token_mask=None)
                        # # weights_start.data.masked_fill_(src_padding_mask.to(torch.bool), -float('inf'))
                        # val_start_probs = F.log_softmax(weights_start.squeeze(), dim=-1)
                        #
                        # weights_end = self.val_pointer_network(
                        #     src_encodings=utterance_encodings_lf_linear,
                        #     query_vec=att_t.unsqueeze(0),
                        #     src_token_mask=None)
                        #
                        # # weights_end.data.masked_fill_(src_padding_mask.to(torch.bool), -float('inf'))
                        # val_end_probs = F.log_softmax(weights_end.squeeze(), dim=-1)
                        # col_padding_mask[i_batch, target_action[i_batch, 3]] = 1
                        # mask_copy = table_mask.clone()

                        weights_start = self.val_pointer_net_start(src_encodings=inputs.encoder_outputs.transpose(0, 1),
                                                                   query_vec=att_t.unsqueeze(0),
                                                                   src_token_mask=None)
                        # weights_start.data.masked_fill_(src_padding_mask.to(torch.bool), -float('inf'))
                        val_start_probs = F.log_softmax(weights_start.squeeze(-1), dim=-1)

                        weights_end = self.val_pointer_net_end(src_encodings=inputs.encoder_outputs.transpose(0, 1),
                                                               query_vec=att_t.unsqueeze(0),
                                                               src_token_mask=None)
                        # weights_end.data.masked_fill_(src_padding_mask.to(torch.bool), -float('inf'))
                        val_end_probs = F.log_softmax(weights_end.squeeze(-1), dim=-1)
                        # col_padding_mask[i_batch, target_action[i_batch, 3]] = 1

                        start_id = torch.argmax(val_start_probs, dim=-1)
                        # print(start_id)

                        end_id = torch.argmax(val_end_probs, dim=-1)
                        end_id = end_id.item()
                        start_id = start_id.item()
                        # end_id = 1
                        # start_id = 1
                        # PUT HERE REAL BEAM SEARCH OF TOP

                        cur_node = cur_node
                        cur_node.append(str((start_id, end_id)))
                        cur_node.visited = True
                        cur_rule_id = r.get_value_rule_id()
                        tmp_cur_node_type = 4
                        cur_node.set_rule_id(cur_rule_id)
                        pre_action = cur_node.rule_id
                        cur_node.target_action = [tmp_cur_node_type, pre_action, 0, 0, 1, start_id, end_id]
                        rule_seq.append(cur_node.target_action)
                        cur_node.parent_action = get_parent(ast, cur_node).rule_id \
                            if get_parent(ast, cur_node) is not None else EOS_id
                        # TODO ???注意这里
                        # rule_seq.append(r.get_value_rule_id())

                        if cur_node.is_all_visited():
                            cur_node = TreeWithPara.next_unvisited(ast, cur_node)

                        else:
                            left_most_node = cur_node.left_most_child_unvisited()
                            cur_node = left_most_node
                        # tmp_cur_node = TreeWithPara.next_unvisited(tmp_cur_node)
                        if get_parent(ast, cur_node) is not None:
                            parent_action = get_parent(ast, cur_node).rule_id
                        else:
                            parent_action = EOS_id
                        current_frontier_node_type = r.determine_type_of_node(pre_action)

                        cur_node_embed = self.rule_type_embedding(
                            torch.tensor(current_frontier_node_type, device=self.device)).squeeze()

                        pre_action_embedd = self.rule_embedding(torch.tensor(pre_action, device=self.device))

                        # pre_action_embedd = pre_action_embedd + self.schema_output(table_embedding[schema_id])
                        curnode_id = 0

                    else:
                        if cur_node.label() == 'Filter':
                            if not r_flag:

                                filtered_rule_set = r.get_rule_by_nonterminals(lhs=cur_node.label())
                            else:
                                filtered_rule_set = r.get_non_recursive_r()
                        # elif cur_node.label() == 'R':
                        #     if not r_flag:
                        #
                        #         filtered_rule_set = r.get_rule_by_nonterminals(lhs=cur_node.label())
                        #     else:
                        #         filtered_rule_set = r.get_non_recursive_r()

                        else:
                            filtered_rule_set = r.get_rule_by_nonterminals(lhs=cur_node.label())

                        filtered_rule_set_keys = list(filtered_rule_set.keys())

                        filtered_rule_embedding = self.rule_embedding(
                            torch.tensor(filtered_rule_set_keys, dtype=torch.long, device=device))
                        # [filtered_length, 1]
                        rule_logits = torch.matmul(filtered_rule_embedding,
                                                   self.query_vec_to_action_embed(att_t).transpose(0, 1))
                        rule_prob = F.softmax(rule_logits.transpose(0, 1), dim=1)
                        if self.args.enable_copy and turn_level > 0:

                            indices = []
                            for i in filtered_rule_set_keys:
                                indices.append(torch.where(previous_query_actions[:, 1] == i)[0])

                            copy_probs = []
                            for idx, i in enumerate(indices):
                                copy_probs.append(torch.index_select(query_probs, 2, i).sum())

                            copy_probs = torch.stack(copy_probs)
                            # print(previous_query_actions.size())
                            rule_probs = torch.add(rule_prob * (1 - copy_switch),
                                                   copy_probs.unsqueeze(0) * copy_switch)
                            # rule_probs = torch.log(rule_probs)
                            max_id = torch.argmax(rule_probs, dim=1)

                            rule_id = filtered_rule_set_keys[max_id.item()]

                        else:
                            rule_prob = torch.log(rule_prob)
                            max_id = torch.argmax(rule_prob, dim=1)

                            rule_id = filtered_rule_set_keys[max_id.item()]

                        pre_action = rule_id
                        rule = r.get_rule_by_index(rule_id)
                        if rule.lhs() == Nonterminal('Filter') and Nonterminal('R') in rule.rhs():
                            r_flag = True
                        cur_node.set_rule_id(rule_id)
                        cur_node_type = get_node_type(cur_node)
                        expand_tree(cur_node, rule, r.non_terminals)
                        cur_node.target_action = [cur_node_type, pre_action, 0, 0, 1, -1, -1]
                        rule_seq.append(cur_node.target_action)
                        tmp_parent = get_parent(ast, cur_node)
                        cur_node.parent_action = tmp_parent.rule_id \
                            if tmp_parent is not None \
                            else EOS_id

                        if cur_node.is_all_visited():
                            cur_node = TreeWithPara.next_unvisited(ast, cur_node)

                        else:
                            left_most_node = cur_node.left_most_child_unvisited()
                            cur_node = left_most_node

                        if get_parent(ast, cur_node) is not None:
                            parent_action = get_parent(ast, cur_node).rule_id
                        else:
                            parent_action = EOS_id

                        current_frontier_node_type = r.determine_type_of_node(pre_action)

                        pre_action_embedd = self.rule_embedding(torch.tensor(pre_action, device=self.device))

                        cur_node_embed = self.rule_type_embedding(
                            torch.tensor(current_frontier_node_type, device=self.device)).squeeze()

            att_tm1 = att_t
            h_tm1 = (h_t, cell_t)
        rule_seq_tensor = torch.tensor(rule_seq, device=self.device)
        return ast, 0, is_group, tree_seq_embed, rule_seq_tensor

    def predict_utterance_layer(self, src_embedding,
                                table_input, table_mask, column_input, col_mask, col_hot_type,
                                col_table_dict, turn_level, processed_trees, col_embedding, table_embedding, inter_mem,
                                device, beam_size):

        # create input mask

        # max_src_seq_len = self.args.max_src_seq_length
        # max_tar_seq_len = self.args.max_action_seq_length
        # src_seq_mask = self._generate_square_subsequent_mask(max_src_seq_len)
        # src_padding_mask = self._generate_padding_mask(src_mask)
        table_padding_mask = self._generate_padding_mask(table_mask)
        col_padding_mask = self._generate_padding_mask(col_mask)

        # not batch first

        encoder_outputs, (last_state, last_cell) = self.lstm_encode(src_embedding, src_embedding.size()[1])

        if turn_level == 0:
            dec_init_vec = (last_state, torch.zeros_like(last_state))
        else:
            atten_rs = self.interaction_attention(last_state.squeeze(), inter_mem, inter_mem)
            dec_init_vec = (atten_rs[2].unsqueeze(0), torch.zeros_like(atten_rs[2]).unsqueeze(0))
        inter_mem.append(last_state.squeeze())

        encoder_outputs = self.dropout(encoder_outputs)
        col_appear_mask = np.zeros((1, col_embedding.size()[1]), dtype=np.float32)
        # col_appear_mask = torch.zeros([1, col_embedding.size()[1]], device=self.device, requires_grad=False)
        inter_ast, inter_ast_embedd = self.predict_generate_ast_embedding(turn_level, processed_trees,
                                                                          column_input, table_input, col_embedding,
                                                                          table_embedding)

        utterance_encodings_lf_linear = self.att_lf_linear(encoder_outputs)

        group_logits = self.group_output(F.relu(dec_init_vec[0]))
        group_prob = F.softmax(group_logits, dim=1)
        group_label = torch.argmax(group_prob, dim=1)
        is_group = True if group_label.item() == 1 else False

        copy_logits = self.copy_out(F.relu(dec_init_vec[0]))
        copy_prob = F.softmax(copy_logits, dim=1)
        copy_label = torch.argmax(copy_prob, dim=1)
        is_copy = True if copy_label.item() == 1 else False

        pre_action = [SOS_id]
        parent_action = [SOS_id]
        current_frontier_node_type = [RULE_TYPE]

        r = RuleGenerator()
        rule_seq = []
        ast = TreeWithPara('Z', [])
        cur_ast_copy = copy.deepcopy(ast)
        cur_node = cur_ast_copy
        curnode_id = None
        cur_node.target_action = [RULE_TYPE, pre_action, 0, 0, 1, 0, 0]
        cur_node.parent_action = parent_action
        end_flag = False
        filter_flag = False

        topk = 1
        t = 0
        # start beam search
        # Number of sentence to generate
        endnodes = []
        number_required = min((topk + 1), topk - len(endnodes))
        # dec_init_vec = self.init_decoder_state(last_cell)
        h_tm1 = dec_init_vec
        att_tm1 = torch.zeros([self.args.hidden_size], device=self.device)
        # combined_embedding = torch.cat((pre_action_embed, att_tm1.squeeze(), current_frontier_node_type_embed))
        # decoder_input_n = []
        # decoder_input_n.append(combined_embedding)
        table_enable = []
        # pre_action_embedd = self.rule_embedding(torch.tensor(pre_action, device=self.device)).squeeze()
        pre_action_embedd = torch.zeros(self.args.action_embed_size, device=device)
        cur_node_embed = torch.zeros(self.args.type_embed_size, device=device)
        node = BeamSearchNode(decoder_input=att_tm1, pre_node=pre_action, parent_node=parent_action,
                              curnode_type=cur_node_embed, logProb=0, length=1, ast=cur_ast_copy,
                              col_mask=col_padding_mask, table_enable=table_enable, curnode_id=curnode_id,
                              pre_action_embedding=pre_action_embedd, cur_node=cur_node, h_tm1=h_tm1,
                              col_appear_mask=col_appear_mask,
                              filter_flag=filter_flag)
        nodes = PriorityQueue()
        # start the queue
        nodes.put((-node.eval(), node))
        qsize = 1

        while qsize < 100000:
            # print(qsize)
            # print('l')
            apply_action = True
            # fetch the best node
            score, n = nodes.get()
            # print('after')

            decoder_input = n.decoder_input

            parent_action = n.parent_node
            pre_action = n.pre_node
            cur_node = n.cur_node
            current_frontier_type_embedd = n.curnode_type
            beam_col_mask = n.col_mask
            beam_table_enable = n.table_enable
            length = n.leng
            cur_ast = n.ast
            pre_action_embedd = n.pre_action_embedding
            h_tm1 = n.h_tm1
            col_appear_mask = n.col_appear_mask
            filter_flag = n.filter_flag
            # print(t)
            # print(cur_ast)
            # parent_action_embedd = self.rule_embedding(torch.tensor(parent_action, device=self.device)).squeeze()

            # current_frontier_type_embedd = self.rule_type_embedding(torch.tensor(current_frontier_node_type, device=self.device)).squeeze()

            # combined_embedding = self.embedding_combine(torch.cat((pre_action_embedd, parent_action_embedd), -1))

            # combined_embedding = combined_embedding + current_frontier_type_embedd
            combined_embedding = torch.cat((pre_action_embedd, decoder_input.squeeze(), current_frontier_type_embedd))
            # combined_embedding = combined_embedding.unsqueeze(1)
            # decoder_input.append(combined_embedding)
            # decoder_input_embedd = torch.stack(decoder_input, dim=0)
            # decoder_input_embedd = decoder_input_embedd * math.sqrt(self.args.rule_embedding_size)
            #
            # decoder_input_embedd = self.pos_decoder(decoder_input_embedd.unsqueeze(0))
            if parent_action == EOS_id:
                endnodes.append((score, n))
                # if we reached maximum # of sentences required
                if len(endnodes) >= number_required:
                    break
                else:
                    print('continue')
                    continue
            tar_seq_mask = self._generate_square_subsequent_mask(length)
            # print(decoder_input_embedd.size())
            # print(tar_seq_mask.size())
            # print(cur_ast)
            (h_t, cell_t), att_t, aw = self.lstm_decode(combined_embedding.unsqueeze(0), h_tm1, encoder_outputs,
                                                        utterance_encodings_lf_linear, self.decoder_lstm,
                                                        self.att_vec_linear, dec_init_vec[0],
                                                        src_token_mask=None, return_att_weight=True)
            if is_copy and self.args.enable_copy and (
                    cur_node.label() == 'Select' or cur_node.label() == 'Filter' or cur_node.label() == 'R') and turn_level > 0:
                # print('copy')
                copy_output = self.copy_output(att_t)
                copy_probs = F.log_softmax(copy_output, dim=-1)
                copy_output_label = torch.argmax(copy_probs, dim=-1)
                if copy_output_label.item() == 1:

                    flat_inter_ast_embedd = []

                    for i in inter_ast_embedd:
                        flat_inter_ast_embedd.extend(i)

                    flat_inter_ast_embedd = torch.stack(flat_inter_ast_embedd, dim=0)
                    flat_inter_ast_embedd = flat_inter_ast_embedd.to(self.device)
                    flat_inter_ast = []
                    for i in inter_ast:
                        flat_inter_ast.extend(i)
                    ast_mask = torch.ones(len(flat_inter_ast), device=device)
                    flat_processed_trees = []
                    for i in processed_trees:
                        flat_processed_trees.extend(i)
                    # filter_rule = r.get_rule_by_nonterminals(lhs=cur_node.label())
                    # filtered_rule_keys = list(filter_rule.keys())

                    for i, f_a in enumerate(flat_inter_ast):
                        if f_a.label() == cur_node.label():
                            apply_action = False
                            ast_mask[i] = 0

                    if apply_action:
                        pass
                    else:

                        # ast_output = self.ast_output(output.transpose(0, 1)[:, -1, :])
                        weights = self.copy_ast_pointer_net(src_encodings=flat_inter_ast_embedd.unsqueeze(0),
                                                            query_vec=att_t.unsqueeze(0),
                                                            src_token_mask=ast_mask.unsqueeze(0))
                        copy_ast_probs = F.log_softmax(weights.squeeze(1), dim=-1)
                        n_ast = ast_mask.eq(0).sum().item()
                        n_beam = n_ast if n_ast < beam_size else beam_size
                        # print(n_beam)
                        log_prob, indexes = torch.topk(copy_ast_probs, n_beam, dim=1)
                        nextnodes = []
                        for new_k in range(n_beam):
                            cur_ast_copy = deep_copy_ast(cur_ast)
                            tmp_cur_node = get_cur_node(cur_ast_copy)
                            decoder_input_copy = copy.deepcopy(decoder_input)
                            # tmp_cur_id = list(cur_ast_copy.subtrees()).index(cur_node)
                            # tmp_cur_node = list(cur_ast_copy.subtrees())[tmp_cur_id]
                            ast_id = indexes[0, new_k].item()
                            log_p = log_prob[0, new_k].item()
                            # print(log_p)
                            # ast_id = torch.argmax(copy_ast_probs, dim=-1)
                            copied_ast = flat_inter_ast[ast_id]
                            # print(copied_ast)
                            # copied_ast = deep_copy_ast_b(copied_ast, tmp_cur_node)
                            copied_ast = deep_copy_ast(copied_ast)
                            # copied_ast._parent = tmp_cur_node
                            for a in list(copied_ast.subtrees()):
                                a.visited = True

                            for i in list(copied_ast):
                                if isinstance(i, str):
                                    tmp_cur_node.append(i)

                                else:
                                    tmp_i = deep_copy_ast(i)

                                    tmp_cur_node.append(tmp_i)
                                    # tmp_i._parent = tmp_cur_node
                            tmp_cur_node.target_action = copied_ast.target_action
                            tmp_cur_node.parent_action = copied_ast.parent_action
                            tmp_cur_node.set_rule_id(copied_ast.rule_id)
                            # tmp_cur_node = TreeWithPara.next_unvisited(cur_ast_copy, tmp_cur_node)
                            # curnode_id = list(cur_ast_copy.subtrees()).index(tmp_cur_node)
                            pro_tr = flat_processed_trees[ast_id]
                            # copied_ast_action = trans_ast_to_action_seq(copied_ast)
                            # pre_actions = torch.from_numpy(copied_ast_action[1])

                            # print(copied_ast_action[2])
                            parent_actions = pro_tr[1][2]
                            target_actions = pro_tr[1][0]
                            # print(parent_actions)
                            # print(target_actions.size)
                            print(len(decoder_input_copy))
                            print(length)
                            for i in range(len(target_actions) - 1):
                                current_frontier_type_embedd = self.rule_type_embedding(
                                    torch.tensor(target_actions[i, 0], device=self.device))
                                decoder_input_embedd = self.rule_embedding(
                                    torch.tensor(target_actions[i, 1], device=self.device))
                                parent_action_embedd = self.rule_embedding(
                                    torch.tensor(parent_actions[i], device=self.device))
                                action_type = r.determine_type_of_node(int(target_actions[i, 1]))
                                # if action_type == TAB_TYPE:
                                #     table_id = target_actions[i, 2]
                                #
                                #     decoder_input_embedd = decoder_input_embedd + self.schema_input(
                                #         table_embedding[:, table_id, :].squeeze())
                                # elif action_type == COL_TYPE:
                                #     col_id = target_actions[i, 3]
                                #     decoder_input_embedd = decoder_input_embedd + self.schema_input(
                                #         col_embedding[:, col_id, :].squeeze())
                                if i == len(target_actions) - 1:
                                    pre_action_embedd = decoder_input_embedd
                                    break
                                combined_embedding = torch.cat(
                                    (decoder_input_embedd, att_t.squeeze(), current_frontier_type_embedd))
                                (h_t, cell_t), att_t, aw = self.lstm_decode(combined_embedding.unsqueeze(0), h_tm1,
                                                                            encoder_outputs,
                                                                            utterance_encodings_lf_linear,
                                                                            self.decoder_lstm,
                                                                            self.att_vec_linear,
                                                                            dec_init_vec[0],
                                                                            src_token_mask=None, return_att_weight=True)
                                h_tm1 = (h_t, cell_t)
                                # combined_embedding = self.embedding_combine(
                                #     torch.cat((decoder_input_embedd, parent_action_embedd)))
                                # combined_embedding = combined_embedding + current_frontier_type_embedd
                                # batch_size, 1, hidden_size
                                # decoder_input_copy.append(combined_embedding)
                            # print(len(decoder_input))
                            pre_action = target_actions[-1, 1]
                            # print(cur_ast_copy)
                            # print(tmp_cur_node)
                            if tmp_cur_node.is_all_visited():
                                tmp_cur_node = TreeWithPara.next_unvisited(cur_ast_copy, tmp_cur_node)

                            else:
                                left_most_node = tmp_cur_node.left_most_child_unvisited()
                                tmp_cur_node = left_most_node

                            if get_parent(cur_ast_copy, tmp_cur_node) is not None:
                                parent_action = get_parent(cur_ast_copy, tmp_cur_node).rule_id
                            else:
                                parent_action = EOS_id
                            # current_frontier_node_type = get_node_type(tmp_cur_node)
                            current_frontier_node_type = r.determine_type_of_node(pre_action)
                            self.rule_type_embedding(torch.tensor(target_actions[i, 0], device=self.device))
                            tmp_cur_node.parent_action = get_parent(cur_ast_copy, tmp_cur_node).rule_id \
                                if get_parent(cur_ast_copy, tmp_cur_node) is not None else EOS_id
                            # tmp_cur_node.parent().rule_id if tmp_cur_node.parent() is not None else EOS_id
                            print('resf')
                            node = BeamSearchNode(decoder_input=copy.deepcopy(att_t), pre_node=pre_action,
                                                  parent_node=parent_action,
                                                  curnode_type=self.rule_type_embedding(
                                                      torch.tensor(current_frontier_node_type,
                                                                   device=self.device)).squeeze(),
                                                  logProb=n.logp + log_p,
                                                  length=n.leng + len(target_actions), ast=cur_ast_copy,
                                                  col_mask=col_padding_mask, table_enable=[], curnode_id=curnode_id,
                                                  pre_action_embedding=pre_action_embedd, cur_node=tmp_cur_node,
                                                  h_tm1=copy.deepcopy((h_t, cell_t)),
                                                  col_appear_mask=copy.deepcopy(col_appear_mask),
                                                  filter_flag=filter_flag)
                            score = -node.eval()
                            nextnodes.append((score, node))
                        for i in range(len(nextnodes)):
                            score, nt = nextnodes[i]
                            # print(score)
                            nodes.put((score, nt))
                            # increase qsize
                        qsize += len(nextnodes) - 1
                        t += len(target_actions)

            if apply_action:
                # rule_output = self.rule_output(output.transpose(0, 1)[:, -1, :])
                # schema_output = self.schema_output(output.transpose(0, 1)[:, -1, :])
                # val_output_end = self.val_output_end(output.transpose(0, 1)[:, -1, :])
                # val_output_start = self.val_output_start(output.transpose(0, 1)[:, -1, :])
                if parent_action == EOS_id:
                    print('eos')
                    rule_seq.append(EOS_id)
                    break
                elif end_flag:
                    break
                else:
                    if cur_node.label() == 'T' or cur_node.label() == 'C':
                        # print('schema')
                        # print(cur_node)

                        if cur_node.label() == 'T':
                            mask_copy = torch.ones_like(table_padding_mask)
                            # table_enable = col_table_dict[tab_id]
                            for i in beam_table_enable:
                                mask_copy[i] = 0
                            # if self.args.use_bert:
                            table_len = table_embedding.size()[1]
                            weights = self.table_pointer_net(src_encodings=table_embedding,
                                                             query_vec=att_t.unsqueeze(0),
                                                             src_token_mask=mask_copy[:table_len].view(1, -1))
                            weights.data.masked_fill_(mask_copy[:table_len].to(torch.uint8), -float('inf'))

                            # weights = self.table_pointer_net(src_encodings=table_embedding,
                            #                                  query_vec=att_t.unsqueeze(0),
                            #                                  src_token_mask=mask_copy.view(1, -1))
                            #
                            # weights.data.masked_fill_(mask_copy.to(torch.bool), -float('inf'))
                            probs = F.log_softmax(weights.squeeze(1), dim=-1)
                            # real_input_len = torch.sum(mask_copy)
                            real_input_len = len(beam_table_enable)
                            # mask_copy = table_mask.clone()
                        elif cur_node.label() == 'C':
                            mask_copy = beam_col_mask.clone()

                            # weights = self.column_pointer_net(src_encodings=col_embedding,
                            #                                   query_vec=att_t.unsqueeze(0),
                            #
                            #                                   src_token_mask=None)
                            col_appear_mask_val = torch.from_numpy(col_appear_mask)
                            if self.cuda:
                                col_appear_mask_val = col_appear_mask_val.to(self.device)
                            if self.args.column_pointer:
                                gate = torch.sigmoid(self.prob_att(att_t))
                                # this equation can be found in the IRNet-Paper, at the end of chapter 2. See the comments in the paper.
                                weights = self.column_pointer_net(src_encodings=col_embedding,
                                                                  query_vec=att_t.unsqueeze(0),
                                                                  src_token_mask=None) * col_appear_mask_val * gate + \
                                          self.column_pointer_net(src_encodings=col_embedding,
                                                                  query_vec=att_t.unsqueeze(0),
                                                                  src_token_mask=None) * (
                                                  1 - col_appear_mask_val) * (
                                                  1 - gate)
                            else:
                                # remember: a pointer network basically just selecting a column from "table_embedding". It is a simplified attention mechanism
                                weights = self.column_pointer_net(src_encodings=col_embedding,
                                                                  query_vec=att_t.unsqueeze(0),
                                                                  src_token_mask=None)
                                # weights.data.masked_fill_(col_padding_mask.to(torch.bool), -float('inf'))
                                # weights.data.masked_fill_(col_padding_mask.to(torch.bool), -float('inf'))
                            # else:
                            #     weights = self.column_pointer_net(src_encodings=col_embedding,
                            #                                       query_vec=att_t.unsqueeze(0),
                            #                                       src_token_mask=mask_copy.view(1, -1))
                            #     weights.data.masked_fill_(col_padding_mask.to(torch.uint8), -float('inf'))

                            probs = F.log_softmax(weights.squeeze(1), dim=-1)
                            real_input_len = torch.sum(col_mask)
                            real_input_len = real_input_len.item()
                        n_beam = real_input_len if real_input_len < beam_size else beam_size
                        log_prob, indexes = torch.topk(probs, n_beam, dim=1)
                        # PUT HERE REAL BEAM SEARCH OF TOP
                        nextnodes = []
                        for new_k in range(n_beam):

                            cur_ast_copy = deep_copy_ast(cur_ast)
                            tmp_cur_node = get_cur_node(cur_ast_copy)
                            tmp_appear_mask = copy.deepcopy(col_appear_mask)
                            schema_id = indexes[0, new_k].item()
                            tmp_appear_mask[0, schema_id] = 1
                            log_p = log_prob[0, new_k].item()
                            if tmp_cur_node.label() == 'C':
                                # mask_copy[schema_id] = 1
                                beam_table_enable = col_table_dict[schema_id]

                            tmp_cur_node.append(schema_id)
                            # print(tmp_cur_node)
                            tmp_cur_node.visited = True
                            cur_rule_id = r.get_table_rule_id() if tmp_cur_node.label() == 'T' else r.get_column_rule_id()
                            cur_node_type = 3 if tmp_cur_node.label() == 'T' else 2
                            tmp_cur_node.set_rule_id(cur_rule_id)
                            pre_action = tmp_cur_node.rule_id
                            pre_action_embedd = self.rule_embedding(torch.tensor(pre_action, device=self.device))
                            # if tmp_cur_node.label() == 'T':
                            #     pre_action_embedd = pre_action_embedd + self.schema_input(table_embedding[:, schema_id, :].squeeze())
                            # else:
                            #     pre_action_embedd = pre_action_embedd + self.schema_input(col_embedding[:, schema_id, :].squeeze())

                            tmp_cur_node.target_action = [cur_node_type, pre_action, schema_id, schema_id, 0, -1, -1]
                            tmp_cur_node.parent_action = get_parent(cur_ast_copy, tmp_cur_node).rule_id \
                                if get_parent(cur_ast_copy, tmp_cur_node) is not None \
                                else EOS_id

                            # TODO ???注意这里
                            # i_tmp_cur_node = TreeWithPara.next_unvisited(i_tmp_cur_node)
                            # i_tmp_cur_node.set_rule_id(r.get_table_rule_id())
                            rule_seq.append(cur_rule_id)
                            if False:
                                pass
                            else:

                                # print(cur_ast)
                                # print(tmp_cur_node)
                                if tmp_cur_node.is_all_visited():
                                    tmp_cur_node = TreeWithPara.next_unvisited(cur_ast_copy, tmp_cur_node)

                                else:
                                    left_most_node = tmp_cur_node.left_most_child_unvisited()
                                    tmp_cur_node = left_most_node
                                if get_parent(cur_ast_copy, tmp_cur_node) is not None:
                                    parent_action = get_parent(cur_ast_copy, tmp_cur_node).rule_id
                                    # print(tmp_cur_node.parent().rule_id)
                                else:
                                    parent_action = EOS_id
                                # current_frontier_node_type = get_node_type(tmp_cur_node)
                                current_frontier_node_type = r.determine_type_of_node(pre_action)
                            # curnode_id = None
                            node = BeamSearchNode(decoder_input=copy.deepcopy(att_t), pre_node=pre_action,
                                                  parent_node=parent_action,
                                                  curnode_type=self.rule_type_embedding(
                                                      torch.tensor(current_frontier_node_type,
                                                                   device=self.device)).squeeze(),
                                                  logProb=n.logp + log_p,
                                                  length=n.leng + 1, ast=cur_ast_copy, col_mask=beam_col_mask.clone(),
                                                  table_enable=copy.deepcopy(beam_table_enable), curnode_id=curnode_id,
                                                  pre_action_embedding=pre_action_embedd, cur_node=tmp_cur_node,
                                                  h_tm1=copy.deepcopy((h_t, cell_t)),
                                                  col_appear_mask=tmp_appear_mask, filter_flag=filter_flag)
                            # print(log_p)
                            score = -node.eval()
                            nextnodes.append((score, node))

                            # put them into queue
                        # print(nodes)
                        for i in range(len(nextnodes)):
                            score, nt = nextnodes[i]
                            # print(score, nt)
                            nodes.put((score, nt))
                            # increase qsize
                        qsize += len(nextnodes) - 1

                    elif cur_node.label() == 'Y':
                        # if not self.args.use_bert:
                        #     weights_start = self.val_pointer_net(src_encodings=src_embedding[:,:utter_lens.item(), :].transpose(0, 1),
                        #                                          query_vec=att_t.unsqueeze(0),
                        #                                          src_token_mask=None)
                        #     # weights_start.data.masked_fill_(src_padding_mask.to(torch.bool), -float('inf'))
                        #     val_start_probs = F.log_softmax(weights_start.squeeze(1), dim=-1)
                        #
                        #     weights_end = self.val_pointer_net(src_encodings=src_embedding[:,:utter_lens.item(), :].transpose(0, 1),
                        #                                        query_vec=att_t.unsqueeze(0),
                        #                                        src_token_mask=None)
                        # else:
                        weights_start = self.val_pointer_net(
                            src_encodings=encoder_outputs.transpose(0, 1),
                            query_vec=att_t.unsqueeze(0),
                            src_token_mask=None)
                        # weights_start.data.masked_fill_(src_padding_mask.to(torch.bool), -float('inf'))
                        val_start_probs = F.log_softmax(weights_start.squeeze(1), dim=-1)

                        weights_end = self.val_pointer_net(
                            src_encodings=encoder_outputs.transpose(0, 1),
                            query_vec=att_t.unsqueeze(0),
                            src_token_mask=None)

                        # weights_end.data.masked_fill_(src_padding_mask.to(torch.bool), -float('inf'))
                        val_end_probs = F.log_softmax(weights_end.squeeze(1), dim=-1)
                        # col_padding_mask[i_batch, target_action[i_batch, 3]] = 1
                        # mask_copy = table_mask.clone()

                        n_beam = encoder_outputs.size()[0] if encoder_outputs.size()[0] < beam_size else beam_size
                        start_log_prob, start_indexes = torch.topk(val_start_probs, n_beam, dim=-1)
                        end_log_prob, end_indexes = torch.topk(val_end_probs, n_beam, dim=-1)
                        # PUT HERE REAL BEAM SEARCH OF TOP
                        nextnodes = []
                        for new_k in range(n_beam):
                            cur_ast_copy = deep_copy_ast(cur_ast)
                            tmp_cur_node = get_cur_node(cur_ast_copy)
                            tmp_appear_mask = copy.deepcopy(col_appear_mask)
                            # tmp_cur_id = list(cur_ast.subtrees()).index(cur_node)
                            # tmp_cur_node = list(cur_ast_copy.subtrees())[tmp_cur_id]
                            start_id = start_indexes[0, new_k].item()
                            end_id = end_indexes[0, new_k].item()
                            start_log_p = start_log_prob[0, new_k].item()
                            end_log_p = end_log_prob[0, new_k].item()
                            log_p = start_log_p + end_log_p
                            tmp_cur_node.append(str((start_id, end_id)))
                            tmp_cur_node.visited = True
                            cur_rule_id = r.get_value_rule_id()
                            tmp_cur_node_type = 4
                            tmp_cur_node.set_rule_id(cur_rule_id)
                            pre_action = tmp_cur_node.rule_id
                            tmp_cur_node.target_action = [tmp_cur_node_type, pre_action, 0, 0, 0, start_id, end_id]
                            tmp_cur_node.parent_action = get_parent(cur_ast_copy, tmp_cur_node).rule_id \
                                if get_parent(cur_ast_copy, tmp_cur_node) is not None else EOS_id
                            # TODO ???注意这里
                            rule_seq.append(r.get_value_rule_id())
                            if False:
                                pass
                            else:
                                if tmp_cur_node.is_all_visited():
                                    tmp_cur_node = TreeWithPara.next_unvisited(cur_ast_copy, tmp_cur_node)

                                else:
                                    left_most_node = tmp_cur_node.left_most_child_unvisited()
                                    tmp_cur_node = left_most_node
                                # tmp_cur_node = TreeWithPara.next_unvisited(tmp_cur_node)
                                if get_parent(cur_ast_copy, tmp_cur_node) is not None:
                                    parent_action = get_parent(cur_ast_copy, tmp_cur_node).rule_id
                                else:
                                    parent_action = EOS_id
                                current_frontier_node_type = r.determine_type_of_node(pre_action)

                            pre_action_embedd = self.rule_embedding(torch.tensor(pre_action, device=self.device))

                            # pre_action_embedd = pre_action_embedd + self.schema_output(table_embedding[schema_id])
                            curnode_id = 0
                            node = BeamSearchNode(decoder_input=copy.deepcopy(att_t), pre_node=pre_action,
                                                  parent_node=parent_action,
                                                  curnode_type=self.rule_type_embedding(
                                                      torch.tensor(current_frontier_node_type,
                                                                   device=self.device)).squeeze(),
                                                  logProb=n.logp + log_p,
                                                  length=n.leng + 1, ast=cur_ast_copy, col_mask=col_padding_mask,
                                                  table_enable=[], curnode_id=curnode_id,
                                                  pre_action_embedding=pre_action_embedd, cur_node=tmp_cur_node,
                                                  h_tm1=copy.deepcopy((h_t, cell_t)),
                                                  col_appear_mask=tmp_appear_mask, filter_flag=filter_flag)
                            score = -node.eval()
                            nextnodes.append((score, node))

                            # put them into queue
                        for i in range(len(nextnodes)):
                            score, nt = nextnodes[i]
                            nodes.put((score, nt))
                            # increase qsize
                        qsize += len(nextnodes) - 1

                    else:
                        # print('rule')
                        if cur_node.label() == 'Filter':
                            if not filter_flag:

                                filtered_rule_set = r.get_rule_by_nonterminals(lhs=cur_node.label())
                            else:
                                filtered_rule_set = r.get_non_recursive_filter()
                        else:
                            filtered_rule_set = r.get_rule_by_nonterminals(lhs=cur_node.label())
                            # print(filtered_rule_set)
                        # else:
                        #     filtered_rule_set = r.get_rule_by_nonterminals(lhs=rule_dict[i_pre_action.item()])
                        filtered_rule_set_keys = list(filtered_rule_set.keys())
                        # [filtered_length, hidden_size]
                        filtered_rule_embedding = self.rule_embedding(
                            torch.tensor(filtered_rule_set_keys, dtype=torch.long, device=device))
                        # [filtered_length, 1]
                        rule_logits = torch.matmul(filtered_rule_embedding,
                                                   self.query_vec_to_action_embed(att_t).transpose(0, 1))
                        rule_prob = torch.log_softmax(rule_logits.transpose(0, 1), dim=1)
                        n_beam = len(filtered_rule_set_keys) if len(filtered_rule_set_keys) < beam_size else beam_size
                        log_prob, indexes = torch.topk(rule_prob, n_beam, dim=1)
                        # PUT HERE REAL BEAM SEARCH OF TOP
                        nextnodes = []
                        # print(cur_ast)
                        # print(cur_node)
                        for new_k in range(n_beam):
                            cur_ast_copy = deep_copy_ast(cur_ast)
                            # print(cur_ast_copy)
                            tmp_cur_node = get_cur_node(cur_ast_copy)
                            tmp_appear_mask = copy.deepcopy(col_appear_mask)
                            # tmp_cur_id = list(cur_ast.subtrees()).index(cur_node)
                            # tmp_cur_node = list(cur_ast_copy.subtrees())[tmp_cur_id]
                            max_id = indexes[0, new_k]
                            log_p = log_prob[0, new_k].item()
                            rule_id = filtered_rule_set_keys[max_id.item()]
                            # print(filtered_rule_set_keys)
                            # print(rule_id)

                            pre_action = rule_id

                            # expand the C node
                            rule = r.get_rule_by_index(rule_id)
                            if rule.lhs() == Nonterminal('Filter') and (
                                    Nonterminal('R') in rule.rhs() or Nonterminal('Filter') in rule.rhs()):
                                filter_flag = True

                            rule_seq.append(rule_id)
                            tmp_cur_node.set_rule_id(rule_id)

                            cur_node_type = get_node_type(tmp_cur_node)
                            if False:
                                pass
                            else:
                                expand_tree(tmp_cur_node, rule, r.non_terminals)
                            # print(tmp_cur_node.rule_id)
                            tmp_cur_node.target_action = [cur_node_type, pre_action, 0, 0, 0, -1, -1]
                            tmp_parent = get_parent(cur_ast_copy, tmp_cur_node)
                            tmp_cur_node.parent_action = tmp_parent.rule_id \
                                if tmp_parent is not None \
                                else EOS_id

                            if tmp_cur_node.is_all_visited():
                                tmp_cur_node = TreeWithPara.next_unvisited(cur_ast_copy, tmp_cur_node)

                            else:
                                left_most_node = tmp_cur_node.left_most_child_unvisited()
                                tmp_cur_node = left_most_node
                                # l = left_most_node.label()

                            # set parent node
                            if get_parent(cur_ast_copy, tmp_cur_node) is not None:
                                parent_action = get_parent(cur_ast_copy, tmp_cur_node).rule_id
                            else:
                                parent_action = EOS_id

                            current_frontier_node_type = r.determine_type_of_node(pre_action)

                            curnode_id = 0
                            pre_action_embedd = self.rule_embedding(torch.tensor(pre_action, device=self.device))

                            # print(cur_ast)
                            node = BeamSearchNode(decoder_input=copy.deepcopy(att_t), pre_node=pre_action,
                                                  parent_node=parent_action,
                                                  curnode_type=self.rule_type_embedding(
                                                      torch.tensor(current_frontier_node_type,
                                                                   device=self.device)).squeeze(),
                                                  logProb=n.logp + log_p, length=n.leng + 1, ast=cur_ast_copy,
                                                  table_enable=[], col_mask=col_padding_mask, curnode_id=curnode_id,
                                                  pre_action_embedding=pre_action_embedd, cur_node=tmp_cur_node,
                                                  h_tm1=copy.deepcopy((h_t, cell_t)), col_appear_mask=tmp_appear_mask,
                                                  filter_flag=filter_flag)
                            score = -node.eval()
                            nextnodes.append((score, node))

                            # put them into queue
                        for i in range(len(nextnodes)):
                            score, nt = nextnodes[i]
                            nodes.put((score, nt))
                            # increase qsize
                        qsize += len(nextnodes) - 1

            t += 1
        # choose n-best paths, back trace them
        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(topk)]

        utterances = []
        losses = []
        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterances.append(n.ast)
            losses.append(n.logp)

        return utterances[0], losses[0], is_group

    def _initialize_discourse_states(self):
        discourse_state = self.initial_discourse_state
        discourse_lstm_states = []
        for lstm in self.discourse_lstms:
            hidden_size = lstm.weight_hh.size()[1]
            # if lstm.weight_hh.is_cuda:
            #     h_0 = torch.cuda.FloatTensor(1,hidden_size).fill_(0)
            #     c_0 = torch.cuda.FloatTensor(1,hidden_size).fill_(0)
            # else:
            h_0 = torch.zeros(1, hidden_size, device=self.device)
            c_0 = torch.zeros(1, hidden_size, device=self.device)
            discourse_lstm_states.append((h_0, c_0))

        return discourse_state, discourse_lstm_states

    def get_decoder_input2(self, turn_level, utter, interaction_input, discourse_state, input_hidden_states,
                           final_states, src_lens):
        # src_sent, src_sent_lens = self.get_utter_input(interaction_input.interactions, self.args.turn_num, turn_level)
        # src_sent, src_sent_lens = self.get_utter_input(interaction_input.interactions, 1, turn_level)
        src_lens.append(len(utter.src_sent))
        num_utterances_to_keep = min(self.args.turn_num, turn_level + 1)
        encoder_outputs, col_embedding, table_embedding, value_encodings, pooled_output = self.encoder(
            [utter.src_sent],
            [interaction_input.column_input],
            [interaction_input.table_input],
            [['value']])
        encoder_size = encoder_outputs.size()
        # print(discourse_state.size())
        # print(encoder_outputs.size())
        encoder_outputs = torch.cat([encoder_outputs, discourse_state.expand(encoder_size[0], encoder_size[1],
                                                                             discourse_state.size()[0])], 2)
        input_hidden_states.append(encoder_outputs)
        input_hidden_states = input_hidden_states[-num_utterances_to_keep:]
        src_lens = src_lens[-num_utterances_to_keep:]

        if self.args.use_turn_pos_enc:
            encoder_outputs = self._add_positional_embeddings(input_hidden_states, src_lens, turn_level)
        else:
            encoder_outputs = torch.cat(input_hidden_states, 0)

        if self.args.use_utterance_attention:
            schema_states = torch.cat([col_embedding, table_embedding], 1)
            utter_schema_att, _ = self.utterance_attention(encoder_outputs.transpose(0, 1),
                                                           schema_states.transpose(0, 1),
                                                           schema_states.transpose(0, 1))
            utter_schema_att = utter_schema_att.transpose(0, 1)
            encoder_outputs = encoder_outputs + utter_schema_att
        #   change
        # utterance to schema attention utter_schema_att
        if self.args.use_utter_schema_att:
            col_attn, _ = self.schema_attention(col_embedding.transpose(0, 1), encoder_outputs.transpose(0, 1),
                                                encoder_outputs.transpose(0, 1))
            table_attn, _ = self.schema_attention(table_embedding.transpose(0, 1), encoder_outputs.transpose(0, 1),
                                                  encoder_outputs.transpose(0, 1))
            col_attn = col_attn.transpose(0, 1)
            table_attn = table_attn.transpose(0, 1)
            col_embedding = col_embedding + col_attn
            table_embedding = table_embedding + table_attn

        if self.args.use_col_type:
            col_type = self.input_type([utter.col_hot_type], len(utter.col_hot_type))
            col_type_var = self.col_type(col_type)
            col_embedding = col_embedding + col_type_var
        inter_mem = [pooled_output[1].squeeze()]
        # final_states = [pooled_output.squeeze()]
        final_states.append(pooled_output)
        final_states = final_states[-num_utterances_to_keep:]
        decoder_input = DecoderInput(encoder_outputs=encoder_outputs, col_embedding=col_embedding,
                                     table_embedding=table_embedding, final_states=final_states,
                                     tgt_actions=utter.tgt_actions,
                                     interaction_input=interaction_input, group_by=utter.group_by, inter_mem=inter_mem)
        return decoder_input

    def get_decoder_input(self, turn_level, utter, interaction_input):
        if self.args.combine_encoding:
            # print(utter.src_sent)
            encoder_outputs, col_embedding, table_embedding, value_encodings, pooled_output = self.encoder(
                [utter.src_sent],
                [interaction_input.column_input],
                [interaction_input.table_input],
                [['value']])
        else:
            src_sent, src_sent_lens = self.get_utter_input(interaction_input.interactions, self.args.turn_num, turn_level)
            schema_sent, _ = self.get_utter_input(interaction_input.interactions, 1, turn_level)

            _, col_embedding, table_embedding, value_encodings, pooled_output = self.encoder(
                [schema_sent],
                [interaction_input.column_input],
                [interaction_input.table_input],
                [['value']])
            encoder_outputs, _ = self.encoder.encode_utterance([src_sent])


        # print(interaction_input.column_input)
        # print(utter.sql)
        # print(utter.src_sent)
        # print(utter.tgt_actions)
        if self.args.use_turn_pos_enc:
            encoder_outputs = self._add_positional_embeddings(encoder_outputs, src_sent_lens, turn_level)

        if self.args.use_utterance_attention:
            schema_states = torch.cat([col_embedding, table_embedding], 1)
            utter_schema_att, _ = self.utterance_attention(encoder_outputs.transpose(0, 1),
                                                           schema_states.transpose(0, 1),
                                                           schema_states.transpose(0, 1))
            utter_schema_att = utter_schema_att.transpose(0, 1)
            encoder_outputs = encoder_outputs + utter_schema_att
        #   change


        # utterance to schema attention utter_schema_att
        if self.args.use_utter_schema_att:
            col_attn, _ = self.schema_attention(col_embedding.transpose(0, 1), encoder_outputs.transpose(0, 1),
                                                encoder_outputs.transpose(0, 1))
            table_attn, _ = self.schema_attention(table_embedding.transpose(0, 1), encoder_outputs.transpose(0, 1),
                                                  encoder_outputs.transpose(0, 1))
            col_attn = col_attn.transpose(0, 1)
            table_attn = table_attn.transpose(0, 1)
            col_embedding = col_embedding + col_attn
            table_embedding = table_embedding + table_attn

        # change
        # if self.args.use_utterance_attention:
        #     encoder_outputs = encoder_outputs + utter_schema_att

        if self.args.use_col_type:
            col_type = self.input_type([utter.col_hot_type], len(utter.col_hot_type))
            col_type_var = self.col_type(col_type)
            col_embedding = col_embedding + col_type_var
        # change
        # if self.args.use_utter_schema_att:
        #     col_embedding = col_embedding + col_attn
        #     table_embedding = table_embedding + table_attn
        inter_mem = [pooled_output.squeeze()]
        final_states = [pooled_output.squeeze()]

        decoder_input = DecoderInput(encoder_outputs=encoder_outputs, col_embedding=col_embedding,
                                     table_embedding=table_embedding, final_states=final_states,
                                     tgt_actions=utter.tgt_actions,
                                     interaction_input=interaction_input, group_by=utter.group_by, inter_mem=inter_mem)

        return decoder_input

    def lstm_decode(self, x, h_tm1, turn_level, src_encodings_att_linear, schema_encoding,
                    pre_query_encoding, decoder, decoder_src_att, decoder_schema_att, decoder_query_att,
                    decoder_query_att_copy, attention_func, attention_func_copy):
        # h_t: (batch_size, hidden_size)
        h_t, cell_t = decoder(x, h_tm1)

        # ctx_t, alpha_t = self.dot_prod_attention(h_t,
        #                                              src_encodings, src_encodings_att_linear,
        #                                              mask=src_token_mask)
        att_vec = []
        att_vec_copy = []
        att_vec.append(h_t)
        ctx_t, alpha_t = decoder_src_att(h_t.unsqueeze(0), src_encodings_att_linear.transpose(0, 1),
                                         src_encodings_att_linear.transpose(0, 1))
        att_vec.append(ctx_t.squeeze(0))
        if self.args.decoder_use_schema:
            ctx_t_schema, _ = decoder_schema_att(h_t.unsqueeze(0), schema_encoding.transpose(0, 1),
                                                 schema_encoding.transpose(0, 1))
            att_vec.append(ctx_t_schema.squeeze(0))

        last_query_encoding = pre_query_encoding[-1] if turn_level > 0 else None
        query_att, _ = self.get_query_attention(h_t, last_query_encoding, turn_level, decoder_query_att)
        att_vec_copy.extend(att_vec)
        if self.args.decoder_use_query:
            att_vec.append(query_att)

        att_t = F.tanh(attention_func(torch.cat(att_vec, 1)))
        att_t = self.dropout(att_t)

        pre_query_encodings = torch.cat(pre_query_encoding, 0) if turn_level > 0 else None
        query_att_copy, alpha_q = self.get_query_attention(h_t, pre_query_encodings, turn_level, decoder_query_att_copy)
        att_vec_copy.append(query_att_copy)
        if self.args.enable_copy:
            copy_switch = torch.sigmoid(attention_func_copy(self.dropout(torch.cat(att_vec_copy, 1))))
        else:
            copy_switch = None

        return (h_t, cell_t), att_t, alpha_t, alpha_q, copy_switch

    def get_query_attention(self, h_t, query_encodings, turn_level, decoder_query_att):

        if turn_level == 0:
            att = torch.zeros_like(h_t)
            alpha_q = None
        else:

            if self.args.use_query_lstm:

                query_embed, _ = self.query_lstm_encode(query_encodings.unsqueeze(0),
                                                        query_encodings.size()[0],
                                                        self.query_encoder_lstm)
                ctx_t_query, alpha_q = decoder_query_att(h_t.unsqueeze(0), query_embed.transpose(0, 1),
                                                         query_embed.transpose(0, 1))
            else:
                ctx_t_query, alpha_q = decoder_query_att(h_t.unsqueeze(0), query_encodings.unsqueeze(1),
                                                         query_encodings.unsqueeze(1))
            att = ctx_t_query.squeeze(0)

        return att, alpha_q

    def get_decoder_init_vec(self, inputs, turn_level):
        if not self.args.use_lstm_encoder:

            if self.args.use_init_decoder_att:

                if turn_level == 0:
                    dec_init_vec = self.init_decoder_state(inputs.final_states[-1].unsqueeze(0))
                else:
                    prev_final_states = torch.stack(inputs.final_states[:-1], 0).unsqueeze(0)
                    last_final_state = inputs.final_states[-1].view(1, 1, -1)
                    att_rs, _ = self.interaction_attention(last_final_state, prev_final_states, prev_final_states)
                    # print(att_rs)
                    final_state_att = inputs.final_states[-1] + att_rs.squeeze()
                    dec_init_vec = self.init_decoder_state(final_state_att.unsqueeze(0))
            else:
                dec_init_vec = self.init_decoder_state(inputs.final_states[-1].unsqueeze(0))
        else:

            if self.args.use_init_decoder_att:
                if turn_level == 0:

                    dec_init_vec = inputs.final_states[-1]

                else:

                    final_states_h = torch.cat([i[0] for i in inputs.final_states], 0).unsqueeze(0)
                    final_states_c = torch.cat([i[1] for i in inputs.final_states], 0).unsqueeze(0)

                    att_h,_ = self.interaction_attention(inputs.final_states[-1][0].unsqueeze(0),
                                                       final_states_h, final_states_h)
                    att_c,_ = self.interaction_attention(inputs.final_states[-1][1].unsqueeze(0),
                                                       final_states_c, final_states_c)

                    init_h = inputs.final_states[-1][0] + att_h.squeeze(0)
                    init_c = inputs.final_states[-1][1] + att_c.squeeze(0)
                    dec_init_vec = (init_h, init_c)

            else:
                dec_init_vec = inputs.final_states[-1]
        return dec_init_vec

    def _add_positional_embeddings(self, utterances_hiddens, lens, turn_level, group=True):
        # grouped_states = []
        #
        # start_index = 0
        # for utterance in utterances:
        #     grouped_states.append(hidden_states[start_index:start_index + len(utterance)])
        #     start_index += len(utterance)
        # assert len(hidden_states) == sum([len(seq) for seq in grouped_states]) == sum([len(utterance) for utterance in utterances])

        new_states = []

        num_utterances_to_keep = min(self.args.turn_num, turn_level + 1)
        positional_sequence = []
        for i, l in enumerate(lens):
            index = num_utterances_to_keep - i - 1
            positional_sequence.append(
                self.positional_embedder(torch.tensor(index, device=self.device)).expand([l, -1]))
            # for state in states[0,:,:]:
            #     positional_sequence.append(torch.cat([state, self.positional_embedder(torch.tensor(index, device=self.device))], dim=0))

            # assert len(positional_sequence) == turn_level+1, \
            #     "Expected utterance and state sequence length to be the same, " \
            #     + "but they were " + str(turn_level+1) \
            #     + " and " + str(len(positional_sequence))
        positional_embedding = torch.cat(positional_sequence, 0).unsqueeze(0)
        states = torch.cat(utterances_hiddens, 1)
        new_states = torch.cat([states, positional_embedding], -1)

        return new_states

    def get_previous_schema_embedds(self, target_actions, table_embedding, col_embedding):
        rule = RuleGenerator()
        col_embeddings = []
        table_embeddings = []
        col_ids = []
        table_ids = []
        for idx, i in enumerate(target_actions[:, 1]):
            action_type = rule.determine_type_of_node(int(i))
            if action_type == TAB_TYPE:
                if target_actions[idx, 2].item() not in table_ids:
                    if target_actions[idx, 2].item() != -1:
                        table_ids.append(target_actions[idx, 2].item())
                        table_embeddings.append(table_embedding[:, target_actions[idx, 2].item(), :].squeeze())

            elif action_type == COL_TYPE:
                if target_actions[idx, 3].item() not in col_ids:
                    col_ids.append(target_actions[idx, 3].item())
                    col_embeddings.append(col_embedding[:, target_actions[idx, 3].item(), :].squeeze())
        if len(table_embeddings) > 0:
            table_embeddings = torch.stack(table_embeddings).unsqueeze(0)
        else:
            table_embeddings = None
        col_embeddings = torch.stack(col_embeddings).unsqueeze(0)
        return table_embeddings, col_embeddings, table_ids, col_ids

    def get_token_indices(self, token, index_to_token):
        """ Maps from a gold token (string) to a list of indices.

        Inputs:
            token (string): String to look up.
            index_to_token (list of tokens): Ordered list of tokens.

        Returns:
            list of int, representing the indices of the token in the probability
                distribution.
        """
        if token in index_to_token:
            if len(set(index_to_token)) == len(index_to_token):  # no duplicates
                return [index_to_token.index(token)]
            else:
                indices = []
                for index, other_token in enumerate(index_to_token):
                    if token == other_token:
                        indices.append(index)
                assert len(indices) == len(set(indices))
                return indices
        else:
            return None

    def get_schema_ids(self, target_actions):
        rule = RuleGenerator()
        col_ids = []
        tab_ids = []
        for idx, i in enumerate(target_actions[:, 1]):
            action_type = rule.determine_type_of_node(int(i))
            if action_type == TAB_TYPE:
                tab_ids.append(target_actions[idx, 2].item())
                # if target_actions[idx, 2].item() not in table_ids:
                #     if target_actions[idx, 2].item() != -1:
                #         table_ids.append(target_actions[idx, 2].item())
                #         table_embeddings.append(table_embedding[:, target_actions[idx, 2].item(), :].squeeze())
            else:
                tab_ids.append(-1)
        for idx, i in enumerate(target_actions[:, 1]):
            action_type = rule.determine_type_of_node(int(i))
            if action_type == COL_TYPE:
                col_ids.append(target_actions[idx, 3].item())
                # if target_actions[idx, 2].item() not in table_ids:
                #     if target_actions[idx, 2].item() != -1:
                #         table_ids.append(target_actions[idx, 2].item())
                #         table_embeddings.append(table_embedding[:, target_actions[idx, 2].item(), :].squeeze())
            else:
                col_ids.append(-1)

        col_ids = torch.tensor(col_ids, device=self.device)
        tab_ids = torch.tensor(tab_ids, device=self.device)
        return col_ids, tab_ids

    # def score_prev_query(self, previous_query_embedd, pre_target_actions, att_t):
    #     previous_query_embedds = torch.cat(previous_query_embedd, 0)
    #
    #     att_t_prev_query = self.prev_query_vec_to_action_embed(att_t)
    #     query_embed, _ = self.query_lstm_encode(previous_query_embedds.unsqueeze(0), previous_query_embedds.size()[0],
    #                                             self.prev_query_encoder_lstm)
    #     scores = torch.t(torch.mm(att_t_prev_query, torch.t(query_embed.squeeze())))
    #
    #     return scores, previous_query_embedds

    def score_prev_schema(self, schema_embedding, att_t, scorer):
        att_t_prev_query = scorer(att_t)
        scores = torch.t(torch.mm(att_t_prev_query, torch.t(schema_embedding.squeeze(0))))
        return scores

    # def get_copy_switch(self, att_t):
    #     copy_switch = torch.sigmoid(self.copy_witch_linear(att_t))
    #     return copy_switch

    def sketch_get_copy_switch(self, att_t):
        copy_switch = torch.sigmoid(self.sketch_copy_witch_linear(att_t))
        return copy_switch

    def generate_bert_input(self, input, model, max_len=None, append_seq=False):
        # output = []
        tmp_input = []
        for i in input:
            if len(i) > 1:
                tmp_input.append(' '.join(i))
            elif len(i) == 0:
                pass
            else:
                tmp_input.append(i[0])
        tmp_input = ' '.join(tmp_input)
        # tmp_input = ' [SEP] '.join(tmp_input)
        tmp_input = '[CLS] ' + tmp_input
        new_embedding = []
        # tmp = []
        # count = 0
        input_ids = self.bert_tokenizer.encode(tmp_input, add_special_tokens=False)
        embedding = model(torch.tensor([input_ids], device=self.device))[0]
        for i, inp in enumerate(input, 1):
            if len(inp) == 1:
                new_embedding.append(embedding[:, i, :])
                i += 1
            elif len(inp) > 1:
                new_embedding.append(torch.mean(embedding[:, i:i + len(inp), :], 1))
                i += len(inp)
            else:
                pass
        # tmp = []
        # for i, id in enumerate(input_ids):
        #     token = self.bert_tokenizer.decode([id])
        #     if token == '[CLS]' or token == '[SEP]':
        #         if len(tmp) > 0:
        #             t = tmp[0]
        #             # t = []
        #
        #             for j, o in enumerate(tmp):
        #                 if j == 0:
        #                     continue
        #                 t += o
        #                 # t.append(o)
        #
        #             new_embedding.append(t)
        #             count += 1
        #         tmp = []
        #         continue
        #
        #     tmp.append(embedding[:,i,:])
        sent_len = len(new_embedding)
        if max_len is not None:
            pad_embedding = torch.zeros([1, 768], device=self.device)
            if len(new_embedding) < max_len:
                new_embedding.extend([pad_embedding] * (max_len - len(new_embedding)))
        # if append_seq:
        #     new_embedding.append(embedding[:,-1,:])
        new_embedding = torch.stack(new_embedding, 1)
        return new_embedding, sent_len

    def process_schema_output(ids, embedding, tokenizer, max_length):
        new_embedding = []
        tmp = []
        count = 0
        for i, id in enumerate(ids):
            if id.item() == 0:
                new_embedding.append(embedding[i])

            token = tokenizer.decode([id.item()])
            if token == '[CLS]' or token == '[SEP]':
                if len(tmp) > 0:
                    t = tmp[0]
                    for j, o in enumerate(tmp):
                        if j == 0:
                            continue
                        t += o

                    new_embedding.append(t)
                    count += 1
                tmp = []
                continue

            tmp.append(embedding[i])
        pad_embedding = embedding[-1]
        if len(new_embedding) < max_length:
            new_embedding.extend([pad_embedding] * (max_length - len(new_embedding)))

        return new_embedding

    def eval_copy(self, batch, device):
        batch_loss = 0
        acc = 0
        for i in range(len(batch.interactions)):
            l, ac = self.interactions_layer(batch.interactions[i], batch.table_names[i], batch.table_mask[i],
                                            batch.table_sents[i], batch.col_set_mask[i],
                                            batch.col_table_dict[i], device)
            batch_loss += l
            acc += ac
        return batch_loss, acc



    # def lstm_decode(self, x, h_tm1, src_encodings, src_encodings_att_linear,src_encodings_att_linear2, decoder, attention_func, turn_level, src_token_mask=None,
    #          return_att_weight=False):
    #     # h_t: (batch_size, hidden_size)
    #     h_t, cell_t = decoder(x, h_tm1)
    #
    #
    #     ctx1, alpha_t = self.dot_prod_attention(h_t,src_encodings[-1], src_encodings_att_linear(src_encodings[-1]),
    #                                                  mask=src_token_mask)
    #
    #
    #     if turn_level > 0:
    #         if self.args.decoder_att_over_all:
    #             prev_embedd = torch.cat(src_encodings[:-1], 1)
    #         else:
    #             prev_embedd = src_encodings[-2]
    #         if self.args.use_level_att:
    #             ctx2, alpha_t = self.dot_prod_attention(ctx1, prev_embedd,
    #                                                      src_encodings_att_linear2(prev_embedd),
    #                                                      mask=src_token_mask)
    #         else:
    #             ctx2, alpha_t = self.dot_prod_attention(h_t, prev_embedd,
    #                                                     src_encodings_att_linear2(prev_embedd),
    #                                                     mask=src_token_mask)
    #
    #         ctx_t = [h_t, ctx1, ctx2]
    #     else:
    #         ctx_t = [h_t, ctx1, torch.zeros_like(ctx1)]
    #
    #
    #     att_t = F.tanh(attention_func(torch.cat(ctx_t, 1)))
    #
    #     att_t = self.dropout(att_t)
    #
    #     if return_att_weight:
    #         return (h_t, cell_t), att_t, alpha_t
    #     else:
    #         return (h_t, cell_t), att_t

    def lstm_decode2(self, x, h_tm1, src_encodings, src_encodings_att_linear_set, decoder,
                     attention_func_set, turn_level, src_token_mask=None,
                     return_att_weight=False):
        num_turn_to_keep = min(self.args.turn_num, turn_level + 1)
        h_t, cell_t = decoder[num_turn_to_keep - 1](x, h_tm1)
        # if turn_level

        his_turn_encodings = torch.cat(src_encodings[-num_turn_to_keep:], 1)

        ctx1, alpha_t = self.dot_prod_attention(h_t, his_turn_encodings,
                                                src_encodings_att_linear_set[num_turn_to_keep - 1](his_turn_encodings),
                                                mask=src_token_mask)

        ctx_t = [h_t, ctx1]
        att_t = F.tanh(attention_func_set[num_turn_to_keep - 1](torch.cat(ctx_t, 1)))

        att_t = self.dropout(att_t)

        if return_att_weight:
            return (h_t, cell_t), att_t, alpha_t
        else:
            return (h_t, cell_t), att_t

    def lstm_decode3(self, x, h_tm1, src_encodings, src_encodings_att_linear_set, decoder,
                     attention_func_set, turn_level, src_token_mask=None,
                     return_att_weight=False):
        num_turn_to_keep = min(self.args.turn_num, turn_level + 1)
        h_t, cell_t = decoder(x, h_tm1)
        # if turn_level

        his_turn_encodings = torch.cat(src_encodings[-num_turn_to_keep:], 1)
        ctx1, alpha_t = self.dot_prod_attention(h_t, his_turn_encodings,
                                                src_encodings_att_linear_set[num_turn_to_keep - 1](his_turn_encodings),
                                                mask=src_token_mask)

        ctx_t = [h_t, ctx1]
        att_t = F.tanh(attention_func_set[num_turn_to_keep - 1](torch.cat(ctx_t, 1)))

        att_t = self.dropout(att_t)

        if return_att_weight:
            return (h_t, cell_t), att_t, alpha_t
        else:
            return (h_t, cell_t), att_t

    def lstm_decoder(self, x, h_tm1, src_encodings, schema_encodings, query_encodings, decoder,
                     attention_func, turn_level, src_token_mask=None,
                     return_att_weight=False):
        h_t, cell_t = decoder(x, h_tm1)
        schema_states = schema_encodings.squeeze().split(1, 0)
        schema_states = [s.squeeze() for s in schema_states]
        schema_attention_rs = self.decoder_schema_attention(h_t.squeeze(), schema_states)[2]

        src_states = src_encodings.squeeze().split(1, 0)
        src_states = [s.squeeze() for s in src_states]
        src_attention_rs = self.decoder_src_attention(h_t.squeeze(), src_states)[2]
        if turn_level > 0:
            query_encodings = torch.cat(query_encodings, 0)
            query_states = query_encodings.squeeze().split(1, 0)
            query_states = [s.squeeze() for s in query_states]
            query_attention_rs = self.decoder_query_attention(h_t.squeeze(), query_states)[2]
        else:
            query_attention_rs = self.start_query_attention_vector
        att_t = torch.cat([h_t.squeeze(), src_attention_rs], 0)
        if self.args.decoder_use_schema:
            att_t = torch.cat([att_t, schema_attention_rs], 0)
        if self.args.decoder_use_query:
            att_t = torch.cat([att_t, query_attention_rs], 0)

        att_t = F.tanh(attention_func(att_t.unsqueeze(0)))
        att_t = self.dropout(att_t)

        return (h_t, cell_t), att_t

    def dot_prod_attention(self, h_t, src_encoding, src_encoding_att_linear, mask=None):
        """
        :param h_t: (batch_size, hidden_size)
        :param src_encoding: (batch_size, src_sent_len, hidden_size * 2)
        :param src_encoding_att_linear: (batch_size, src_sent_len, hidden_size)
        :param mask: (batch_size, src_sent_len)
        """
        # (batch_size, src_sent_len)
        att_weight = torch.bmm(src_encoding_att_linear, h_t.unsqueeze(2)).squeeze(2)
        if mask is not None:
            att_weight.data.masked_fill_(mask.to(torch.uint8), -float('inf'))
        att_weight = F.softmax(att_weight, dim=-1)

        att_view = (att_weight.size(0), 1, att_weight.size(1))
        # (batch_size, hidden_size)
        ctx_vec = torch.bmm(att_weight.view(*att_view), src_encoding).squeeze(1)

        return ctx_vec, att_weight

    def init_decoder_state(self, enc_last_cell):
        h_0 = self.decoder_cell_init(enc_last_cell)
        h_0 = F.tanh(h_0)

        return h_0, torch.zeros_like(h_0)

    def query_lstm_encode(self, src_token_embed, src_sents_len, encoder, q_onehot_project=None):
        """
        encode the source sequence
        :return:
            src_encodings: Variable(batch_size, src_sent_len, hidden_size * 2)
            last_state, last_cell: Variable(batch_size, hidden_size)
        """
        # src_token_embed = self.gen_x_batch(src_sents_var)
        #
        # if q_onehot_project is not None:
        #     src_token_embed = torch.cat([src_token_embed, q_onehot_project], dim=-1)

        # All RNN modules accept packed sequences as inputs.
        src_len = torch.tensor([src_sents_len])
        packed_src_token_embed = pack_padded_sequence(src_token_embed, src_len, batch_first=True)
        # src_encodings: (tgt_query_len, batch_size, hidden_size)
        src_encodings, (last_state, last_cell) = encoder(packed_src_token_embed)
        src_encodings, _ = pad_packed_sequence(src_encodings, batch_first=True)
        # src_encodings: (batch_size, tgt_query_len, hidden_size)
        # src_encodings = src_encodings.permute(1, 0, 2)
        # (batch_size, hidden_size * 2)
        last_state = torch.cat([last_state[0], last_state[1]], -1)
        last_cell = torch.cat([last_cell[0], last_cell[1]], -1)

        return src_encodings, (last_state, last_cell)

    def lstm_encode(self, src_token_embed, src_sents_len, encoder):
        """
        encode the source sequence
        :return:
            src_encodings: Variable(batch_size, src_sent_len, hidden_size * 2)
            last_state, last_cell: Variable(batch_size, hidden_size)
        """
        # src_token_embed = self.gen_x_batch(src_sents_var)
        #
        # if q_onehot_project is not None:
        #     src_token_embed = torch.cat([src_token_embed, q_onehot_project], dim=-1)

        # All RNN modules accept packed sequences as inputs.
        src_len = torch.tensor([src_sents_len])
        packed_src_token_embed = pack_padded_sequence(src_token_embed, src_len, batch_first=True)
        # src_encodings: (tgt_query_len, batch_size, hidden_size)
        src_encodings, (last_state, last_cell) = encoder(packed_src_token_embed)
        src_encodings, _ = pad_packed_sequence(src_encodings, batch_first=True)
        # src_encodings: (batch_size, tgt_query_len, hidden_size)
        # src_encodings = src_encodings.permute(1, 0, 2)
        # (batch_size, hidden_size * 2)
        last_state = torch.cat([last_state[0], last_state[1]], -1)
        last_cell = torch.cat([last_cell[0], last_cell[1]], -1)

        return src_encodings, (last_state, last_cell)

    def schema_lstm_encode(self, src_token_embed, src_sents_len, q_onehot_project=None):
        """
        encode the source sequence
        :return:
            src_encodings: Variable(batch_size, src_sent_len, hidden_size * 2)
            last_state, last_cell: Variable(batch_size, hidden_size)
        """
        # src_token_embed = self.gen_x_batch(src_sents_var)
        #
        # if q_onehot_project is not None:
        #     src_token_embed = torch.cat([src_token_embed, q_onehot_project], dim=-1)

        # All RNN modules accept packed sequences as inputs.
        src_len = torch.tensor([src_sents_len])
        packed_src_token_embed = pack_padded_sequence(src_token_embed, src_len, batch_first=True)
        # src_encodings: (tgt_query_len, batch_size, hidden_size)
        src_encodings, (last_state, last_cell) = self.schema_encoder_lstm(packed_src_token_embed)
        src_encodings, _ = pad_packed_sequence(src_encodings, batch_first=True)
        # src_encodings: (batch_size, tgt_query_len, hidden_size)
        # src_encodings = src_encodings.permute(1, 0, 2)
        # (batch_size, hidden_size * 2)
        last_state = torch.cat([last_state[0], last_state[1]], -1)
        last_cell = torch.cat([last_cell[0], last_cell[1]], -1)

        return src_encodings, (last_state, last_cell)

    def generate_mask(self, length, max_length, device):
        if length <= max_length:
            mask = [1] * length
            padding = [0] * (max_length - length)
            mask.extend(padding)
        else:
            raise Exception('token length exceed max length')
        mask = torch.tensor(mask, dtype=torch.long, device=device)
        return mask

    def col_lstm_encode(self, src_token_embed, src_sents_len, q_onehot_project=None):
        """
        encode the source sequence
        :return:
            src_encodings: Variable(batch_size, src_sent_len, hidden_size * 2)
            last_state, last_cell: Variable(batch_size, hidden_size)
        """
        # src_token_embed = self.gen_x_batch(src_sents_var)
        #
        # if q_onehot_project is not None:
        #     src_token_embed = torch.cat([src_token_embed, q_onehot_project], dim=-1)

        # All RNN modules accept packed sequences as inputs.
        src_len = torch.tensor([src_sents_len])
        packed_src_token_embed = pack_padded_sequence(src_token_embed, src_len, batch_first=True)
        # src_encodings: (tgt_query_len, batch_size, hidden_size)
        src_encodings, (last_state, last_cell) = self.col_encoder_lstm(packed_src_token_embed)
        src_encodings, _ = pad_packed_sequence(src_encodings, batch_first=True)
        # src_encodings: (batch_size, tgt_query_len, hidden_size)
        # src_encodings = src_encodings.permute(1, 0, 2)
        # (batch_size, hidden_size * 2)
        last_state = torch.cat([last_state[0], last_state[1]], -1)
        last_cell = torch.cat([last_cell[0], last_cell[1]], -1)

        return src_encodings, (last_state, last_cell)

    def eval_copy_layer(self, src_embedding, src_mask, copy_ast_arg, device):
        # src_shape = src_tensor.size()
        loss = 0
        acc = 0
        max_src_seq_len = self.args.max_src_seq_length

        src_seq_mask = self._generate_square_subsequent_mask(max_src_seq_len)
        # 0 for unmasked token, 1 for masked token
        src_padding_mask = self._generate_padding_mask(src_mask)

        # not batch first
        encoder_outputs = self.encoder(src_embedding.transpose(0, 1), mask=src_seq_mask,
                                       src_key_padding_mask=src_padding_mask.unsqueeze(0))

        # copy_layer
        is_copy = True if copy_ast_arg is not None else False
        if is_copy:
            copy_label = torch.tensor([1], device=self.device)
        else:
            copy_label = torch.tensor([0], device=self.device)
        copy_loss_fct = nn.CrossEntropyLoss()
        copy_logits = self.copy_out(encoder_outputs[-1, :, :])
        copy_prob = F.softmax(copy_logits, dim=1)
        pred = torch.argmax(copy_prob, dim=1)
        if pred.item() == copy_label.item():
            acc += 1
        copy_loss = copy_loss_fct(copy_prob, copy_label)
        return copy_loss, acc

    def embedding_padding(self, encoder_input):
        zero_embedding = torch.zeros([1, 1, self.args.embed_size], dtype=torch.float, device=self.device)
        src_len = encoder_input.size()[1]
        new_encoder_input = [encoder_input]
        if encoder_input.size()[1] < self.args.max_src_seq_length:
            new_encoder_input.extend([zero_embedding] * (self.args.max_src_seq_length - src_len))
            new_encoder_input = torch.cat(new_encoder_input, dim=1)
        return new_encoder_input

    def encode_ast_by_alignment(self, column_input, table_input, col_align, table_align, col_embedding=None,
                                table_embedding=None):
        col = [column_input[a] for a in col_align]
        tab = [table_input[a] for a in table_align]
        col.extend(tab)
        # print(col)

        # with torch.no_grad():
        col_embed = [col_embedding[:, a, :] for a in col_align]
        tab_embed = [table_embedding[:, a, :] for a in table_align]
        col_embed.extend(tab_embed)
        if len(col_embed) == 0:
            alignment_embedd = torch.zeros([1, 1, self.args.hidden_size])
        else:
            alignment_embedd = torch.stack(col_embed, 1)
            # alignment_embedd = torch.cat([col_embed, tab_embed], 1)
            # alignment_embedd,_ = self.generate_bert_input(col, self.bert_encoder, max_len=self.args.max_align_length, append_seq=False)
            # if self.args.use_schema_lstm:
            #     alignment_embedd,_ = self.col_lstm_encode(alignment_embedd, len(col))
        # else:
        #     alignment_embedd = self.generate_src_embedding([col], max_length=self.args.max_align_length, args=self.args)
        return torch.mean(alignment_embedd[:, :len(col), :], dim=1).squeeze()

    def copy_layer(self, copy_ast_arg, encoder_outputs):
        # copy layer
        is_copy = True if copy_ast_arg is not None else False
        if is_copy:
            copy_label = torch.tensor([1], device=self.device)
        else:
            copy_label = torch.tensor([0], device=self.device)
        copy_loss_fct = nn.CrossEntropyLoss()
        m = nn.LeakyReLU(0.1)
        copy_logits = self.copy_out(self.act(encoder_outputs))
        # copy_prob = F.softmax(copy_logits, dim=1)
        copy_loss = copy_loss_fct(copy_logits, copy_label)
        return copy_loss

    def group_by_layer(self, is_group_by, last_cell):
        if is_group_by:
            copy_label = torch.tensor([1], device=self.device)
        else:
            copy_label = torch.tensor([0], device=self.device)
        copy_loss_fct = nn.CrossEntropyLoss()
        m = nn.LeakyReLU(0.1)
        copy_logits = self.group_output(self.act(last_cell))
        # copy_prob = F.softmax(copy_logits, dim=1)
        copy_loss = copy_loss_fct(copy_logits, copy_label)
        return copy_loss

    def generate_ast_embedding(self, turn_level, interactions, column_input, table_input, col_embedding=None,
                               table_embedding=None):

        inter_ast = []
        inter_ast_embedding = []
        for turn_level, utter in enumerate(interactions[:turn_level]):
            # generate scr embedding
            utter_ast = []
            utter_ast_embedding = []
            for a in utter.processed_ast:
                (ast, actions, column_alignment, table_alignment) = a
                # [embed_size]
                alignment_embedd = self.encode_ast_by_alignment(column_input, table_input, column_alignment,
                                                                table_alignment, col_embedding, table_embedding)
                utter_ast.append(ast)
                utter_ast_embedding.append(alignment_embedd)
            inter_ast.append(utter_ast)
            inter_ast_embedding.append(utter_ast_embedding)
        return inter_ast, inter_ast_embedding

    def generate_snippet_embedding(self, turn_level, interactions, column_input, table_input, col_embedding=None,
                                   table_embedding=None):

        inter_ast = []
        inter_ast_embedding = []
        for turn_level, utter in enumerate(interactions[:turn_level]):
            # generate scr embedding
            utter_ast = []
            utter_ast_embedding = []
            for a in utter.processed_ast:
                (ast, actions, column_alignment, table_alignment) = a
                # [embed_size]
                target_actions = torch.from_numpy(actions[0])
                target_actions = target_actions.to(self.device)
                target_action_embeddings = self.rule_embedding(target_actions[:, 1]).squeeze(1)
                rule = RuleGenerator()
                for idx, i in enumerate(target_actions[:, 1]):
                    action_type = rule.determine_type_of_node(int(i))
                    if action_type == TAB_TYPE:
                        target_action_embeddings[idx] = self.table_rnn_input(
                            table_embedding[:, target_actions[idx, 2].item(), :].squeeze())
                    elif action_type == COL_TYPE:
                        target_action_embeddings[idx] = self.column_rnn_input(
                            col_embedding[:, target_actions[idx, 3].item(), :].squeeze())

                query_embed, (last_state, last_cell) = self.query_lstm_encode(target_action_embeddings.unsqueeze(0),
                                                                              target_action_embeddings.size()[0],
                                                                              self.query_encoder_lstm)
                # print(query_embed.size())
                averaged_query_embed = torch.mean(query_embed.squeeze(0), 0)
                # print(averaged_query_embed.size())
                # alignment_embedd = self.encode_ast_by_alignment(column_input, table_input, column_alignment,
                #                                                 table_alignment, col_embedding, table_embedding)
                utter_ast.append(ast)
                utter_ast_embedding.append(averaged_query_embed)
            inter_ast.append(utter_ast)
            inter_ast_embedding.append(utter_ast_embedding)
        return inter_ast, inter_ast_embedding

    def pred_generate_snippet_embedding(self, turn_level, processed_trees, column_input, table_input,
                                        col_embedding=None, table_embedding=None):

        inter_ast = []
        inter_ast_embedding = []
        for turn_level, t in enumerate(processed_trees[:turn_level]):
            # generate scr embedding
            utter_ast = []
            utter_ast_embedding = []
            for a in t:
                (ast, actions, column_alignment, table_alignment) = a
                # [embed_size]
                target_actions = torch.from_numpy(actions[0])
                target_actions = target_actions.to(self.device)
                target_action_embeddings = self.rule_embedding(target_actions[:, 1]).squeeze(1)
                rule = RuleGenerator()
                for idx, i in enumerate(target_actions[:, 1]):
                    action_type = rule.determine_type_of_node(int(i))
                    if action_type == TAB_TYPE:
                        target_action_embeddings[idx] = self.table_rnn_input(
                            table_embedding[:, target_actions[idx, 2].item(), :].squeeze())
                    elif action_type == COL_TYPE:
                        target_action_embeddings[idx] = self.column_rnn_input(
                            col_embedding[:, target_actions[idx, 3].item(), :].squeeze())
                query_embed, (last_state, last_cell) = self.query_lstm_encode(target_action_embeddings.unsqueeze(0),
                                                                              target_action_embeddings.size()[0],
                                                                              self.query_encoder_lstm)
                # print(query_embed.size())
                averaged_query_embed = torch.mean(query_embed.squeeze(0), 0)
                # print(averaged_query_embed.size())
                # alignment_embedd = self.encode_ast_by_alignment(column_input, table_input, column_alignment,
                #                                                 table_alignment, col_embedding, table_embedding)
                utter_ast.append(ast)
                utter_ast_embedding.append(averaged_query_embed)
            inter_ast.append(utter_ast)
            inter_ast_embedding.append(utter_ast_embedding)
        return inter_ast, inter_ast_embedding

    def predict_generate_ast_embedding(self, turn_level, processed_trees, column_input, table_input, col_embedding=None,
                                       table_embedding=None):

        inter_ast = []
        inter_ast_embedding = []
        if turn_level > len(processed_trees):
            end = len(processed_trees)
        else:
            end = turn_level
        for turn_level, t in enumerate(processed_trees[:end]):
            # generate scr embedding
            utter_ast = []
            utter_ast_embedding = []
            for a in t:
                (ast, actions, column_alignment, table_alignment) = a
                # [embed_size]
                alignment_embedd = self.encode_ast_by_alignment(column_input, table_input, column_alignment,
                                                                table_alignment, col_embedding, table_embedding)
                utter_ast.append(ast)
                utter_ast_embedding.append(alignment_embedd)
            inter_ast.append(utter_ast)
            inter_ast_embedding.append(utter_ast_embedding)
        return inter_ast, inter_ast_embedding

    def sep_utterance_embedding(self, utter_embeddings, sep_embedding=None):
        """
        add seperate token ';' and add turn_level position embedding to every token
        :param utter_embeddings:
        :param turn_level:
        :return:
        """
        if sep_embedding is None:
            sep_embedding = self.word_emb.get(';', np.zeros(self.args.col_embed_size, dtype=np.float32))

            sep_embedding = torch.from_numpy(sep_embedding)
            sep_embedding = sep_embedding.float()
            sep_embedding = sep_embedding.to(self.device)
        sep_embedding = sep_embedding.view(1, 1, -1)
        # combine all previous turn and current turn

        new_utter_embedding = []
        for idx, e in enumerate(utter_embeddings):
            # modify turn
            if self.args.turn_num == 1 or self.args.turn_num == 2:
                # turn_pos_encoding = self.turn_pos_encoder.pe[:, idx, :]
                # turn_pos_encoding = turn_pos_encoding.unsqueeze(1)
                new_e = torch.cat([e, sep_embedding], dim=1)
                # new_e = torch.add(new_e, turn_pos_encoding)
                new_utter_embedding.append(new_e)
            else:
                turn_pos_encoding = self.turn_pos_encoder.pe[:, idx, :]
                turn_pos_encoding = turn_pos_encoding.unsqueeze(1)
                new_e = torch.cat([e, sep_embedding], dim=1)
                new_e = torch.add(new_e, turn_pos_encoding)
                new_utter_embedding.append(new_e)
        sep_utter_embeddings = torch.cat(new_utter_embedding, dim=1)
        return sep_utter_embeddings

    def sep_utterance_mask(self, masks):
        sep_mask = torch.ones(1, 1)
        new_mask = []
        for idx, m in enumerate(masks):
            new_m = torch.cat([m, sep_mask], dim=1)
            new_mask.append(new_m)
        new_sep_mask = torch.cat(new_mask, dim=1)
        return new_sep_mask

    def _generate_square_subsequent_mask(self, sz):
        device = torch.device('cuda', self.args.cuda_device_num) if self.args.cuda else torch.device('cpu')
        # print(device)
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.to(device)
        # print(mask.device)
        return mask

    def _generate_padding_mask(self, mask):
        device = torch.device('cuda', self.args.cuda_device_num) if self.args.cuda else torch.device('cpu')
        mask = mask.eq(0)
        mask = mask.to(device)
        # print('mask:{}'.format(mask.device))
        return mask

    def generate_src_embedding(self, q, max_length, args):
        B = len(q)
        val_embs = []
        val_len = np.zeros(B, dtype=np.int64)
        is_list = False
        if type(q[0][0]) == list:
            is_list = True
        for i, one_q in enumerate(q):
            if not is_list:
                q_val = list(
                    map(lambda x: self.word_emb.get(x, np.zeros(self.args.col_embed_size, dtype=np.float32)), one_q))
            else:
                q_val = []
                for ws in one_q:
                    emb_list = []
                    ws_len = len(ws)
                    for w in ws:
                        emb_list.append(self.word_emb.get(w, self.word_emb['unk']))
                    if ws_len == 0:
                        raise Exception("word list should not be empty!")
                    elif ws_len == 1:
                        q_val.append(emb_list[0])
                    else:
                        q_val.append(sum(emb_list) / float(ws_len))

            val_embs.append(q_val)
            val_len[i] = len(q_val)

        # max_len = max(val_len)

        val_emb_array = np.zeros((B, max_length, self.args.col_embed_size), dtype=np.float32)
        for i in range(B):
            for t in range(len(val_embs[i])):
                val_emb_array[i, t, :] = val_embs[i][t]
        val_inp = torch.from_numpy(val_emb_array)
        if self.args.cuda:
            val_inp = val_inp.to(torch.device('cuda', args.cuda_device_num))
        return val_inp

    def embedding_cosine(self, src_embedding, table_embedding, table_unk_mask):
        embedding_differ = []
        for i in range(table_embedding.size(1)):
            one_table_embedding = table_embedding[:, i, :]
            one_table_embedding = one_table_embedding.unsqueeze(1).expand(table_embedding.size(0),
                                                                          src_embedding.size(1),
                                                                          table_embedding.size(2))

            topk_val = F.cosine_similarity(one_table_embedding, src_embedding, dim=-1)

            embedding_differ.append(topk_val)
        embedding_differ = torch.stack(embedding_differ).transpose(1, 0)
        embedding_differ.data.masked_fill_(table_unk_mask.unsqueeze(2).expand(
            table_embedding.size(0),
            table_embedding.size(1),
            embedding_differ.size(2)
        ).to(torch.uint8), 0)

        return embedding_differ

    def input_type(self, values_list, max_len):
        B = len(values_list)
        val_len = []
        for value in values_list:
            val_len.append(len(value))
        # max_len = max(val_len)
        # for the Begin and End
        val_emb_array = np.zeros((B, max_len, values_list[0].shape[1]), dtype=np.float32)
        for i in range(B):
            val_emb_array[i, :val_len[i], :] = values_list[i][:, :]

        val_inp = torch.from_numpy(val_emb_array)
        if self.args.cuda:
            val_inp = val_inp.cuda(self.args.cuda_device_num)

        return val_inp

    def get_utter_input(self, interactions, max_turn, turn_level):
        num_utterances_to_keep = min(max_turn, turn_level + 1)
        tar = interactions[:turn_level + 1][-num_utterances_to_keep:]
        src_sent = []
        src_sent_lens = []
        for u in tar:
            src_sent.extend(u.src_sent)
            src_sent.append([';'])
            src_sent_lens.append(len(u.src_sent) + 1)
        return src_sent, src_sent_lens

    def utter_self_attention(self, utter_hs, turn_level):
        utter_hidden_states = torch.cat(utter_hs, 1).transpose(0, 1)
        utter_hidden_state_self_att = self.utter_muiltiheadatt(utter_hidden_states, utter_hidden_states,
                                                               utter_hidden_states)[0]
        utter_hidden_state_self_att = utter_hidden_state_self_att.transpose(0, 1)
        utter_lens = [i.size()[1] for i in utter_hs]

        utter_hs_ls = [utter_hidden_state_self_att[:, :l, :] for l in utter_lens]

        return utter_hs_ls


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class BeamSearchNode(object):
    def __init__(self, decoder_input, pre_node, parent_node, curnode_type, logProb, length, ast, col_mask,
                 table_enable, curnode_id, pre_action_embedding, cur_node, h_tm1, filter_flag, col_appear_mask=None,
                 table_id=None, column_id=None):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.decoder_input = decoder_input
        self.pre_node = pre_node
        self.parent_node = parent_node
        self.curnode_type = curnode_type
        self.table_id = table_id
        self.column_id = column_id
        self.ast = ast
        self.col_mask = col_mask
        self.curnode_id = curnode_id
        self.pre_action_embedding = pre_action_embedding

        self.logp = logProb
        self.leng = length
        # self.cur_node = list(self.ast.subtrees())[curnode_id]
        self.cur_node = cur_node
        self.table_enable = table_enable
        self.h_tm1 = h_tm1
        self.col_appear_mask = col_appear_mask
        self.filter_flag = filter_flag

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward

    def __eq__(self, other):
        return self.logp == other.logp

    def __lt__(self, other):
        return self.logp < other.logp


def deep_copy_ast(source):
    copy_ast = copy.deepcopy(source)
    copy_ast_subtrees = list(copy_ast.subtrees())
    for i, a in enumerate(list(source.subtrees())):
        # if a.parent() is None:
        #     copy_ast_subtrees[i]._parent = None
        # else:
        #     copy_ast_subtrees[i]._parent = copy.deepcopy(a.parent())
        #     # print('asfd', a.parent())
        #     copy_ast_subtrees[i]._parent.__dict__ = copy.deepcopy(a.parent().__dict__)
        for k, v in a.__dict__.items():
            if k != '_parent':
                copy_ast_subtrees[i].__dict__[k] = copy.deepcopy(a.__dict__[k])

    return copy_ast


def deep_copy_ast_b(source, cur_node):
    copy_ast = copy.deepcopy(source)

    copy_ast_subtrees = list(copy_ast.subtrees())
    for i, a in enumerate(list(source.subtrees())):
        # if a.parent() is None:
        #     copy_ast_subtrees[i]._parent = None
        # else:
        #     copy_ast_subtrees[i]._parent = copy.deepcopy(a.parent())
        #     # print('asfd', a.parent())
        #     copy_ast_subtrees[i]._parent.__dict__ = copy.deepcopy(a.parent().__dict__)
        for k, v in a.__dict__.items():
            if k != '_parent':
                copy_ast_subtrees[i].__dict__[k] = copy.deepcopy(a.__dict__[k])

                p1 = a
                p2 = copy_ast_subtrees[i]
                while p1.parent() is not None:

                    if p2._parent is None:
                        # p2._parent = copy.deepcopy(cur_node)
                        # print(p2._parent)
                        # p2._parent.__dict__ = cur_node.__dict__
                        break
                    p2._parent.__dict__[k] = copy.deepcopy(p1.parent().__dict__[k])
                    p1 = p1.parent()
                    p2 = p2._parent

    return copy_ast


def get_node_type(node):
    if node.label() == 'C':
        node_type = COL_TYPE
    elif node.label() == 'T':
        node_type = TAB_TYPE
    elif node.label() == 'Y':
        node_type = VAL_TYPE
    else:
        node_type = RULE_TYPE
    return node_type

# def hobbs_algorithm(sents):
#     p = ["He", "he", "Him", "him", "She", "she", "Her",
#          "her", "It", "it", "They", "they"]
#     r = ["Himself", "himself", "Herself", "herself",
#          "Itself", "itself", "Themselves", "themselves"]
#
#     parser = StanfordParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
#     res = list(parser.parse(sents))
#     trees = [Tree.fromstring(s) for s in res]
#     for pro in p:
#         pos = get_pos(trees[-1], pro)
#         pos = pos[:-1]
#         tree, pos = hobbs(trees, pos)
#         for t in trees:
#             print(t, '\n')
#         print("Proposed antecedent for '" + pro + "':", tree[pos])
#     for pro in r:
#         pos = get_pos(trees[-1], pro)
#         pos = pos[:-1]
#         tree, pos = resolve_reflexive(trees, pos)
#         for t in trees:
#             print(t, '\n')
#         print("Proposed antecedent for '" + pro + "':", tree[pos])
