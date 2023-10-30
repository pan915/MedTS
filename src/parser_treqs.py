import os
import json
import copy
import pickle
from num2words import num2words
from pyparsing import basestring
from tqdm import tqdm
import pandas as pd
from nltk import ParentedTree
from tree import TreeWithPara
from nltk.grammar import Production, Nonterminal
from functools import lru_cache
from RuleGenerator import RuleGenerator
from shared_variables import COL_TYPE, TAB_TYPE, RULE_TYPE, PAD_id, SOS_id
from graph import Graph
from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()

class Rule(object):
    def __init__(self, rule, pre=None, parent=None, length=None, frontier=None, rule_id=None, rule_type=None, data=None):
        self.rule = rule
        self.parent = parent
        self.pre = pre
        self.frontier = frontier
        self.length = length
        self.rule_id = rule_id


class FromSQLParser(object):

    CLAUSE_KEYWORDS = ('select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except')
    JOIN_KEYWORDS = ('join', 'on', 'as')

    WHERE_OPS = ('not', 'between', 'e', 'mt', 'lt', 'mte', 'lte', 'ne', 'in', 'like', 'is', 'exists')
    UNIT_OPS = ('none', 'sub', 'add', "mul", 'div')
    AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')
    TABLE_TYPE = {
        'sql': "sql",
        'table_unit': "table_unit",
    }
    COND_OPS = ('and', 'or')
    SQL_OPS = ('intersect', 'union', 'except')
    ORDER_OPS = ('desc', 'asc')

    def __init__(self, sql, utterance_arg, interaction, turn_level):
        self.select = sql['select']
        self.is_distinct = self.select[0]
        self.order_by = sql['orderBy']
        self.where = sql['where']
        self.group_by = sql['groupBy']
        self.having = sql['having']
        self.limit = sql['limit']
        self.intersect = sql['intersect']
        self.except_clause = sql['except']
        self.union = sql['union']
        # self.table_unit = sql['from']['table_units']
        self.utterance_arg = utterance_arg
        self.interaction = interaction
        self.turn_level = turn_level

    def parse_sql(self, root, rules: RuleGenerator, column_names, table_ids, table_names,
                  col_set, rule_seq, rule_stack, is_root, parent=None, pre=None):
        """

        :param root:
        :param rules:
        :param column_names:
        :param table_ids:
        :param table_names:
        :param rule_seq:
        :param rule_stack:
        :param is_root:
        :param parent:
        :param pre:
        :return:
        """

        if is_root:
            if self.except_clause is not None:
                rule = rules.get_rule_by_rhs_lhs(rhs='except', lhs='Z')
                r = Rule(rule=rule[0], parent=root, pre=root, length=3, frontier=1)
            elif self.union is not None:
                rule = rules.get_rule_by_rhs_lhs(rhs='union', lhs='Z')
                r = Rule(rule=rule[0], parent=root, pre=root, length=3, frontier=1)
            elif self.intersect is not None:
                rule = rules.get_rule_by_rhs_lhs(rhs='intersect', lhs='Z')
                r = Rule(rule=rule[0], parent=root, pre=root, length=3, frontier=1)
            else:
                rule = Production(lhs=Nonterminal("Z"), rhs=[Nonterminal("R")])
                r = Rule(rule=rule, parent=root, length=1, pre=root, frontier=0)

            rule_seq.append(r)
            rule_stack.append(r)

        # parse sql
        has_order_by = False if len(self.order_by) == 0 else True
        has_filter = False if len(self.where) + len(self.having) == 0 else True

        # sql query is distinct and has order by and filter
        if self.is_distinct and has_order_by and has_filter:
            rule = rules.get_rule_by_rhs_lhs(rhs='distinct Select Filter Order', lhs='R')
            length = 4
            frontier_index = 1
        # sql query is distinct and has no order by and filter
        elif self.is_distinct and not has_order_by and has_filter:
            rule = rules.get_rule_by_rhs_lhs(rhs='distinct Select Filter', lhs='R')
            length = 3
            frontier_index = 1
        # sql query is distinct and has order by and no filter
        elif self.is_distinct and has_order_by and not has_filter:
            rule = rules.get_rule_by_rhs_lhs(rhs='distinct Select Order', lhs='R')
            length = 3
            frontier_index = 1
        elif self.is_distinct and not has_order_by and not has_filter:
            rule = rules.get_rule_by_rhs_lhs(rhs='distinct Select', lhs='R')
            length = 2
            frontier_index = 1

        elif not self.is_distinct and has_order_by and has_filter:
            rule = rules.get_rule_by_rhs_lhs(rhs='Select Filter Order', lhs='R')
            length = 3
            frontier_index = 0
        # sql query is distinct and has no order by and filter
        elif not self.is_distinct and not has_order_by and has_filter:
            rule = rules.get_rule_by_rhs_lhs(rhs='Select Filter', lhs='R')
            length = 2
            frontier_index = 0
        # sql query is distinct and has order by and no filter
        elif not self.is_distinct and has_order_by and not has_filter:
            rule = rules.get_rule_by_rhs_lhs(rhs='Select Order', lhs='R')
            length = 2
            frontier_index = 0
        elif not self.is_distinct and not has_order_by and not has_filter:
            rule = rules.get_rule_by_rhs_lhs(rhs='Select', lhs='R')
            length = 1
            frontier_index = 0
        else:
            rule = None
            length = 1
            frontier_index = 0

        if rule is None:
            raise Exception('Parsing error: Sql contains invalid syntax')

        if is_root:
            r = Rule(rule=rule[0], parent=rule_stack[-1], pre=rule_seq[-1], length=length, frontier=frontier_index)
            if rule_stack[-1].frontier == rule_stack[-1].length - 1:
                rule_stack.pop()
        else:
            r = Rule(rule=rule[0], parent=parent, pre=pre, length=length, frontier=frontier_index)
        # if rule_stack[-1].frontier == rule_stack[-1].length - 1:
        #     rule_seq.pop()
        rule_seq.append(r)
        rule_stack.append(r)

        # expand select
        if len(self.select[1]) <= 5:
            rule = Production(lhs=Nonterminal('Select'), rhs=[Nonterminal('A')]*len(self.select[1]))
            r = Rule(rule=rule, parent=rule_stack[-1], pre=rule_seq[-1], length=len(self.select[1]), frontier=0)
        else:
            rule = Production(lhs=Nonterminal('Select'), rhs=[Nonterminal('A')] * 5)
            r = Rule(rule=rule, parent=rule_stack[-1], pre=rule_seq[-1], length=5, frontier=0, data=self.select)
        if rule_stack[-1].frontier == rule_stack[-1].length - 1:
            rule_stack.pop()
        rule_seq.append(r)
        rule_stack.append(r)
        # print(self.select)
        rule_seq, rule_stack = self.expand_a(select_unit=self.select[1][0],
                                             rule_seq=rule_seq,
                                             rule_stack=rule_stack,
                                             rules=rules,
                                             column_names=column_names,
                                             table_ids=table_ids,
                                             table_names=table_names,
                                             col_set=col_set)

        while len(rule_stack) > 0:
            while rule_stack[-1].frontier == rule_stack[-1].length - 1:
                rule_stack.pop()
                if len(rule_stack) == 0:
                    break
            if len(rule_stack) == 0:
                break
            rule = rule_stack[-1].rule
            cur_node = rule.rhs()[rule_stack[-1].frontier + 1]
            rule_stack[-1].frontier += 1
            # print(rule)
            if cur_node == Nonterminal('A'):
                rule_seq, rule_stack = self.expand_a(select_unit=self.select[1][rule_stack[-1].frontier],
                                                     rule_seq=rule_seq,
                                                     rule_stack=rule_stack,
                                                     rules=rules,
                                                     column_names=column_names,
                                                     table_ids=table_ids,
                                                     table_names=table_names,
                                                     col_set=col_set)

                # rule_stack[-1].frontier += 1
                continue
            if cur_node == Nonterminal('Filter'):

                rule_seq, rule_stack = self.expand_filter(where=self.where,
                                                          rule_seq=rule_seq,
                                                          rule_stack=rule_stack,
                                                          rules=rules,
                                                          table_names=table_names,
                                                          table_ids=table_ids,
                                                          column_names=column_names,
                                                          col_set=col_set)
                # rule_stack[-1].frontier += 1
                continue

            if cur_node == Nonterminal("R"):
                sub_sql = None
                if self.union is not None:
                    sub_sql = self.union
                elif self.except_clause is not None:
                    sub_sql = self.except_clause
                elif self.intersect is not None:
                    sub_sql = self.intersect

                parser = FromSQLParser(sub_sql, self.utterance_arg, self.interaction, self.turn_level)
                rule_seq_t = []
                rule_stack_t = []
                parent_t = rule_stack[-1]
                pre_t = rule_seq[-1]
                rule_seq_t, rule_stack_t = parser.parse_sql(root=cur_node,
                                                            rules=rules,
                                                            column_names=column_names,
                                                            table_ids=table_ids,
                                                            table_names=table_names,
                                                            rule_seq=rule_seq_t,
                                                            rule_stack=rule_stack_t,
                                                            is_root=False,
                                                            parent=parent_t,
                                                            pre=pre_t,
                                                            col_set=col_set)
                rule_stack.extend(rule_stack_t)
                rule_seq.extend(rule_seq_t)
                continue
            if cur_node == Nonterminal("Order"):
                rule_seq, rule_stack = self.expand_order_by(self.order_by,
                                                            rules=rules,
                                                            column_names=column_names,
                                                            table_ids=table_ids,
                                                            table_names=table_names,
                                                            rule_seq=rule_seq,
                                                            rule_stack=rule_stack,
                                                            col_set=col_set)

        return rule_seq, rule_stack

    def expand_a(self, select_unit, rule_seq, rule_stack, rules, column_names, table_ids, table_names, col_set):
        """
        :type rule_seq: list
        :param select_unit:
        :param rule_seq:
        :param rule_stack:
        :param rules:
        :param column_names:
        :param table_ids:
        :param table_names:
        :return:
        """
        agg_id = select_unit[0]
        op = self.AGG_OPS[agg_id]
        rule = rules.get_rule_by_rhs_lhs(rhs=op, lhs="A")
        r = Rule(rule=rule[0], parent=rule_stack[-1], pre=rule_seq[-1], length=2, frontier=1)
        if rule_stack[-1].frontier == rule_stack[-1].length - 1:
            rule_stack.pop()
        rule_seq.append(r)
        rule_stack.append(r)
        rule_seq, rule_stack = self.expand_val_unit(val_unit=select_unit[1],
                                                    rule_stack=rule_stack,
                                                    rule_seq=rule_seq,
                                                    rules=rules,
                                                    column_names=column_names,
                                                    table_ids=table_ids,
                                                    table_names=table_names,
                                                    col_set=col_set)
        return rule_seq, rule_stack

    def expand_val_unit(self, val_unit, rule_seq, rule_stack, rules, column_names, table_ids,
                        table_names, col_set, is_expand_having=False):
        # expand val_unit
        val_unit = val_unit
        unit_op_id = val_unit[0]
        if self.UNIT_OPS[unit_op_id] == 'none':
            length = 2
            frontier_index = 1
        else:
            length = 3
            frontier_index = 1

        rule = rules.get_rule_by_rhs_lhs(rhs=self.UNIT_OPS[unit_op_id], lhs="V")
        r = Rule(rule=rule[0],
                 parent=rule_stack[-1],
                 pre=rule_seq[-1], frontier=frontier_index, length=length)

        if rule_stack[-1].frontier == rule_stack[-1].length - 1:
            rule_stack.pop()
        rule_stack.append(r)
        rule_seq.append(r)

        # expand col_unit
        col_units = val_unit[1:]
        rule_seq, rule_stack = self.expand_col_unit(col_units=col_units,
                                                    rule_seq=rule_seq,
                                                    rule_stack=rule_stack,
                                                    rules=rules,
                                                    table_names=table_names,
                                                    table_ids=table_ids,
                                                    column_names=column_names,
                                                    is_expand_having=is_expand_having,
                                                    col_set=col_set)
        return rule_seq, rule_stack

    def expand_con_unit(self, cond_unit, rule_seq, rule_stack, rules, table_names, table_ids, column_names, col_set,
                        is_expand_having=False):

        cond_unit = cond_unit
        not_ops = cond_unit[0]
        where_op_id = cond_unit[1]
        where_val_unit = cond_unit[2]
        where_val1 = cond_unit[3]
        where_val2 = cond_unit[4]

        contain_sql = True if isinstance(where_val1, dict) else False
        contain_col_unit = True if isinstance(where_val1, list) else False
        if contain_col_unit:
            where_val1 = 1
            where_val2 = None
        # print(where_val1)
        # print(where_val2)
        # print(self.interaction[self.turn_level]['query'])
        where_op = self.WHERE_OPS[where_op_id]
        # print(where_op)
    # if not contain_col_unit:
        if where_op == 'not' and contain_sql:
            rule = rules.get_rule_by_rhs_lhs(lhs="Filter", rhs="not V R")
        elif not_ops and contain_sql:
            rule = rules.get_rule_by_rhs_lhs(lhs="Filter", rhs="not" + "-" + where_op + ' V R')
        elif contain_sql:
            rule = rules.get_rule_by_rhs_lhs(lhs="Filter", rhs=where_op + ' V R')
        # elif where_op == 'not' and contain_col_unit:
        #     rule = rules.get_rule_by_rhs_lhs(lhs="Filter", rhs="not V X")
        # elif not_ops and contain_col_unit:
        #     rule = rules.get_rule_by_rhs_lhs(lhs="Filter", rhs="not" + "-" + where_op + ' V X')
        # elif contain_col_unit:
        #     rule = rules.get_rule_by_rhs_lhs(lhs="Filter", rhs=where_op + ' V X')

        elif where_op == 'not':
            rule = rules.get_rule_by_rhs_lhs(lhs="Filter", rhs="not V Y")
        elif not_ops:
            rule = rules.get_rule_by_rhs_lhs(lhs="Filter", rhs="not" + "-" + where_op + ' V Y')
        else:
            rule = rules.get_rule_by_rhs_lhs(lhs="Filter", rhs=where_op + ' V Y')

        r = Rule(rule=rule[0], pre=rule_stack[-1], parent=rule_stack[-1], length=3, frontier=2)

        if rule_stack[-1].frontier == rule_stack[-1].length - 1:
            rule_stack.pop()
        rule_seq.append(r)
        rule_stack.append(r)

        parent = rule_stack[-1]

        if contain_sql:
            # expand val_unit
            rule_seq, rule_stack = self.expand_val_unit(val_unit=where_val_unit,
                                                        rule_stack=rule_stack,
                                                        rule_seq=rule_seq,
                                                        rules=rules,
                                                        table_names=table_names,
                                                        table_ids=table_ids,
                                                        column_names=column_names,
                                                        is_expand_having=is_expand_having,
                                                        col_set=col_set)
            # expand_sql
            parser = FromSQLParser(where_val1, self.utterance_arg, self.interaction, self.turn_level)
            rule_seq_tmp = []
            rule_stack_tmp = []
            parent_tmp = rule_stack[-1]
            pre_tmp = rule_seq[-1]
            rule_seq_tmp, rule_stack_tmp = parser.parse_sql(root=rule_stack[-1],
                                                            rules=rules,
                                                            rule_seq=rule_seq_tmp,
                                                            rule_stack=rule_stack_tmp,
                                                            table_names=table_names,
                                                            table_ids=table_ids,
                                                            column_names=column_names,
                                                            is_root=False,
                                                            parent=parent_tmp,
                                                            pre=pre_tmp,
                                                            col_set=col_set)

            rule_stack.extend(rule_stack_tmp)
            rule_seq.extend(rule_seq_tmp)
        else:
            rule_seq, rule_stack = self.expand_val_unit(val_unit=where_val_unit,
                                                        rule_stack=rule_stack,
                                                        rule_seq=rule_seq,
                                                        rules=rules,
                                                        table_names=table_names,
                                                        table_ids=table_ids,
                                                        column_names=column_names,
                                                        is_expand_having=is_expand_having,
                                                        col_set=col_set)

            # if contain_col_unit:
                # col_units = [where_val1, None]
                # rule_seq, rule_stack = self.expand_col_unit(col_units=col_units,
                #                                             rule_seq=rule_seq,
                #                                             rule_stack=rule_stack,
                #                                             rules=rules,
                #                                             table_names=table_names,
                #                                             table_ids=table_ids,
                #                                             column_names=column_names,
                #                                             is_expand_having=is_expand_having,
                #                                             col_set=col_set)
                # pass
            # else:
                # expand_Y:
            if True:
                val1_start, val1_end = self.get_val_id(where_val1)
                rule = Production(lhs=Nonterminal('Y'), rhs=[str([val1_start, val1_end])])

                r = Rule(rule=rule, pre=rule_seq[-1], parent=parent, frontier=1, length=2)
                rule_seq.append(r)
                if Nonterminal('between') in parent.rule.rhs():
                    val2_start, val2_end = self.get_val_id(where_val2)
                    rule = Production(lhs=Nonterminal('Y'), rhs=[str([val2_start, val2_end])])
                    r = Rule(rule=rule, pre=rule_seq[-1], parent=parent, frontier=1, length=2)
                    rule_seq.append(r)

        return rule_seq, rule_stack

    def get_val_id(self, where_val):
        is_int = False
        if isinstance(where_val, str):
            punctuation = '!,.;:?"\''
            for i in punctuation:
                if i in where_val:
                    where_val = where_val.replace(i, ' ' + i)
            val_list = where_val.lower().replace("\"", '').replace("%", '').strip().split()
            val = [[wordnet_lemmatizer.lemmatize(s)] for s in val_list]
        elif isinstance(where_val, float):
            if where_val.is_integer():
                is_int = True
                where_val = int(where_val)
            val = [[str(where_val)]]
        elif isinstance(where_val, int):
            is_int = True
            val = [[str(where_val)]]


        utterance_args = []
        for utter in self.interaction[:self.turn_level+1]:
            utterance_args.extend(utter['utterance_arg'])
            utterance_args.append([';'])
        # print(utterance_args)
        # utterance_args.reverse()

        start = -1
        if len(val) == 1:
            utterance_args_copy = copy.deepcopy(utterance_args)
            utterance_args_copy.reverse()
            if val[0] in utterance_args_copy:
                start = utterance_args_copy.index(val[0])
                start = len(utterance_args_copy) - start - 1
            elif is_int and [num2words(val[0][0])] in utterance_args_copy:
                start = utterance_args_copy.index([num2words(val[0][0])])
                start = len(utterance_args_copy) - start - 1
        elif len(val) > 1:
            # utterance_args_copy = copy.deepcopy(utterance_args)
            rs = []
            for i, s in enumerate(utterance_args):
                sdc = []
                if i < len(utterance_args) - len(val) + 1:
                    for j in range(len(val)):
                        sdc.append(utterance_args[i + j])
                    rs.append(sdc)
            rs.reverse()
            if val in rs:
                start = rs.index(val)
                start = len(rs) - start - 1
            elif val[0] == ['the']:
                val.pop(0)
                if val in rs:
                    start = rs.index(val)
                    start = len(rs) - start - 1



        if start == -1:
            end = -1
        else:
            end = start+len(val)
        return start, end

    def expand_filter(self, where, rule_seq: list, rule_stack: list, rules, table_names, table_ids, column_names, col_set,
                      is_expand_having=False):
        """
        :type rule_stack: list
        :param where:
        :param rule_seq:
        :param rule_stack:
        :param rules:
        :param table_names:
        :param table_ids:
        :param column_names:
        :param is_expand_having:
        :return:
        """
        while len(where) > 0:
            # expand con_unit
            if len(where) == 1:
                rule_seq, rule_stack = self.expand_con_unit(cond_unit=where[0],
                                                            rule_seq=rule_seq,
                                                            rule_stack=rule_stack,
                                                            rules=rules,
                                                            table_names=table_names,
                                                            table_ids=table_ids,
                                                            column_names=column_names,
                                                            is_expand_having=is_expand_having,
                                                            col_set=col_set)
                where.remove(where[0])

            if len(where) > 1:
                # generate rule Filter -> and Filter Filter; Filter -> or Filter Filter
                if where[1] == 'and':
                    rule = rules.get_rule_by_rhs_lhs(lhs='Filter', rhs='and')
                    r = Rule(rule=rule[0], pre=rule_seq[-1], parent=rule_stack[-1], frontier=1, length=3)
                elif where[1] == 'or':
                    rule = rules.get_rule_by_rhs_lhs(lhs='Filter', rhs='or')
                    r = Rule(rule=rule[0], pre=rule_seq[-1], parent=rule_stack[-1], frontier=1, length=3)
                else:
                    rule = None
                    r = None

                if rule is None:
                    raise Exception('Where-clause include invalid syntax.')

                if rule_stack[-1].frontier == rule_stack[-1].length - 1:
                    rule_stack.pop()
                rule_seq.append(r)
                rule_stack.append(r)

                rule_seq, rule_stack = self.expand_con_unit(cond_unit=where[0],
                                                            rule_seq=rule_seq,
                                                            rule_stack=rule_stack,
                                                            rules=rules,
                                                            table_names=table_names,
                                                            table_ids=table_ids,
                                                            column_names=column_names,
                                                            is_expand_having=is_expand_having,
                                                            col_set=col_set)

                where.pop(0)
                where.pop(0)

        while len(self.having) > 0:
            if len(self.having) == 1:
                rule_seq, rule_stack = self.expand_con_unit(cond_unit=self.having[0],
                                                            rule_seq=rule_seq,
                                                            rule_stack=rule_stack,
                                                            rules=rules,
                                                            table_names=table_names,
                                                            table_ids=table_ids,
                                                            column_names=column_names,
                                                            is_expand_having=True,
                                                            col_set=col_set)
                self.having.remove(self.having[0])

            if len(self.having) > 1:
                rule = rules.get_rule_by_rhs_lhs(lhs='Filter', rhs='and')
                r = Rule(rule=rule[0], pre=rule_seq[-1], parent=rule_stack[-1], frontier=1, length=3)

                if rule_stack[-1].frontier == rule_stack[-1].length - 1:
                    rule_stack.pop()
                rule_seq.append(r)
                rule_stack.append(r)

                rule_seq, rule_stack = self.expand_con_unit(cond_unit=self.having[0],
                                                            rule_seq=rule_seq,
                                                            rule_stack=rule_stack,
                                                            rules=rules,
                                                            table_names=table_names,
                                                            table_ids=table_ids,
                                                            column_names=column_names,
                                                            is_expand_having=True,
                                                            col_set=col_set)

                self.having.pop(0)
                self.having.pop(0)

        return rule_seq, rule_stack

    def expand_having(self, where: list, rule_seq, rule_stack, rules, table_names, table_ids, column_names, col_set):
        while len(where) > 0:

            # expand con_unit
            if len(where) == 1:
                rule_seq, rule_stack = self.expand_con_unit(cond_unit=where[0],
                                                            rule_seq=rule_seq,
                                                            rule_stack=rule_stack,
                                                            rules=rules,
                                                            table_names=table_names,
                                                            table_ids=table_ids,
                                                            column_names=column_names,
                                                            col_set=col_set)
                where.remove(where[0])

            if len(where) > 1:
                # generate rule Filter -> and Filter Filter; Filter -> or Filter Filter
                #
                if where[1] == 'and':
                    rule = rules.get_rule_by_rhs_lhs(lhs='Filter', rhs='and')
                    r = Rule(rule=rule[0], pre=rule_seq[-1], parent=rule_stack[-1], frontier=1, length=3)
                elif where[1] == 'or':
                    rule = rules.get_rule_by_rhs_lhs(lhs='Filter', rhs='or')
                    r = Rule(rule=rule[0], pre=rule_seq[-1], parent=rule_stack[-1], frontier=1, length=3)
                else:
                    rule = None
                    r = None

                if rule is None:
                    raise Exception('Where-clause Include invalid syntax.')

                if rule_stack[-1].frontier == rule_stack[-1].length - 1:
                    rule_stack.pop()
                rule_seq.append(r)
                rule_stack.append(r)

                rule_seq, rule_stack = self.expand_con_unit(cond_unit=where[0],
                                                            rule_seq=rule_seq,
                                                            rule_stack=rule_stack,
                                                            rules=rules,
                                                            table_names=table_names,
                                                            table_ids=table_ids,
                                                            column_names=column_names,
                                                            col_set=col_set)

                where.pop(0)
                where.pop(0)
        return rule_seq, rule_stack

    def expand_col_unit(self, col_units: list, rule_seq, rule_stack, rules, table_names, table_ids, column_names, col_set,
                        is_expand_having):

        if col_units[1] is None:
            col_agg_id = col_units[0][0]
            col_id = col_units[0][1]
            rule = rules.get_rule_by_rhs_lhs(rhs=self.AGG_OPS[col_agg_id], lhs='X')
            r = Rule(parent=rule_stack[-1], rule=rule[0], pre=rule_seq[-1], length=3, frontier=2)
            if rule_stack[-1].frontier == rule_stack[-1].length - 1:
                rule_stack.pop()
            rule_stack.append(r)
            rule_seq.append(r)

            # expand C and T
            col_name = column_names[col_id]
            col_set_id = col_set.index(col_name)
            if col_id == 0 and is_expand_having:
                g_col_id = self.group_by[0][1]
                # g_col_name = column_names[g_col_id]
                # col_id_tmp = col_set.index(g_col_name)

                tb_id = table_ids[g_col_id]
                # tb_name = table_names[table_ids[tb_id]]
            elif col_id == 0:
                # n/a
                tb_id = -1
                tb_name = "n/a"
            else:
                tb_id = table_ids[col_id]
                # tb_name = table_names[tb_id]

            c = Production(lhs=Nonterminal('C'), rhs=[col_set_id])
            t = Production(lhs=Nonterminal('T'), rhs=[tb_id])

            c_rule = Rule(rule=c, parent=rule_stack[-1], pre=rule_seq[-1], length=1, frontier=0)
            t_rule = Rule(rule=t, parent=rule_stack[-1], pre=rule_seq[-1], length=1, frontier=0)
            rule_seq.extend([c_rule, t_rule])
        else:
            parent = rule_stack[-1]

            for col in col_units:
                col_agg_id = col[0]
                col_id = col[1]
                rule = rules.get_rule_by_rhs_lhs(rhs=self.AGG_OPS[col_agg_id], lhs='X')
                r = Rule(rule=rule[0], parent=parent, pre=rule_seq[-1], length=3, frontier=2)
                # if rule_stack[-1].frontier == rule_stack[-1].length - 1:
                #     rule_stack.pop()
                # rule_stack.append(r)
                rule_seq.append(r)

                # expand C and T
                col_name = column_names[col_id]
                col_set_id = col_set.index(col_name)
                if col_id == 0 and is_expand_having:
                    g_col_id = self.group_by[0][1]
                    # g_col_name = column_names[g_col_id]
                    # col_id_tmp = col_set.index(g_col_name)
                    tb_id = table_ids[g_col_id]
                    # tb_name = table_names[table_ids[tb_id]]
                elif col_id == 0:
                    # n/a
                    tb_id = -1
                    # tb_name = "n/a"
                else:
                    tb_id = table_ids[col_id]
                    # tb_name = table_names[tb_id]

                # c = Production(lhs=Nonterminal('C'), rhs=[col_name])
                # t = Production(lhs=Nonterminal('T'), rhs=[tb_name])
                c = Production(lhs=Nonterminal('C'), rhs=[col_set_id])
                t = Production(lhs=Nonterminal('T'), rhs=[tb_id])

                c_rule = Rule(rule=c, parent=rule_stack[-1], pre=rule_seq[-1], length=1, frontier=0)
                t_rule = Rule(rule=t, parent=rule_stack[-1], pre=rule_seq[-1], length=1, frontier=0)
                rule_seq.extend([c_rule, t_rule])
        return rule_seq, rule_stack

    def expand_group_by(self, col_units: list, rule_seq, rule_stack, rules, table_names, table_ids, column_names, col_set):
        rule_seq, rule_stack = self.expand_col_unit(col_units=col_units,
                                                    rule_seq=rule_seq,
                                                    rule_stack=rule_stack,
                                                    rules=rules,
                                                    table_names=table_names,
                                                    table_ids=table_ids,
                                                    column_names=column_names,
                                                    is_expand_having=True,
                                                    col_set=col_set)
        return rule_seq, rule_stack

    def expand_order_by(self, order_by: list, rule_seq, rule_stack, rules, table_names, table_ids, column_names, col_set):
        if order_by[0] == 'asc':
            rule = Production(lhs=Nonterminal('Order'), rhs=[Nonterminal('asc'), Nonterminal('V')])
        elif order_by[0] == 'desc':
            rule = Production(lhs=Nonterminal('Order'), rhs=[Nonterminal('desc'), Nonterminal('V')])
        else:
            rule = None
        if rule is None:
            raise Exception('order-by identifier is asc or desc.')

        r = Rule(rule=rule, parent=rule_stack[-1], pre=rule_seq[-1], length=2, frontier=1)
        rule_stack.pop()
        rule_stack.append(r)
        rule_seq.append(r)
        val_units = order_by[1]

        for val_unit in val_units:
            rule_seq, rule_stack = self.expand_val_unit(val_unit=val_unit,
                                                        rule_stack=rule_stack,
                                                        rule_seq=rule_seq,
                                                        rules=rules,
                                                        column_names=column_names,
                                                        table_ids=table_ids,
                                                        table_names=table_names,
                                                        col_set=col_set)
        return rule_seq, rule_stack


class FromAstToSql(object):
    def __init__(self, ast, col_set, table_names, col_table_dict,  args, schema, utter, column_names, origin_column_names,origin_table_names,
                 pretrain_weights_shortcut=None, tokenizer=None,):
        self.ast = ast[0]
        self.is_group_by = ast[1]
        print(self.is_group_by)
        # self.col_ids = col_ids
        # self.table_ids = table_ids
        # self.db_id = db_id
        self.table_names = table_names
        self.col_set = col_set
        self.col_table_dict = col_table_dict
        self.utter = utter
        # print(utter)
        preprocess_schema(schema)
        self.schema = schema
        self.column_names = column_names
        self.origin_column_names = origin_column_names
        self.origin_table_names = origin_table_names
        # self.col_table_ids = col_table_ids
        # if tokenizer is None:
        #     self.tokenizer = BertTokenizer.from_pretrained(pretrain_weights_shortcut, do_lower_case=False)
        # else:
        #     self.tokenizer = tokenizer
        self.sql_expr = ''
        r = RuleGenerator()
        self.rule_dict = r.get_rule_ids()
        self.select_expr = []
        # self.col_tokens = process_schema_output(col_ids, tokenizer)
        # self.table_tokens = process_schema_output(table_ids, tokenizer)
        # self.datas, self.schema = load_dataSets(args=args)
        # self.database_until = DatabaseUtil('../data/tables.json')

    def parse(self):
        # self.ast.pretty_print(maxwidth=10)
        return self.parse_z(self.ast)

        # root = ast.
        # [101, 115, 102, 4706, 25021, 102, 2450, 102, 1271, 102, 3211, 102, 2439, 102, 6905, 102, 1903, 102, 2483, 25021,
        #  102, 1271, 102, 1583, 102, 1461, 1271, 102, 1461, 1836, 1214, 102, 1425, 102, 1110, 2581, 102, 3838, 25021, 102,
        #  3838, 1271, 102, 3815, 102, 4706, 25021, 102, 1214, 102, 3838, 25021, 102, 2483, 25021, 102, 0, 0, 0, 0, 0, 0, 0,
        #  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        #  [101, 4706, 102, 2483, 102, 3838, 102, 2483, 1107, 3838, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        #  51

    def parse_z(self, ast):
        z_expr = []
        rule_id = ast.rule_id
        # print(list(ast.subtrees(lambda t: t.label() == 'R')))
        if Nonterminal('intersect') in self.rule_dict[rule_id].rhs():
            sql_expr1 = self.parse_sql(list(ast.subtrees(lambda t: t.label() == 'R'))[0])
            sql_expr2 = self.parse_sql(list(ast.subtrees(lambda t: t.label() == 'R'))[1])
            z_expr.extend([sql_expr1, 'INTERSECT', sql_expr2])
        elif Nonterminal('union') in self.rule_dict[rule_id].rhs():
            sql_expr1 = self.parse_sql(list(ast.subtrees(lambda t: t.label() == 'R'))[0])
            sql_expr2 = self.parse_sql(list(ast.subtrees(lambda t: t.label() == 'R'))[1])
            z_expr.extend([sql_expr1, 'UNION', sql_expr2])
        elif Nonterminal('except') in self.rule_dict[rule_id].rhs():
            sql_expr1 = self.parse_sql(list(ast.subtrees(lambda t: t.label() == 'R'))[0])
            sql_expr2 = self.parse_sql(list(ast.subtrees(lambda t: t.label() == 'R'))[1])
            z_expr.extend([sql_expr1, 'EXCEPT', sql_expr2])
        else:
            sql_expr = self.parse_sql(list(ast.subtrees(lambda t: t.label() == 'R'))[0])
            z_expr.extend([sql_expr])
        return z_expr

    def parse_sql(self, ast):
        sql_expr = []
        rule_id = ast.rule_id
        rule = self.rule_dict[rule_id]
        col_id_ls = set()
        tab_id_ls = dict()
        # if Nonterminal('distinct') in rule.rhs():
        #     sql_expr.extend(['DISTINCT'])
        select_expr = []
        filter_expr = []
        order_expr = []
        for n in rule.rhs():
            if str(n) == 'Select':
                # print(list(ast.subtrees(lambda t: t.label() == 'Select')))

                select_ast = list(ast.subtrees(lambda t: t.label() == 'Select'))[0]
                select_expr = self.parse_select(select_ast, col_id_ls, tab_id_ls)
                if Nonterminal('distinct') in rule.rhs():
                    select_expr.insert(1,'DISTINCT')
                # sql_expr.append(select_expr)
            elif str(n) == 'Filter':
                filter_ast = list(ast.subtrees(lambda t: t.label() == 'Filter'))[0]
                # print(filter_ast)
                filter_expr = self.parse_filter(filter_ast, col_id_ls, tab_id_ls)
                # filter_expr = ['WHERE', filter_expr]
                # sql_expr.append(filter_expr)
            elif str(n) == 'Order':
                order_ast = list(ast.subtrees(lambda t: t.label() == 'Order'))[0]
                order_expr = self.parse_order_by(order_ast, col_id_ls, tab_id_ls)
                # sql_expr.append(order_expr)

        # from_expr = self.parse_from(list(col_id_ls))
        from_expr = infer_from_clause(tab_id_ls, self.schema, col_id_ls)
        # print(select_expr)
        if self.is_group_by:
            group_by_expr = None
            # group_by_expr = self.get_group_by_expr(select_expr)
        else:
            group_by_expr = None

        sql_expr.extend(self.update_table_name(select_expr, tab_id_ls))
        sql_expr.append(from_expr)
        sql_expr.extend(self.update_table_name(filter_expr, tab_id_ls))


        if group_by_expr is not None and self.is_group_by:
            print(group_by_expr)
            sql_expr.append(self.update_table_name([group_by_expr], tab_id_ls))
        sql_expr.extend(self.update_table_name(order_expr, tab_id_ls))
        return sql_expr

    def get_group_by_expr(self, select_expr):
        flattened_str = [str(s) for s in self.flatten(copy.deepcopy(select_expr))]
        flattened_str = ' '.join(flattened_str)
        sel_seg = flattened_str.replace('SELECT', ' ').strip().split(',')
        for s in sel_seg:
            tmp = False
            for agg in ['min', 'max', 'count', 'sum', 'avg']:
                if agg in s:
                    tmp = True
            if tmp:
                continue
            else:
                return 'GROUP BY '+s.strip()+' '
        return None

    def update_table_name(self, expr, tab_dict):
        flattened_str = self.flatten(expr)
        flattened_str = [str(s) for s in flattened_str]
        flattened_str = ' '.join(flattened_str)

        # indexing table name
        # if len(tab_dict.keys()) > 1:
        #     for k, v in tab_dict.items():
        #         if k in flattened_str:
        #             flattened_str = flattened_str.replace(k, v)
        # elif len(tab_dict.keys()) == 1:
        #     for k, v in tab_dict.items():
        #         if k in flattened_str:
        #             flattened_str = flattened_str.replace(k + '.', '')
        return [flattened_str]

    def parse_select(self, ast, col_ids, tab_ids):
        select_expr = []
        rule_id = ast.rule_id
        rule = self.rule_dict[rule_id]
        count = len(rule.rhs())
        l = list(ast.subtrees(lambda t: t.label() == 'A'))
        select_expr.append('SELECT')
        for i in range(count):

            a_ast = l[i]
            # print(a_ast)
            a_expr = self.parse_a(a_ast, col_ids, tab_ids, True)
            select_expr.extend(a_expr)
            self.select_expr.append(a_expr)
            if i < (count-1):
                select_expr.extend(',')
        return select_expr

    def parse_a(self, ast, col_ids, tab_ids, is_select=False):
        a_expr = []
        rule_id = ast.rule_id
        rule = self.rule_dict[rule_id]
        agg = str(rule.rhs()[0])
        a = list(ast.subtrees(lambda t: t.label() == 'V'))[0]
        v_expr, _ = self.parse_v(a, col_ids, tab_ids)
        if agg == 'none':
            a_expr.extend(v_expr)
        else:
            v_expr, _ = self.parse_v(a, col_ids, tab_ids)
            v_expr = self.flatten(v_expr)
            v_expr = ' '.join(v_expr)
            if agg == 'count' and is_select:
                a_expr.extend([agg + ' (' +' DISTINCT '+ v_expr + ' )'])
            else:
                a_expr.extend([agg+' ( '+ v_expr+ ' )'])

        return a_expr

    def parse_v(self, ast, col_ids, tab_ids):
        v_expr = []
        rule_id = ast.rule_id
        rule = self.rule_dict[rule_id]
        ops = str(rule.rhs()[0])
        # print(ops)
        # print(list(ast.subtrees())[1])
        # print(list(ast.subtrees())[2])
        all_flag = False
        if ops == 'sub':
            ops = '-'
        elif ops == 'add':
            ops = '+'
        elif ops == 'mul':
            ops = '*'
        elif ops == 'div':
            ops = '/'
        l = list(ast.subtrees(lambda t: t.label() == 'X'))
        if ops == 'none':
            x = l[0]
            v_expr, all_flag = self.parse_x(x, col_ids, tab_ids)
        else:
            l = list(ast.subtrees(lambda t: t.label() == 'X'))
            x1 = l[0]
            x2 = l[1]
            x_expr1, _ = self.parse_x(x1, col_ids, tab_ids)
            x_expr2, _ = self.parse_x(x2, col_ids, tab_ids)
            v_expr.extend([x_expr1, ops, x_expr2])
        return v_expr, all_flag

    def parse_x(self, ast, col_ids, tab_ids):
        # from src.run_sparc_dataset_utils import convert_id_to_db_name
        x_expr = []
        rule_id = ast.rule_id
        rule = self.rule_dict[rule_id]
        all_flag = False
        agg = str(rule.rhs()[0])
        if agg == 'avg':
            v = list(ast.subtrees(lambda x: x.label() == 'V'))[0]
            v_expr, _ = self.parse_v(v, col_ids, tab_ids)
            x_expr.extend([agg, v_expr])
        else:
            # print(list(ast.subtrees()))
            c = list(ast.subtrees())[1]
            t = list(ast.subtrees())[2]

            col_set = self.col_set
            table_names = self.table_names
            # col_name = self.col_tokens[c_id]
            # tab_name = self.table_tokens[t_id]
            c_id = c[0]
            t_id = t[0]
            # print(t_id)
            # print(table_names)
            # print(self.ast)
            col_name = col_set[c_id]
            tab_name = table_names[t_id]
            col_name = self.origin_column_names[self.column_names.index(col_name)]
            tab_name = self.origin_table_names[self.table_names.index(tab_name)]

            # print(table_names)
            # if c_id == 0 and len(col_ids) > 0:
            #     pass
            # else:
            if tab_name not in tab_ids.keys():
                tab_note = str(len(tab_ids.keys())+1)
                tab_ids[tab_name] = 'T' + tab_note

            col_ids.add((agg, tab_name, col_name))
            # tab_ids.add(t_id)
            if len(tab_ids.keys()) > 1:
                t_name = tab_name
            else:
                t_name = tab_name
            # t_name = t_name.upper()

            is_count = False
            for u in self.utter:
                if 'many' in u:
                    is_count = True
            if c_id == 0 and agg == 'none':
                x_expr.extend([col_name])
            elif c_id == 0 and agg == 'none' and is_count:
                x_expr.extend(['count'+'('+'\"'+ col_name+'\"'+ ')'])
                # all_flag = True
            elif c_id == 0 and agg != 'none':
                x_expr.extend([agg+'('+'\"'+col_name+'\"'+')'])
            elif agg == 'none':
                x_expr.extend([t_name+'.'+'\"'+col_name+'\"'])
            elif agg == 'count':
                x_expr.extend([agg+'('+t_name+'.'+'\"'+col_name+'\"'+')'])
                all_flag = True
            else:
                x_expr.extend([agg+'('+ t_name+ '.'+ '\"'+ col_name+ '\"' + ')'])

        return x_expr, all_flag

    def parse_filter(self, ast, col_ids, tab_ids):
        where_ops = {'not': 'NOT',
                     'between': 'between',
                     'e': '=',
                     'mt': '>',
                     'lt': '<',
                     'mte': '>=',
                     'lte': '<=',
                     'ne': '!=',
                     'in': 'IN',
                     'like': 'LIKE',
                     'is': 'IS',
                     'exists': 'exists',
                     'not-in': 'NOT IN',
                     'not-like': 'NOT LIKE',
                     'not-exists': 'not exists'
                     }

        filter_expr = []
        rule_id = ast.rule_id
        rule = self.rule_dict[rule_id]
        ops = str(rule.rhs()[0])
        choice = str(rule.rhs()[2])
        if ops == 'and' or ops == 'or':
            ast.pretty_print()
            filter_ast1 = list(ast.subtrees(lambda t: t.label() == 'Filter'))[1]
            filter_ast2 = list(ast.subtrees(lambda t: t.label() == 'Filter'))[2]
            filter_ast1.pretty_print()
            filter_ast2.pretty_print()

            filter_expr1 = self.parse_filter(filter_ast1, col_ids, tab_ids)
            filter_expr2 = self.parse_filter(filter_ast2, col_ids, tab_ids)
            if len(filter_expr1) > 0 and len(filter_expr2) > 0:
                filter_expr1.pop(0)
                filter_expr2.pop(0)
            filter_expr.extend(['WHERE', filter_expr1, ops, filter_expr2])
            print(filter_expr)
            return filter_expr

        if choice == 'Y':
            if ops == 'between':
                v = list(ast.subtrees(lambda t: t.label() == 'V'))[0]
                l = list(ast.subtrees(lambda t: t.label() == 'Y'))
                y1 = l[0]
                y2 = l[1]
                y_val1 = y1[0]
                y_val2 = y2[0]
                v_expr, _ = self.parse_v(v, col_ids, tab_ids)
                filter_expr.extend(['WHERE', v_expr, 'BETWEEN', ' \'val1\' ', 'AND', ' \'val2\' '])

                # filter_expr.extend(['WHERE', v_expr,'BETWEEN', self.val_to_str(y_val1), 'AND', self.val_to_str(y_val2)])
            else:
                v = list(ast.subtrees(lambda t: t.label() == 'V'))[0]
                y = list(ast.subtrees(lambda t: t.label() == 'Y'))[0]
                y_val = y[0]
                v_expr, all_flag = self.parse_v(v, col_ids, tab_ids)

                if v_expr[0] == 'count(*)':
                    # print(self.ast)
                    filter_expr.extend(['GROUP BY', self.select_expr[0], 'HAVING', v_expr, where_ops[ops], self.val_to_str(y_val)])
                    self.is_group_by = False
                else:
                    filter_expr.extend([ 'WHERE', v_expr, where_ops[ops], self.val_to_str(y_val)])

        elif choice == 'R':
            if ops == 'between':
                v = list(ast.subtrees(lambda t: t.label() == 'V'))[0]
                r = list(ast.subtrees(lambda t: t.label() == 'R'))[0]
                sql_expr = self.parse_sql(r)
                v_expr, _ = self.parse_v(v, col_ids, tab_ids)
                filter_expr.extend(['WHERE', v_expr,'BETWEEN','(', sql_expr, ')'])
            else:
                v = list(ast.subtrees(lambda t: t.label() == 'V'))[0]
                r = list(ast.subtrees(lambda t: t.label() == 'R'))[0]
                sql_expr = self.parse_sql(r)
                v_expr, all_flag = self.parse_v(v, col_ids, tab_ids)
                filter_expr.extend(['WHERE', v_expr, where_ops[ops], '(', sql_expr, ')'])

        print(filter_expr)
        return filter_expr

    def val_to_str(self, y_val):
        if eval(y_val)[0] == -1:
            return 'val'
        else:
            val_str = self.utter[eval(y_val)[0]: eval(y_val)[1]+1]
            if len(val_str) == 0:
                return 'val'
            else:
                flat_val_str = []
                for i in val_str:
                    flat_val_str.extend(i)
                return '\"'+' '.join(flat_val_str)+'\"'
        # return '\'val\''

    def parse_order_by(self, ast, col_ids, tab_ids):
        order_expr = []
        rule_id = ast.rule_id
        rule = self.rule_dict[rule_id]
        ops = str(rule.rhs()[0])
        is_limit = False
        for u in self.utter:
            if 'st' in u:
                is_limit = True
        if ops == 'asc':
            v = list(ast.subtrees(lambda t: t.label() == 'V'))[0]
            v_expr,_ = self.parse_v(v, col_ids, tab_ids)

            if v_expr[0] == 'count(*)':
                # if not self.is_group_by:
                order_expr.extend(['GROUP BY', self.select_expr[0], 'ORDER BY', v_expr, 'ASC'])
                self.is_group_by = False
                # else:
                # order_expr.extend(['ORDER BY', v_expr, 'ASC', 'LIMIT 1'])

                # self.is_group_by = False
            else:
                order_expr.extend(['ORDER BY', v_expr, 'ASC'])
            if is_limit:
                order_expr.extend(['LIMIT 1'])
            # order_expr.extend(['GROUP BY', self.select_expr[0], 'ORDER BY', v_expr, 'ASC', 'LIMIT 1'])
        else:
            v = list(ast.subtrees(lambda t: t.label() == 'V'))[0]
            v_expr,_ = self.parse_v(v, col_ids, tab_ids)
            if v_expr[0] == 'count(*)':

                order_expr.extend(['GROUP BY', self.select_expr[0], 'ORDER BY', v_expr,'DESC' ])
                self.is_group_by = False

                # order_expr.extend(['ORDER BY', v_expr, 'DESC', 'LIMIT 1'])
            else:
                order_expr.extend(['ORDER BY', v_expr, 'DESC'])
            if is_limit:
                order_expr.extend(['LIMIT 1'])


        return order_expr

    def flatten(self, l):
        for el in l:
            if hasattr(el, "__iter__") and not isinstance(el, str):
                for sub in self.flatten(el):
                    yield sub
            else:
                yield el



def preprocess_schema(schema):
    tmp_col = []
    for cc in [x[1] for x in schema['column_names']]:
        if cc not in tmp_col:
            tmp_col.append(cc)
    schema['col_set'] = tmp_col
    # print table
    schema['schema_content'] = [col[1] for col in schema['column_names']]
    schema['col_table'] = [col[0] for col in schema['column_names']]
    graph = build_graph(schema)
    schema['graph'] = graph

def build_graph(schema):
    relations = list()
    foreign_keys = schema['foreign_keys']
    for (fkey, pkey) in foreign_keys:
        fkey_table = schema['table_names_original'][schema['column_names'][fkey][0]]
        pkey_table = schema['table_names_original'][schema['column_names'][pkey][0]]
        relations.append((fkey_table, pkey_table))
        relations.append((pkey_table, fkey_table))
    return Graph(relations)

def infer_from_clause(table_names, schema, columns):
    tables = list(table_names.keys())
    # print(tables)
    start_table = None
    end_table = None
    join_clause = list()
    if len(tables) == 1:
        join_clause.append((tables[0], table_names[tables[0]]))
    elif len(tables) == 2:
        use_graph = True
        # print(schema['graph'].vertices)
        for t in tables:
            # print(t)
            if t not in schema['graph'].vertices:
                use_graph = False
                break
        if use_graph:
            start_table = tables[0]
            end_table = tables[1]
            _tables = list(schema['graph'].dijkstra(tables[0], tables[1]))
            # print('Two tables: ', _tables)
            max_key = 1
            for t, k in table_names.items():
                _k = int(k[1:])
                if _k > max_key:
                    max_key = _k
            for t in _tables:
                if t not in table_names:
                    table_names[t] = 'T' + str(max_key + 1)
                    max_key += 1
                join_clause.append((t, table_names[t],))
        else:
            join_clause = list()
            for t in tables:
                join_clause.append((t, table_names[t],))
    else:
        # > 2
        # print('More than 2 table')
        for t in tables:
            join_clause.append((t, table_names[t],))

    if len(join_clause) >= 3:
        star_table = None
        for agg, col, tab in columns:
            if col == '*':
                star_table = tab
                break
        if star_table is not None:
            star_table_count = 0
            for agg, col, tab in columns:
                if tab == star_table and col != '*':
                    star_table_count += 1
            if star_table_count == 0 and ((end_table is None or end_table == star_table) or (start_table is None or start_table == star_table)):
                # Remove the table the rest tables still can join without star_table
                new_join_clause = list()
                for t in join_clause:
                    if t[0] != star_table:
                        new_join_clause.append(t)
                join_clause = new_join_clause

    relations = dict()
    foreign_keys = schema['foreign_keys']
    primary_keys = schema['primary_keys']
    c = [i[1] for i in columns]
    for (fkey, pkey) in foreign_keys:
        fkey_table = schema['column_names_original'][fkey][0]
        pkey_table = schema['column_names_original'][pkey][0]
        if (fkey_table, pkey_table) in relations.keys():
            if schema['column_names_original'][fkey] in c or schema['column_names_original'][fkey] in c:
                relations[(fkey_table, pkey_table)] = (fkey, pkey)
                relations[(pkey_table, fkey_table)] = (pkey, fkey)
        else:
            relations[(fkey_table, pkey_table)] = (fkey, pkey)
            relations[(pkey_table, fkey_table)] = (pkey, fkey)
        # relations.append((pkey_table, fkey_table))
    # for i in join_clause:
    # print(relations)
    # print(join_clause)
    if len(join_clause) == 1:
        # from_str = join_clause[0][0].upper()
        from_str = join_clause[0][0]

    elif len(join_clause) == 2:
        # print(join_clause)
        # print(foreign_keys)
        # print(schema['column_names'])
        # print(primary_keys)
        # print(schema['table_names_original'])
        ft = schema['table_names_original'].index(join_clause[0][0])
        tt = schema['table_names_original'].index(join_clause[1][0])
        # print(ft,tt)
        if (ft, tt) not in relations.keys():
            if ft > len(primary_keys) - 1 or tt > len(primary_keys) - 1:
                join_con = ''
            else:

                # join_con = 'ON %s.%s = %s.%s' % (
                # join_clause[0][1], schema['column_names_original'][primary_keys[ft]][1],
                # join_clause[1][1], schema['column_names_original'][primary_keys[tt]][1])

                join_con = 'ON %s.%s = %s.%s' % (
                    join_clause[0][0], schema['column_names_original'][primary_keys[ft]][1],
                    join_clause[1][0], schema['column_names_original'][primary_keys[tt]][1])
        else:
            link = relations[(ft, tt)]

            # join_con = 'ON %s.%s = %s.%s' % (join_clause[0][1], schema['column_names_original'][link[0]][1],
            #                                  join_clause[1][1], schema['column_names_original'][link[1]][1])

            join_con = 'ON %s.%s = %s.%s' % (join_clause[0][0], schema['column_names_original'][link[0]][1],
                                         join_clause[1][0], schema['column_names_original'][link[1]][1])

        # join table (indexing)
        # from_str = ' INNER JOIN '.join(['%s AS %s' % (jc[0], jc[1]) for jc in join_clause])
        # without indexing
        from_str = ' INNER JOIN '.join(['%s' % jc[0] for jc in join_clause])


        from_str = ' '.join([from_str, join_con])

    else:
        from_str_list = []
        for i in range(len(join_clause)):
            if i == 0:
                # from_str_list.append('%s AS %s' % (join_clause[i][0], join_clause[i][1]))
                from_str_list.append('%s' % (join_clause[i][0]))

            else:

                ft = schema['table_names_original'].index(join_clause[i-1][0])
                tt = schema['table_names_original'].index(join_clause[i][0])
                # link = relations[(ft, tt)]
                if (ft, tt) not in relations.keys():
                    if ft > len(primary_keys) - 1 or tt > len(primary_keys) - 1 :
                        # join_clause = ''
                        pass
                    else:


                        # from_str_list.append('%s AS %s ON %s.%s = %s.%s' % (join_clause[i][0],
                        #                                                     join_clause[i][1],
                        #                                                     join_clause[i - 1][1],
                        #                                                     schema['column_names_original'][
                        #                                                         primary_keys[ft]][1],
                        #                                                     join_clause[i][1],
                        #                                                     schema['column_names_original'][
                        #                                                         primary_keys[tt]][1]))

                        from_str_list.append('%s ON %s.%s = %s.%s' % (join_clause[i][0],
                                                                            join_clause[i-1][0],
                                                                            schema['column_names_original'][
                                                                                primary_keys[ft]][1],
                                                                            join_clause[i][0],
                                                                            schema['column_names_original'][
                                                                                primary_keys[tt]][1]))


                else:
                    link = relations[(ft, tt)]

                    #
                    # from_str_list.append('%s AS %s ON %s.%s = %s.%s' % (join_clause[i][0],
                    #                                         join_clause[i][1],
                    #                                         join_clause[i - 1][1],
                    #                                         schema['column_names_original'][link[0]][1],
                    #                                         join_clause[i][1],
                    #                                         schema['column_names_original'][link[1]][1]))

                    from_str_list.append('%s ON %s.%s = %s.%s' % (join_clause[i][0],
                                                                        join_clause[i-1][0],
                                                                        schema['column_names_original'][link[0]][1],
                                                                        join_clause[i][0],
                                                                        schema['column_names_original'][link[1]][1]))

        from_str = ' INNER JOIN '.join(from_str_list)
    return 'FROM ' + from_str

def col_to_str(agg, col, tab, table_names, N=1):
    _col = col.replace(' ', '_')
    if agg == 'none':
        if tab not in table_names:
            table_names[tab] = 'T' + str(len(table_names) + N)
        table_alias = table_names[tab]
        if col == '*':
            return '*'
        return '%s.%s' % (table_alias, _col)
    else:
        if col == '*':
            if tab is not None and tab not in table_names:
                table_names[tab] = 'T' + str(len(table_names) + N)
            return '%s(%s)' % (agg, _col)
        else:
            if tab not in table_names:
                table_names[tab] = 'T' + str(len(table_names) + N)
            table_alias = table_names[tab]
            return '%s(%s.%s)' % (agg, table_alias, _col)


def process_schema_output(ids, tokenizer):
    new_embedding = []
    tmp = []
    for i, id in enumerate(ids):
        token = tokenizer.decode([id])
        if id == 0:
            new_embedding.append(token)

        if token == '[CLS]' or token == '[SEP]':
            if len(tmp) > 0:
                t = tmp[0]
                for j, o in enumerate(tmp):
                    if j == 0:
                        continue
                    t+=' '
                    t += o

                new_embedding.append(t)
            tmp = []
            continue

        tmp.append(token)
    return new_embedding


