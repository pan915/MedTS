from nltk import CFG
from nltk.grammar import Nonterminal, Production
from nltk.tree import ParentedTree
from nltk.grammar import nonterminals
from shared_variables import SOS_id, EOS_id, PAD_id, RULE_TYPE, COL_TYPE, TAB_TYPE,VAL_TYPE
import json


class RuleGenerator(object):
    """Generate rules in format of Context Free Grammar(CFG)

    Returns:
        grammar: grammar in CFG format, contains all productions of grammar
        rule_set: dict. a dict: {index: rule}, the rule is a object of
        grammar.production
        rule_set_tree: dict. A dict: {index: rule}, the rule is a object
        of ParentedTree
        non_terminals: Nonterminals. all non_terminals in grammar.
        terminals: Nonterminals. all terminals in grammar
    """

    def __init__(self):

        # self.grammar = CFG.fromstring("""
        # Z -> 'intersect' R R | 'union' R R | 'except' R R | R
        # R -> Select | Select Filter | Select Order | Select Superlative | Select Order Filter | Select Superlative Filter
        # Select -> A | A A | A A A | A A A A
        # Order -> 'asc' A | 'desc' A
        # Superlative -> 'most' A| 'least' A
        # Filter -> 'and' Filter Filter | 'or' Filter Filter | '>' A| '>' A R | '<' A| '<' A R | '≥' A | '≥' A R | '=' A | '=' A R | '≠' A| '≠' A R | 'between' A | 'like' A | 'not-like' A | 'in' A R | 'not-in' A R
        # A -> 'max' C T | 'min' C T | 'count' C T | 'sum' C T | 'avg' C T | 'none' C T
        # C -> column
        # T -> t
        # """)
        # self.grammar = CFG.fromstring("""
        # Z -> 'intersect' R R | 'union' R R | 'except' R R | R
        # R -> Select | Select Filter | Select Order | Select Order Filter | SelectDistinct | SelectDistinct Filter | SelectDistinct Order | SelectDistinct Order Filter
        # Select -> A | A A | A A A | A A A A
        # SelectDistinct -> A | A A | A A A | A A A A
        # Order -> 'asc' A | 'desc' A
        # Filter -> 'and' Filter Filter | 'or' Filter Filter | '>' A| '>' A R | '<' A| '<' A R | '≥' A | '≥' A R | '=' A | '=' A R | '≠' A| '≠' A R | 'between' A | 'like' A | 'not-like' A | 'in' A R | 'not-in' A R
        # A -> 'max' V | 'min' V | 'count' V | 'sum' V | 'avg' V | 'none' V
        # V -> 'none' X | '-' X X | '+' X X | '*' X X | '/' X X
        # X -> 'max' C T | 'min' C T | 'count' C T | 'sum' C T | 'none' C T
        # C -> column
        # T -> t
        # """)
        #  all symbols are non_terminals for convenience
        self.grammar = CFG.fromstring("""
        Z -> intersect R R | union R R | except R R | R
        R -> Select | Select Filter | Select Order | Select Filter Order | distinct Select | distinct Select Filter | distinct Select Order | distinct Select Filter Order
        Select -> A | A A | A A A | A A A A | A A A A A
        Order -> asc V | desc V
        Filter -> and Filter Filter | or Filter Filter | mt V Y | mt V R | lt V Y | lt V R | mte V Y | mte V R | lte V Y | lte V R | e V Y | e V R | ne V Y | ne V R | between V Y Y | between V R | like V Y | not-like V Y | in V R | not-in V R | not V R | not V Y | is V R | is V Y | exists V R | exists V Y | not-exists V R | not-exists V Y | mt V X | lt V X | mte V X | lte V X | e V X |  ne V X |  between V Y Y |  like V X | not-like V X | not V X | is V X | exists V X | not-exists V X
        A -> max V | min V | count V | sum V | avg V | none V
        V -> none X | sub X X | add X X | mul X X | div X X
        X -> max C T | min C T | count C T | sum C T | avg V | none C T 
        C -> column
        T -> t
        Y -> val
        """)
        self.non_terminals = nonterminals(
            "Z R Select Filter Order A V X Y C T")
        self.terminals = nonterminals(
            "intersect union except asc desc most least and or > < = ≠ ≥ between like not-like in not-in max min count sum avg none")

        # Construct a dictionary {index:rule}
        self.rule_set = {index: rule for index, rule in enumerate(self.grammar.productions())}

        # Construct a dictionary {index: ParentedTree('Z', ['R', 'R'])}
        self.rule_set_tree = {index: ParentedTree(str(rule.lhs()), list(rule.rhs())) for index, rule in
                              enumerate(self.grammar.productions())}

        self.rule_dict = self.get_rule_ids()

    def rules_data(self):
        return self.grammar, self.rule_set, self.rule_set_tree, self.non_terminals, self.terminals

    def get_rule_by_nonterminals(self, lhs, rhs=None):
        if isinstance(lhs, str):
            lhs = Nonterminal(lhs)
        if isinstance(rhs, str) and rhs:
            rhs = Nonterminal(rhs)
            # for production in grammar.productions(lhs=lhs_nonterminal,rhs_nonterminal)
            productions = self.grammar.productions(lhs=lhs, rhs=rhs)
        productions = self.grammar.productions(lhs=lhs)
        return {index: rule for index, rule in self.rule_dict.items() if rule in productions}

    def get_rule_by_rhs(self, rhs):
        """Filter rule set by right hand side

        Args:
            rhs:[string]. a string that contain symbol is seperated by a space

        Returns:
            [list]: a list of filtered productions
        """

        rhs_ls = rhs.split(' ')
        rhs_t = [Nonterminal(s) for s in rhs_ls]

        productions = []
        for production in list(self.grammar.productions()):
            is_match = False
            for t in rhs_t:
                is_match = True if rhs_t in production.rhs() else False
            if is_match:
                productions.append(production)
        return productions
    def get_eos_id(self):
        r = {k: v for k, v in self.rule_dict.items() if v == 'EOS'}
        return r

    def get_pad_id(self):
        r = {k: v for k, v in self.rule_dict.items() if v == 'PAD'}
        return r
    def get_rule_by_rhs_lhs(self, rhs, lhs):
        """Filter rule set by both left hand side and right hand side

        Args:
            rhs: [string]. a string that contain symbol separated by a space
            lhs: [string]. a string that contains a single nonterminal

        Returns:
            [list]: a list of filtered productions
        """
        rhs_ls = rhs.split(' ')
        lhs_ls = lhs.split(' ')
        lhs_t = Nonterminal(lhs_ls[0])
        rhs_t = [Nonterminal(s) for s in rhs_ls]

        productions = []
        for production in list(self.grammar.productions()):
            is_match = True
            for t in rhs_t:
                is_match = True if t in production.rhs() and is_match else False

            is_lhs_match = True if lhs_t == production.lhs() else False
            if is_match and is_lhs_match:
                productions.append(production)
        return productions
        # if isinstance(rhs, tuple):
        #     grammar_new = self.grammar
        #     for t in rhs:
        #         if isinstance(t, str):
        #             productions = grammar_new.productions(lhs=lhs, rhs=t)
        #         elif isinstance(t, Nonterminal):
        #             productions = grammar_new.productions(lhs=lhs, rhs=t)
        #         grammar_new = CFG(start=productions[0].lhs(), productions=productions)
        #     return productions
        # else:
        #     productions = self.grammar.productions(lhs=lhs, rhs=rhs)
        #     return productions

    def get_rule_by_index(self, index):
        return self.rule_dict[index]

    def get_table_rule_id(self):
        for k, v in self.rule_dict.items():
            if isinstance(v, Production):
                if v.lhs() == Nonterminal('T'):
                    return k
    def get_column_rule_id(self):
        for k, v in self.rule_dict.items():
            if isinstance(v, Production):
                if v.lhs() == Nonterminal('C'):
                    return k

    def get_value_rule_id(self):
        for k, v in self.rule_dict.items():
            if isinstance(v, Production):
                if v.lhs() == Nonterminal('Y'):
                    return k
    def nonterminals_to_id(self, nonterminal):
        return self.non_terminals.index(nonterminal)
    def get_left_most_noon_terminal(self, ls):

        for t in ls:
            if t in self.non_terminals:
                return  t

    def get_non_recursive_filter(self):

        productions = self.grammar.productions(lhs=Nonterminal('Filter'))
        non_recursive_rule = [p for p in productions if Nonterminal('Filter') not in p.rhs()]
        return {index: rule for index, rule in self.rule_dict.items() if rule in non_recursive_rule}
    def get_non_recursive_r(self):

        productions = self.grammar.productions(lhs=Nonterminal('Filter'))
        non_recursive_rule = [p for p in productions if Nonterminal('R') not in p.rhs()]
        return {index: rule for index, rule in self.rule_dict.items() if rule in non_recursive_rule}

    def determine_type_of_node(self, rule_id):
        rule_dict = self.get_rule_ids()
        rule = rule_dict[rule_id]
        if rule is None:
            return RULE_TYPE
        elif rule == 'PAD':
            return RULE_TYPE
        elif rule == 'EOS':
            return RULE_TYPE
        elif rule_id == EOS_id:
            return RULE_TYPE
        elif rule.lhs() == Nonterminal('C'):
            return COL_TYPE
        elif rule.lhs() == Nonterminal('T'):
            return TAB_TYPE
        elif rule.lhs() == Nonterminal('Y'):
            return VAL_TYPE
        else:
            return RULE_TYPE

    def get_rule_ids(self):
        productions = self.grammar.productions()

        rule_dict = {index: production for index, production in enumerate(productions, 3)}
        rule_dict[SOS_id] = None
        rule_dict[EOS_id] = 'EOS'
        rule_dict[PAD_id] = 'PAD'
        keys = sorted(rule_dict.keys())
        rule_dict = {key: rule_dict[key] for key in keys}
        return rule_dict

    def dump_rule_dict_to_json(self, file_path):
        with open(file_path, 'w') as f:
            rule_dict = self.get_rule_ids()
            rule_dict = {k: str(v) for k, v in rule_dict.items()}
            json.dump(rule_dict, f, indent=4)
            return 0

    def read_rules_from_json(self, file_path):
        with open(file_path, 'r') as f:
            rule_dict = json.load(f)
        rule_dict = {int(k): CFG.fromstring(v).productions()[0] for k, v in rule_dict.items() if v != 'None' and v != 'EOS' and v != 'PAD'}
        rule_dict[0] = 'PAD'
        rule_dict[1] = None
        rule_dict[2] = 'EOS'
        # sort rule_dict by keys
        keys = sorted(rule_dict.keys())
        rule_dict = {key: rule_dict[key] for key in keys}
        return rule_dict

#
# #%%
# r = RuleGenerator()
# # prdcts = r.grammar.productions()
# # r.dump_rule_dict_to_json('../data/rule.json')
# r_dict = r.read_rules_from_json('../data/rule.json')

# print(r_dict)
