from nltk.grammar import Nonterminal
from nltk.tree import ParentedTree

class TreeWithPara(ParentedTree):

    pos_in_nt = None
    eos = None
    sos = None

    def __init__(self, label, sub_tree=None, visited=False, is_curnode=False, rule_id=None, parent_action=None, pre_action=None,
                 target_action=None, is_all_vis=False):
        super(TreeWithPara, self).__init__(label, sub_tree)
        # self.layer_feature = layer_feature
        self.visited = visited
        self.rule_id = rule_id
        self.is_cur_node = is_curnode
        self.parent_action = parent_action
        self.pre_action = pre_action
        self.target_action = target_action
        self.is_all_vis = is_all_vis

    def add_child(self, child):
        self.append(child)

    def set_rule_id(self, rule_id):
        self.rule_id = rule_id
        self.is_all_visited()

    # def get_parent(self):
    #     return self.parent()

    def unvisited_child(self):
        return list(self.subtrees(lambda t: t.visited is False))

    def get_cur_node(self):
        return list(self.subtrees(lambda t: t.is_cur_node is True))

    def left_most_child_unvisited(self):
        unvisited = filter(lambda x: isinstance(x, TreeWithPara) and  len(x)==0, list(self))
        # print(list(unvisited))
        return list(unvisited)[0]

    def left_most_child_unvisited_b(self):
        unvisited = self.unvisited_child()
        return unvisited[0]

    def is_terminal(self, non_terminal_set):
        if Nonterminal(self.label()) in non_terminal_set:
            return False
        else:
            return True

    def is_all_visited(self):
        # cannot used on parent()
        for i in list(self.subtrees()):
            if len(i) == 0:
                return False
        return True
        # if len(self.unvisited_child()) == 0:
        #     self.is_all_vis = True
        #     return True
        # self.is_all_vis = False
        # return False
    def is_all_visited_b(self):

        if len(self.unvisited_child()) == 0:
            # self.is_all_vis = True
            return True
        # self.is_all_vis = False
        return False




    def set_tree_to_default(self):
        for s in self.subtrees():
            s.visited = False

    @classmethod
    def next_unvisited(cls, cur_ast, cur_node):
        while get_parent(cur_ast, cur_node) is not None:
            # print(get_parent(cur_ast, cur_node))
            if get_parent(cur_ast, cur_node).is_all_visited():
                cur_node = get_parent(cur_ast, cur_node)
                # print('sdfddddddd')
            else:
                p = cur_node.parent()
                # print('parent', get_parent(cur_ast, cur_node))
                cur_node = get_parent(cur_ast, cur_node).left_most_child_unvisited()
                # print('asdfadsf')
                break
        return cur_node

    @classmethod
    def next_unvisited_b(cls, cur_node):
        while cur_node.parent() is not None:
            # print(cur_node.label())
            if cur_node.parent().is_all_visited_b():
                cur_node = cur_node.parent()
            else:
                cur_node = cur_node.parent().left_most_child_unvisited_b()
                break
        return cur_node


def get_cur_node(ast: TreeWithPara):
    for a in list(ast.subtrees()):
        if len(a) == 0:
            return a


def get_parent(ast, curnode):
    #
    # if curnode.parent() is None:
    #     return None
    # else:
    #     # print(list(ast.subtrees()))
    #     # print(curnode)
    #     p = list(ast.subtrees())[
    #         list(ast.subtrees()).index(curnode.parent())]
    #     return p
    return curnode.parent()
