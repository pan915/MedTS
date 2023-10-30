import tqdm
import args as arg
from parser_treqs import FromAstToSql
from modeling_treqs import *
from run_sparc_dataset_utils import *
from evaluation3 import *
import torch
import json
import os
from tqdm import tqdm
import pickle


def eval(args):
    if args.no_sl:
        suffix = '_no_sl'
    else:
        suffix = ''
    device = torch.device('cuda', args.cuda_device_num) if args.cuda else torch.device('cpu')
    b_eval = load_data(args.eval_data, device)
    list = os.listdir(os.path.join('.', args.model_dir))  # 列出文件夹下所有的目录与文件
    model = NL2SQLTransformer(args)
    model.to(device)
    for i in range(0, len(list)):
        if args.model and list[i] != args.model:
            continue
        path = os.path.join(args.model_dir, list[i])
        try:
            if os.path.isfile(path):
                print('load pretrained model from %s' % (path))
                pretrained_model = torch.load(path, map_location=lambda storage, loc: storage)
                model.load_state_dict(pretrained_model, strict=False)

                ast_total_eval = []
                for e in tqdm(b_eval.batches):
                    with torch.no_grad():
                        ast, loss = model.predict(e, device)
                        ast_total_eval.append(ast)

                schemas = load_schema(args)
                eval_output = infer_sql(schemas, ast_total_eval, args, list[i], b_eval, 'eval', suffix)
                print("The predicted results are saved in %s" % eval_output)

        except Exception as e:
            print(list[i])
            print(e)

def infer_sql(schemas, asts, args, model_name, b, mode, suffix):
    results = []
    lines = []

    result_path = args.output_dir

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    file_path = os.path.join(result_path, model_name.split("_")[0] + suffix + '.json')
    with open(file_path, 'w', encoding='utf8') as treqs:
        for i, batch_ast in enumerate(asts):
            e = b.batches[i]
            for batch_id, a in enumerate(batch_ast):
                for turn_id, utter_a in enumerate(a):
                    if utter_a is None:
                        pred_s = 'AST sequence is too long'
                        results.append(pred_s)
                        print("ast is too long")
                    else:
                        # print(len(utter_a))
                        # utter_a.pretty_print(maxwidth=10)
                        parser = FromAstToSql(utter_a, e.col_set[batch_id], e.origin_table_names[batch_id],
                                              e.col_table_dict[batch_id], args, schemas[e.db_ids[batch_id]],
                                              e.interactions[batch_id][turn_id].src_sent_origin,
                                              e.column_names[batch_id],
                                              e.origin_column_names[batch_id], e.origin_table_names[batch_id])
                        # try:
                        pred_s = parser.parse()
                        flattened_str = parser.flatten(pred_s)
                        result = [str(s) for s in flattened_str]
                        res_str = ' '.join(result)
                        res_str = res_str.lower()
                        question_toks = []
                        for i in e.interactions[batch_id][turn_id].src_sent_origin:
                            question_toks.extend(i)
                        gold = e.interactions[batch_id][turn_id].sql.lower()
                        new_gold = ''
                        for idx, gs in enumerate(gold):
                            if gs == '(' and gold[idx + 1] != ' ':
                                new_gold += '( '
                            elif gs == ')' and gold[idx - 1] != ' ':
                                new_gold += ' )'
                            else:
                                new_gold += gs

                        res_str = rocover_pred(res_str, ' '.join(question_toks))

                        line_dict = {"sql_gold": new_gold, "sql_pred": res_str + "<stop>"}

                        line = json.dumps(line_dict)

                        lines.append(line)
                        # treqs.write(line)

        treqs.writelines([l + '\n' for l in lines])
    return file_path


def rocover_pred(res_str, question):
    agg = ['avg', 'sum', 'max', 'min']
    agg_expr = ['average', 'sum', 'max', 'min']
    ops = {'>': ['more', 'above', 'after', 'older'], '<': ['less', 'under', 'before', 'younger']}
    # res_str = res_str.replace('lab.subject_id', 'lab.hadm_id')
    res_str = res_str.replace('female', 'f')
    res_str = res_str.replace('male', 'm')

    res_str = res_str.replace(' , ', ',')
    res_str = res_str.replace('unmarried', 'single')
    split_str = res_str.split(' ')
    if split_str[1] in agg:
        for idx, i in enumerate(agg_expr):
            if i in question.lower():
                split_str[1] = agg[idx]
                break
    res_str = ' '.join(split_str)

    res_str = res_str.replace('spanish', 'span').replace('russian', 'russ')
    return res_str


def load_schema(args):
    with open(os.path.join(args.dataset, 'tables.json'), 'r', encoding='utf8') as f:
        table_datas = json.load(f)
    schemas = dict()
    for i in range(len(table_datas)):
        schemas[table_datas[i]['db_id']] = table_datas[i]
    return schemas


def load_data(data, device):
    with open(data, 'rb') as f:
        b = pickle.load(f)
    for e in b.batches:

        for i in e.interactions:
            for u in i:
                u.tgt_actions = u.tgt_actions.to(device)
                u.pre_actions = u.pre_actions.to(device)
                u.parent_actions = u.parent_actions.to(device)
                u.masked_target_actions = u.masked_target_actions.to(device)
                u.masked_pre_actions = u.masked_pre_actions.to(device)
                u.masked_parent_actions = u.masked_parent_actions.to(device)
                u.src_sent_mask = u.src_sent_mask.to(device)
        e.col_set_mask = e.col_set_mask.to(device)
        e.table_mask = e.table_mask.to(device)
    return b


if __name__ == '__main__':
    arg_parser = arg.init_arg_parser()
    arg_parser.add_argument('--model_dir', default='./data', type=str)
    arg_parser.add_argument('--model', default='./data', type=str)
    arg_parser.add_argument('--eval_data', default='./data/dev.pkl', type=str)
    arg_parser.add_argument('--output_dir', default='./data', type=str)
    args = arg.init_config(arg_parser)
    device = torch.device('cuda', args.cuda_device_num) if args.cuda else torch.device('cpu')
    print(args)
    eval(args)