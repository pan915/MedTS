import traceback
import args as arg
import utils
import sys
from modeling_treqs import *
import time
import torch
from tqdm import tqdm
from torch import optim
from optimizer import build_optimizer_encoder
import pickle
from tree import *
from torch.autograd import gradcheck
cur_dir = os.getcwd()
path = os.path.dirname(cur_dir)
sys.path.append(path)


def train(args):
    """
    :param args:
    :return:
    """
    # bid, uid
    # dirty_data = [[0, 4], [2, 3], [3,5], [11, 0]]
    dirty_data = []
    device = torch.device('cuda', args.cuda_device_num) if args.cuda else torch.device('cpu')
    b = load_data(args.train_data, device)
    # b_test = load_data(args.dev_data, device)
    model = NL2SQLTransformer(args)
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)
    model.to(device)
    # now get the optimizer
    # optimizer_cls = eval('torch.optim.%s' % args.optimizer)
    # optimizer = optimizer_cls(model.group_output.parameters(), lr=args.lr_transformer)
    optimizer, scheduler = build_optimizer_encoder(model,
                                                   args.lr_transformer, args.lr_connection, args.lr_base, args.lr_copy,
                                                   args.scheduler_gamma)
    print('Enable Learning Rate Scheduler: ', args.lr_scheduler)
    # if args.lr_scheduler:
    #     scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[21, 41], gamma=args.lr_scheduler_gammar)
    # else:
    #     scheduler = None

    print('Loss epoch threshold: %d' % args.loss_epoch_threshold)
    # print('Sketch loss coefficient: %f' % args.sketch_loss_coefficient)

    if args.load_model:
        print('load pretrained model from %s'% (args.load_model))
        pretrained_model = torch.load(args.load_model,
                                      map_location=lambda storage, loc: storage)
        pretrained_modeled = copy.deepcopy(pretrained_model)
        for k in pretrained_model.keys():
            if k not in model.state_dict().keys():
                del pretrained_modeled[k]

        model.load_state_dict(pretrained_modeled, strict=False)

    # model.word_emb = load_word_emb(args.glove_embed_path)
    # if not args.use_bert:
    #     with open(args.glove_embed_path, 'rb') as wf:
    #         model.word_emb = pickle.load(wf)
    # begin train
    model_save_path = utils.init_log_checkpoint_path(args)
    utils.save_args(args, os.path.join(model_save_path, 'config.json'))
    # best_dev_acc = .0

    try:
        with open(os.path.join(model_save_path, 'epoch.log'), 'w') as epoch_fd:
            model.train()
            for epoch in tqdm(range(args.epoch)):
                # if args.lr_scheduler:
                #     scheduler.step()
                epoch_begin = time.time()

                epoch_total_loss = 0
                for index, one_batch in enumerate(b.batches):
                    # optimizer.zero_grad()
                    t_loss = model(one_batch, device, optimizer, scheduler, index, dirty_data)
                    if torch.isnan(t_loss):
                        print("dirty bid:", index)
                        sys.exit(0)
                    # if epoch > args.loss_epoch_threshold:
                    #     loss = action_loss + args.sketch_loss_coefficient * sketch_loss
                    # else:
                    #     loss = action_loss + sketch_loss
                    print(t_loss)
                    if args.batch_loss:
                        if t_loss != 0:
                            t_loss.backward()
                            if args.clip_grad > 0.:
                                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                            optimizer.step()
                            # if args.lr_scheduler:
                            #     scheduler.step()
                    if t_loss!=0:
                        epoch_total_loss += t_loss.item()

                if args.lr_scheduler:
                    scheduler.step()
                epoch_avg_loss = epoch_total_loss/len(b.batches)
                epoch_end = time.time()

                utils.save_checkpoint(model, os.path.join(model_save_path, '{%s}_{%s}.model') % (epoch, epoch_avg_loss))

                log_str = 'Epoch: %d, Loss: %f,  time: %f\n' % (
                    epoch + 1, epoch_avg_loss,  epoch_end - epoch_begin)
                tqdm.write(log_str)
                epoch_fd.write(log_str)
                epoch_fd.flush()
    except Exception as e:
        # Save model
        utils.save_checkpoint(model, os.path.join(model_save_path, 'end_model.model'))
        print(e)
        tb = traceback.format_exc()
        print(tb)
    else:
        utils.save_checkpoint(model, os.path.join(model_save_path, 'end_model.model'))


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




def eval_acc(preds, sqls):
    sketch_correct, best_correct = 0, 0
    for i, (pred, sql) in enumerate(zip(preds, sqls)):
        if pred['model_result'] == sql['rule_label']:
            best_correct += 1
    print(best_correct / len(preds))
    return best_correct / len(preds)


if __name__ == '__main__':
    print('update')
    arg_parser = arg.init_arg_parser()
    args = arg.init_config(arg_parser)
    args = arg_parser.parse_args()
    # args.load_model = './saved_model/1585737057/end_model.model'
    print(args)
    train(args)
