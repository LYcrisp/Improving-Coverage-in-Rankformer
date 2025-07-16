from parse import args
from dataloader import dataset
from model import Model
import torch
import numpy as np


def print_test_result():
    global best_epoch
    global test_pre, test_recall, test_ndcg
    # 新增这三个
    global test_cov, test_tail_recall, test_tail_cov

    print(f'===== Test Result(at {best_epoch:d} epoch) =====')
    for i, k in enumerate(args.topks):
        print(f'ndcg@{k:d} = {test_ndcg[i]:f}, '
              f'recall@{k:d} = {test_recall[i]:f}, '
              f'pre@{k:d} = {test_pre[i]:f}')
    # 新指标
    print(f'Coverage         = {test_cov:.4f}')
    print(f'Tail-Recall@{args.topks[-1]:d} = {test_tail_recall:.4f}')
    print(f'Tail-Coverage    = {test_tail_cov:.4f}')


def train():
    train_loss = model.train_func()
    if epoch % args.show_loss_interval == 0:
        print(f'epoch {epoch:d}, train_loss = {train_loss:f}.')


def valid(epoch):
    global best_valid_ndcg, best_epoch
    global test_pre, test_recall, test_ndcg
    # 同样新增这三个
    global test_cov, test_tail_recall, test_tail_cov

    # unpack 多了三个返回值
    valid_pre, valid_recall, valid_ndcg, valid_cov, valid_tail_recall, valid_tail_cov = model.valid_func()

    # 原有的 precision/recall/ndcg 打印
    for i, k in enumerate(args.topks):
        print(f'[{epoch:d}/{args.max_epochs:d}] Valid Result: '
              f'ndcg@{k:d} = {valid_ndcg[i]:f}, '
              f'recall@{k:d} = {valid_recall[i]:f}, '
              f'pre@{k:d} = {valid_pre[i]:f}.')

    # 打印验证集上的新指标
    print(f'Coverage         = {valid_cov:.4f}')
    print(f'Tail-Recall@{args.topks[-1]:d} = {valid_tail_recall:.4f}')
    print(f'Tail-Coverage    = {valid_tail_cov:.4f}')

    # 如果验证集 NDCG 有所提升，则保存对应的测试指标
    if valid_ndcg[-1] > best_valid_ndcg:
        best_valid_ndcg, best_epoch = valid_ndcg[-1], epoch
        # 同样 unpack 三个新指标
        test_pre, test_recall, test_ndcg, test_cov, test_tail_recall, test_tail_cov = model.test_func()
        print_test_result()
        if args.save_emb:
            model.save_emb()
        return True

    return False


model = Model(dataset).to(args.device)
if args.load_emb:
    model.load_emb()

best_valid_ndcg, best_epoch = 0., 0
test_pre, test_recall, test_ndcg = torch.zeros(len(args.topks)), torch.zeros(len(args.topks)), torch.zeros(len(args.topks))
valid(epoch=0)
for epoch in range(1, args.max_epochs+1):
    train()
    if epoch % args.valid_interval == 0:
        if not valid(epoch) and epoch-best_epoch >= args.stopping_step*args.valid_interval:
            break
print('---------------------------')
print('done.')
print_test_result()
