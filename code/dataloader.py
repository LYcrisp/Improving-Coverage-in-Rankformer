import torch
from torch.utils.data import Dataset
import pandas as pd
from parse import args
import numpy as np
import torch.nn.functional as F


class MyDataset(Dataset):
    def __init__(self, train_file, valid_file, test_file, device):
        self.device = device  # 设置设备（CPU 或 GPU）

        # 加载 train/valid/test 三份交互数据
        train_data = pd.read_table(train_file, header=None, sep=' ')
        train_data = torch.from_numpy(train_data.values).to(self.device)
        self.train_data = train_data[torch.argsort(train_data[:, 0]), :]
        self.train_user, self.train_item = self.train_data[:, 0], self.train_data[:, 1]

        valid_data = pd.read_table(valid_file, header=None, sep=' ')
        valid_data = torch.from_numpy(valid_data.values).to(self.device)
        self.valid_data = valid_data[torch.argsort(valid_data[:, 0]), :]
        self.valid_user, self.valid_item = self.valid_data[:, 0], self.valid_data[:, 1]

        test_data = pd.read_table(test_file, header=None, sep=' ')
        test_data = torch.from_numpy(test_data.values).to(self.device)
        self.test_data = test_data[torch.argsort(test_data[:, 0]), :]
        self.test_user, self.test_item = self.test_data[:, 0], self.test_data[:, 1]

        # 统计用户数、物品数
        self.num_users = int(max(self.train_user.max(),
                                 self.valid_user.max(),
                                 self.test_user.max()).cpu()) + 1
        self.num_items = int(max(self.train_item.max(),
                                 self.valid_item.max(),
                                 self.test_item.max()).cpu()) + 1
        self.num_nodes = self.num_users + self.num_items

        print(f'{self.num_users} users, {self.num_items} items.')
        print(f'train: {len(self.train_user)}, valid: {len(self.valid_user)}, test: {len(self.test_user)}.')

        # 统计每个物品在训练集中的交互频次，作为 item_freq
        item_inter_counts = torch.zeros(self.num_items, device=self.device) \
                              .index_add(0, self.train_item,
                                         torch.ones_like(self.train_item, dtype=torch.float))
        self.item_freq = item_inter_counts  # [num_items]，后面可以直接用

        # 根据频次分位数标记长尾物品：底部 20% 作为长尾
        freq_np = item_inter_counts.cpu().numpy()
        threshold = np.percentile(freq_np[freq_np > 0], 20)
        is_tail = (freq_np <= threshold).astype(int)
        self.is_tail = torch.from_numpy(is_tail).to(self.device)       # 0/1 标记
        self.tail_items = set(np.where(is_tail == 1)[0].tolist())     # 长尾 item 索引集合

        # 构建批处理索引
        self.build_batch()

        # 统计并打印一些额外信息：用户平均交互数 & 物品平均被交互数
        user_inter_counts = torch.zeros(self.num_users, device=self.device) \
                               .index_add(0, self.train_user,
                                          torch.ones_like(self.train_user, dtype=torch.float))
        avg_items_per_user = user_inter_counts.mean().item()
        avg_users_per_item = item_inter_counts.mean().item()
        print(f'Average number of items per user (train): {avg_items_per_user:.2f}')
        print(f'Average number of users per item (train): {avg_users_per_item:.2f}')

    def build_batch(self):
        # 计算每个用户的训练交互数（即每个用户的“度”）
        self.train_degree = torch.zeros(self.num_users).long().to(args.device).index_add(
            0, self.train_user, torch.ones_like(self.train_user)
        )
        self.test_degree = torch.zeros(self.num_users).long().to(args.device).index_add(
            0, self.test_user, torch.ones_like(self.test_user)
        )
        self.valid_degree = torch.zeros(self.num_users).long().to(args.device).index_add(
            0, self.valid_user, torch.ones_like(self.valid_user)
        )

        # 构造批量用户索引列表，每批最多 args.test_batch_size 个用户
        self.batch_users = [
            torch.arange(i, min(i + args.test_batch_size, self.num_users)).to(args.device)
            for i in range(0, self.num_users, args.test_batch_size)
        ]

        # 将三类交互数据按用户 degree 划分为多个 batch
        self.train_batch = list(self.train_data.split(
            [self.train_degree[batch_user].sum() for batch_user in self.batch_users]
        ))
        self.test_batch = list(self.test_data.split(
            [self.test_degree[batch_user].sum() for batch_user in self.batch_users]
        ))
        self.valid_batch = list(self.valid_data.split(
            [self.valid_degree[batch_user].sum() for batch_user in self.batch_users]
        ))

dataset = MyDataset(args.train_file, args.valid_file, args.test_file, args.device)
