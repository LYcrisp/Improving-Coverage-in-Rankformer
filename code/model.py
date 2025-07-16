from parse import args
import torch
from torch import nn
import torch.nn.functional as F
import rec
from rec import GCN, Rankformer
import numpy as np


def InfoNCE(x, y, tau=0.15, b_cos=True):
    """
    计算对比学习损失
    """
    if b_cos:
        x, y = F.normalize(x), F.normalize(y)
    return -torch.diag(F.log_softmax((x@y.T)/tau, dim=1)).mean()


def test(pred, test, recall_n):
    """
    评估预测结果的 Precision/Recall/NDCG@k
    """
    pred = torch.isin(pred[recall_n > 0], test)
    recall_n = recall_n[recall_n > 0]
    pre, recall, ndcg = [], [], []
    for k in args.topks:
        right_pred = pred[:, :k].sum(1)
        recall_k = recall_n.clamp(max=k)
        # precision
        pre.append((right_pred/k).sum())
        # recall
        recall.append((right_pred/recall_k).sum())
        # ndcg
        dcg = (pred[:, :k]/torch.arange(2, k+2).to(args.device).unsqueeze(0).log2()).sum(1)
        d_val = (1/torch.arange(2, k+2).to(args.device).log2()).cumsum(0)
        idcg = d_val[recall_k-1]
        ndcg.append((dcg / idcg).sum())
    return recall_n.shape[0], torch.tensor(pre), torch.tensor(recall), torch.tensor(ndcg)


def multi_negative_sampling(u, i, m, k):
    """
    返回每个正样本配对的 k 个负样本
    """
    edge_id = u*m+i
    j = torch.randint(0, m, (i.shape[0], k)).to(u.device)
    mask = torch.isin(u.unsqueeze(1)*m+j, edge_id)
    while mask.sum() > 0:
        j[mask] = torch.randint_like(j[mask], 0, m)
        mask = torch.isin(u.unsqueeze(1)*m+j, edge_id)
    return j


def negative_sampling(u, i, m):
    """
    返回一个负样本
    """
    edge_id = u*m+i
    j = torch.randint_like(i, 0, m)
    mask = torch.isin(u*m+j, edge_id)
    while mask.sum() > 0:
        j[mask] = torch.randint_like(j[mask], 0, m)
        mask = torch.isin(u*m+j, edge_id)
    return j


class Model(nn.Module):
    def __init__(self, dataset):
        super(Model, self).__init__()
        self.dataset = dataset  # 数据集对象，包含用户数、物品数、交互图等

        # 嵌入维度，从外部参数 args 中获取
        self.hidden_dim = args.hidden_dim

        # 初始化用户和物品嵌入，每个嵌入向量长度为 hidden_dim
        self.embedding_user = nn.Embedding(self.dataset.num_users, self.hidden_dim)
        self.embedding_item = nn.Embedding(self.dataset.num_items, self.hidden_dim)

        # 使用正态分布初始化嵌入参数，标准差为 0.1
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

        # 将用户和物品嵌入参数打包为参数列表，用于优化器
        self.my_parameters = [
            {'params': self.embedding_user.parameters()},
            {'params': self.embedding_item.parameters()},
        ]

        # 用于存储后续的 GCN 层或其他网络模块（可选，当前为空）
        self.layers = []

        # 初始化图卷积模块，融合用户-物品图的结构信息
        # 参数 gcn_left 和 gcn_right 控制两边图的聚合权重
        self.GCN = GCN(dataset, args.gcn_left, args.gcn_right)

        # 初始化基于 Transformer 的排序增强模块
        # 参数 rankformer_alpha 控制排序损失的权重或影响力
        self.Rankformer = Rankformer(dataset, args.rankformer_alpha)

        # 用于缓存正负样本用户/物品嵌入，常用于对比学习或损失计算
        self._users, self._items, self._users_cl, self._items_cl = None, None, None, None

        # 使用 Adam 优化器训练嵌入参数
        self.optimizer = torch.optim.Adam(
            self.my_parameters,
            lr=args.learning_rate)  # 学习率从外部 args 获取

        # 指定 BPR（Bayesian Personalized Ranking）损失函数作为训练目标
        self.loss_func = self.loss_bpr

    def computer(self):
        """
        生成用户和物品的最终表示（embedding），包含 GCN 聚合、对比学习扰动（可选），
        以及 Rankformer 排序增强模块（可选），并分别保存正向表示和对比学习表示。
        """
        # 从训练数据中获取所有的用户和物品交互边
        u, i = self.dataset.train_user, self.dataset.train_item

        # 获取当前用户和物品嵌入权重（初始化或训练中更新的参数）
        users_emb, items_emb = self.embedding_user.weight, self.embedding_item.weight

        # 将用户和物品嵌入拼接成一个整体的嵌入矩阵，shape: [num_users + num_items, hidden_dim]
        all_emb = torch.cat([users_emb, items_emb])

        # emb_cl 用于对比学习中保存的中间层嵌入（默认为原始嵌入）
        emb_cl = all_emb

        # 若使用图卷积（Graph Convolution Network）模块
        if args.use_gcn:
            embs = [all_emb]  # 保存每一层的嵌入（用于平均）
            for _ in range(args.gcn_layers):
                # 执行一次图卷积传播（聚合邻居特征）
                all_emb = self.GCN(all_emb, u, i)

                # 若开启对比学习模块，对嵌入施加扰动以生成正样本的不同视图
                if args.use_cl:
                    random_noise = torch.rand_like(all_emb)  # 生成同形状的随机噪声
                    # 使用归一化的扰动乘以 eps 系数并加权到原始表示上（sign 处理方向）
                    all_emb += torch.sign(all_emb) * F.normalize(random_noise, dim=-1) * args.cl_eps

                # 如果当前层是设置的用于对比学习的目标层（如第 cl_layer 层），则保存此时的表示为对比学习表示
                if _ == args.cl_layer - 1:
                    emb_cl = all_emb

                # 保存当前层的 all_emb（用于最后平均）
                embs.append(all_emb)

            # 若设置使用多层 GCN 嵌入的平均作为最终输出
            if args.gcn_mean:
                # 对所有层嵌入堆叠后求平均，shape 保持为 [num_users + num_items, hidden_dim]
                all_emb = torch.stack(embs, dim=-1).mean(-1)

        # 若使用 Rankformer（Transformer-based 排序增强模块）
        if args.use_rankformer:
            for _ in range(args.rankformer_layers):
                # Rankformer 对当前嵌入执行排序引导的表示学习，返回增强后的 embedding
                rec_emb = self.Rankformer(all_emb, u, i)

                # 使用 tau 系数将 Rankformer 输出与原始 embedding 进行线性融合
                all_emb = (1 - args.rankformer_tau) * all_emb + args.rankformer_tau * rec_emb

        # 最终将所有用户和物品嵌入从总嵌入矩阵中按顺序切分出来（前面是用户，后面是物品）
        self._users, self._items = torch.split(all_emb, [self.dataset.num_users, self.dataset.num_items])

        # 同样地，将对比学习用的嵌入切分出来，用于 CL 损失计算
        self._users_cl, self._items_cl = torch.split(emb_cl, [self.dataset.num_users, self.dataset.num_items])

    def evaluate(self, test_batch, test_degree):
        """
        如果 --use_coverage 未打开，返回 (pre, recall, ndcg)；
        否则额外返回 (coverage, tail_recall_at_K, tail_coverage)。
        """
        self.eval()
        if self._users is None:
            self.computer()

        user_emb, item_emb = self._users, self._items
        max_K = max(args.topks)
        all_pre = torch.zeros(len(args.topks))
        all_recall = torch.zeros(len(args.topks))
        all_ndcg = torch.zeros(len(args.topks))
        all_cnt = 0

        if not args.use_coverage:
            with torch.no_grad():
                for batch_users, batch_train, ground_true in zip(
                        self.dataset.batch_users,
                        self.dataset.train_batch,
                        test_batch):
                    u_e = user_emb[batch_users]
                    rating = torch.mm(u_e, item_emb.t())
                    rating[batch_train[:, 0] - batch_users[0],
                    batch_train[:, 1]] = -1e9
                    _, pred_items = torch.topk(rating, k=max_K, dim=1)

                    cnt, pre, recall, ndcg = test(
                        batch_users.unsqueeze(1) * self.dataset.num_items + pred_items,
                        ground_true[:, 0] * self.dataset.num_items + ground_true[:, 1],
                        test_degree[batch_users]
                    )
                    all_pre += pre
                    all_recall += recall
                    all_ndcg += ndcg
                    all_cnt += cnt

            all_pre /= all_cnt
            all_recall /= all_cnt
            all_ndcg /= all_cnt
            return all_pre, all_recall, all_ndcg, 0.0, 0.0, 0.0

        # ---- 计算新的三个指标 ----
        recommended = set()
        recommended_tail = set()
        total_tail_recall = 0.0
        total_tail_users = 0

        with torch.no_grad():
            for batch_users, batch_train, ground_true in zip(
                    self.dataset.batch_users,
                    self.dataset.train_batch,
                    test_batch):
                u_e = user_emb[batch_users]
                rating = torch.mm(u_e, item_emb.t())
                rating[batch_train[:, 0] - batch_users[0],
                batch_train[:, 1]] = -1e9
                _, pred_items = torch.topk(rating, k=max_K, dim=1)

                # Coverage 统计
                for recs in pred_items.tolist():
                    for item in recs:
                        recommended.add(item)
                        if self.dataset.is_tail[item]:
                            recommended_tail.add(item)

                # Tail-Recall 统计
                for u in batch_users.tolist():
                    true_items = [item for (usr, item) in ground_true.tolist()
                                  if usr == u and self.dataset.is_tail[item]]
                    if not true_items:
                        continue
                    idx = batch_users.tolist().index(u)
                    recs = pred_items[idx].tolist()
                    hits = len(set(recs) & set(true_items))
                    total_tail_recall += hits / len(true_items)
                    total_tail_users += 1

                # 原有指标
                cnt, pre, recall, ndcg = test(
                    batch_users.unsqueeze(1) * self.dataset.num_items + pred_items,
                    ground_true[:, 0] * self.dataset.num_items + ground_true[:, 1],
                    test_degree[batch_users]
                )
                all_pre += pre
                all_recall += recall
                all_ndcg += ndcg
                all_cnt += cnt

        all_pre /= all_cnt
        all_recall /= all_cnt
        all_ndcg /= all_cnt

        # 计算 coverage 和 tail 指标
        coverage = len(recommended) / float(self.dataset.num_items)
        tail_recall_at_K = (total_tail_recall / total_tail_users) if total_tail_users > 0 else 0.0
        tail_coverage = len(recommended_tail) / float(len(self.dataset.tail_items))

        return all_pre, all_recall, all_ndcg, coverage, tail_recall_at_K, tail_coverage

    def valid_func(self):
        return self.evaluate(self.dataset.valid_batch, self.dataset.valid_degree)

    def test_func(self):
        return self.evaluate(self.dataset.test_batch, self.dataset.test_degree)

    def train_func(self):
        self.train()  # 将模型设置为训练模式，启用 Dropout、BatchNorm 等训练行为

        # 若 loss_batch_size 设置为 0，则表示不使用 mini-batch，直接整体训练
        if args.loss_batch_size == 0:
            return self.train_func_one_batch(self.dataset.train_user, self.dataset.train_item)

        # 用于保存每个 mini-batch 的训练损失
        train_losses = []

        # 将训练样本的索引随机打乱，实现数据打乱训练
        shuffled_indices = torch.randperm(self.dataset.train_user.shape[0], device=args.device)

        # 按打乱的索引顺序重新排列训练用户和物品
        train_user = self.dataset.train_user[shuffled_indices]
        train_item = self.dataset.train_item[shuffled_indices]

        # 按照设定的 batch_size 将训练数据划分为多个 mini-batch 进行训练
        for _ in range(0, train_user.shape[0], args.loss_batch_size):
            # 取当前 batch 的用户和物品，并进行一次前向+反向训练
            batch_loss = self.train_func_one_batch(
                train_user[_:_ + args.loss_batch_size],
                train_item[_:_ + args.loss_batch_size]
            )
            train_losses.append(batch_loss)  # 保存该 batch 的训练损失

        # 所有 batch 的损失求平均后作为本轮训练的总损失返回
        return torch.stack(train_losses).mean()

    def train_func_one_batch(self, u, i):
        if args.use_sgl:
            return self.train_func_sgl(u, i)
        else:
            return self.train_func_plain(u, i)

    def train_func_sgl(self, u, i):
        # ---- GCN+BPR ----
        self.computer()  # 执行一次 GCN → Rankformer → ...，结果存在 self._users, self._items
        bpr_loss = self.loss_bpr(u, i)

        # ---- 获取第二套图视图嵌入 ----
        with torch.no_grad():
            orig_user_emb = self.embedding_user.weight
            orig_item_emb = self.embedding_item.weight
        all_emb = torch.cat([orig_user_emb, orig_item_emb], dim=0)
        aug_emb = self.GCN(
            all_emb,
            self.dataset.train_user,
            self.dataset.train_item
        )
        users_aug, items_aug = torch.split(aug_emb, [self.dataset.num_users, self.dataset.num_items])
        users_orig, items_orig = self._users, self._items

        # ---- 计算 GraphCL 对比学习损失 ----
        u_idx = torch.unique(u)
        i_idx = torch.unique(i)
        user_cl_loss = InfoNCE(users_orig[u_idx], users_aug[u_idx], tau=args.cl_tau)
        item_cl_loss = InfoNCE(items_orig[i_idx], items_aug[i_idx], tau=args.cl_tau)
        sgl_loss = (user_cl_loss + item_cl_loss) * args.sgl_lambda

        # ---- 总损失 & 更新 ----
        loss = bpr_loss + sgl_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def train_func_plain(self, u, i):
        self.computer()
        train_loss = self.loss_func(u, i)
        self.optimizer.zero_grad()
        train_loss.backward()
        self.optimizer.step()
        # memory_allocated = torch.cuda.max_memory_allocated(args.device)
        # print(f"Max memory allocated after backward pass: {memory_allocated} bytes = {memory_allocated/1024/1024:.4f} MB = {memory_allocated/1024/1024/1024:.4f} GB.")
        return train_loss


    def loss_bpr(self, u, i):
        """
        BPR + L2 正则 + 可选对比学习损失
        可选长尾物品加权（由 args.use_tail_loss 控制）
        """
        # 负采样
        j = negative_sampling(u, i, self.dataset.num_items)

        # 嵌入检索：原始嵌入（*_emb0）和传播后嵌入（*_emb）
        u_emb0, u_emb = self.embedding_user(u), self._users[u]
        i_emb0, i_emb = self.embedding_item(i), self._items[i]
        j_emb0, j_emb = self.embedding_item(j), self._items[j]

        # 计算每对样本的 BPR 损失
        scores_ui     = torch.sum(u_emb * i_emb, dim=-1)
        scores_uj     = torch.sum(u_emb * j_emb, dim=-1)
        per_pair_loss = F.softplus(scores_uj - scores_ui)

        # 根据 args.use_tail_loss 决定是否做长尾加权
        if args.use_tail_loss:
            # dataset.is_tail[i] 是 0/1 tensor，标记该 item 是否属于长尾
            tail_mask   = self.dataset.is_tail[i].float()
            weight_tail = 1.0 + args.tail_lambda * tail_mask
            bpr_loss    = (weight_tail * per_pair_loss).sum() / weight_tail.sum()
        else:
            bpr_loss = per_pair_loss.mean()

        # L2 正则（只基于原始嵌入）
        reg_loss = 0.5 * (
            u_emb0.norm(2).pow(2) +
            i_emb0.norm(2).pow(2) +
            j_emb0.norm(2).pow(2)
        ) / float(u.shape[0])

        # 汇总基础损失
        loss = bpr_loss + args.reg_lambda * reg_loss

        # 可选对比学习损失
        if args.use_cl:
            u_idx = torch.unique(u)
            i_idx = torch.unique(i)
            cl_loss = (
                InfoNCE(self._users[u_idx],    self._users_cl[u_idx],    tau=args.cl_tau, b_cos=True)
            + InfoNCE(self._items[i_idx],    self._items_cl[i_idx],    tau=args.cl_tau, b_cos=True)
            )
            loss = loss + args.cl_lambda * cl_loss

        return loss


    def train_func_batch(self):
        train_losses = []
        train_user = self.dataset.train_user
        train_item = self.dataset.train_item
        for _ in range(0, train_user.shape[0], args.loss_batch_size):
            self.computer()
            train_loss = self.loss_bpr(train_user[_:_+args.loss_batch_size], train_item[_:_+args.loss_batch_size])
            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()
            train_losses.append(train_loss)
        return torch.stack(train_losses).mean()

    def save_emb(self):
        """
        保存嵌入
        """
        torch.save(self.embedding_user, f'../saved/{args.data:s}_user.pt')
        torch.save(self.embedding_item, f'../saved/{args.data:s}_item.pt')

    def load_emb(self):
        """
        加载嵌入
        """
        self.embedding_user = torch.load(f'../saved/{args.data:s}_user.pt').to(args.device)
        self.embedding_item = torch.load(f'../saved/{args.data:s}_item.pt').to(args.device)







