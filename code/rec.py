import torch
import torch.nn as nn
import torch.nn.functional as F
from parse import args

def InfoNCE(x, y, tau=0.15, b_cos=True):
    """
    计算对比学习损失
    """
    if b_cos:
        x, y = F.normalize(x), F.normalize(y)
    return -torch.diag(F.log_softmax((x@y.T)/tau, dim=1)).mean()

def sparse_sum(values, indices0, indices1, n):
    if indices0 is None:
        assert (len(indices1.shape) == 1 and values.shape[0] == indices1.shape[0])
    else:
        assert (len(indices0.shape) == 1 and len(indices1.shape) == 1)
        assert (indices0.shape[0] == indices1.shape[0])
    # assert (len(values.shape) <= 2)
    return torch.zeros([n]+list(values.shape)[1:], device=values.device, dtype=values.dtype).index_add(0, indices1, values if indices0 is None else values[indices0])


def rest_sum(values, indices0, indices1, n):
    return values.sum(0).unsqueeze(0)-sparse_sum(values, indices0, indices1, n)


class GCN(nn.Module):
    """
    图神经网络（Graph Convolutional Network）模块，
    用于在用户-物品交互图上传播信息并更新嵌入表示。
    """

    def __init__(self, dataset, alpha=1.0, beta=0.0):
        super(GCN, self).__init__()
        self.dataset = dataset  # 数据集对象，包含用户数、物品数、交互边等信息
        self.alpha, self.beta = alpha, beta  # 控制邻居归一化权重的两个超参数

    def forward(self, x, u, i):
        # 获取用户和物品总数
        n, m = self.dataset.num_users, self.dataset.num_items

        # 统计每个用户的入度 du（被交互的次数），shape: [n]
        du = sparse_sum(torch.ones_like(u), None, u, n).clamp(1)  # 避免除以 0

        # 统计每个物品的入度 di（被交互的次数），shape: [m]
        di = sparse_sum(torch.ones_like(i), None, i, m).clamp(1)

        # 构建用户接收信息的归一化权重 w1，来自于物品邻居
        # 公式近似为：w1 ∝ 1 / (du^α * di^β)
        w1 = (torch.ones_like(u) / du[u].pow(self.alpha) / di[i].pow(self.beta)).unsqueeze(-1)  # shape: [E, 1]

        # 构建物品接收信息的归一化权重 w2，来自于用户邻居
        # 公式近似为：w2 ∝ 1 / (du^β * di^α)
        w2 = (torch.ones_like(u) / du[u].pow(self.beta) / di[i].pow(self.alpha)).unsqueeze(-1)  # shape: [E, 1]

        # 将输入嵌入 x 分离为用户嵌入 xu 和物品嵌入 xi
        xu, xi = torch.split(x, [n, m])  # xu: [n, dim], xi: [m, dim]

        # 对于每个用户，根据其交互过的物品（xi[i]）加权聚合，得到更新后的用户表示
        zu = sparse_sum(xi[i] * w1, None, u, n)  # shape: [n, dim]

        # 对于每个物品，根据其交互过的用户（xu[u]）加权聚合，得到更新后的物品表示
        zi = sparse_sum(xu[u] * w2, None, i, m)  # shape: [m, dim]

        # 将用户和物品的更新嵌入拼接返回，shape: [n + m, dim]
        return torch.concat([zu, zi], 0)



class Rankformer(nn.Module):
    """
    排序感知建模模块：
    - 结合用户与物品的正负交互信息（交互与未交互）
    - 引入评分相似性与注意机制
    - 支持基准项（benchmark）、残差增强、归一化控制
    """

    def __init__(self, dataset, alpha):
        super(Rankformer, self).__init__()
        self.dataset = dataset  # 数据集（包含用户数、物品数、交互边等）
        self.my_parameters = []  # 可用于注册额外参数（此处为空）
        self.alpha = alpha      # 残差增强常数，用于稳定优化

    def forward(self, x, u, i):
        # 用户数和物品数
        n, m = self.dataset.num_users, self.dataset.num_items

        # 统计每个用户的正交互个数（dui），负交互数（duj） = m - dui
        dui = sparse_sum(torch.ones_like(u), None, u, n)  # shape: [n]
        duj = m - dui
        dui, duj = dui.clamp(1).unsqueeze(1), duj.clamp(1).unsqueeze(1)  # 避免除以 0，扩展维度为 [n,1]

        # 将嵌入划分为用户和物品，并对其做归一化（用于相似度计算）
        xu, xi = torch.split(F.normalize(x), [n, m])  # 归一化后的用户和物品表示
        vu, vi = torch.split(x, [n, m])              # 原始（未归一化）嵌入

        # 计算每个 (u, i) 对的评分相似度（点积）
        xui = (xu[u] * xi[i]).sum(1).unsqueeze(1)  # shape: [E, 1]

        # 正交互物品（sxi）、负交互物品（sxj）：基于 u-i 对索引聚合
        sxi = sparse_sum(xi, i, u, n)              # 正交互物品的向量和（用户维度）
        sxj = xi.sum(0) - sxi                      # 剩余物品（负样本）和 = 全体物品 - 正样本和

        # 原始物品向量的正交互/负交互聚合（未归一化）
        svi = sparse_sum(vi, i, u, n)
        svj = vi.sum(0) - svi

        # 计算 benchmark 项：b_pos 为正样本平均相似度，b_neg 为负样本平均相似度
        b_pos = (xu * sxi).sum(1).unsqueeze(1) / dui  # 正交互基准评分
        b_neg = (xu * sxj).sum(1).unsqueeze(1) / duj  # 负交互基准评分

        # 可选：是否删除 benchmark 项
        if args.del_benchmark:
            b_pos, b_neg = 0, 0

        # 构建物品-物品之间的相似性（注意力机制基础）
        xxi = xi.unsqueeze(1) * xi.unsqueeze(2)  # shape: [m, dim, dim]
        xvi = xi.unsqueeze(1) * vi.unsqueeze(2)

        # 用户损失权重项 d1/d2（用于加权归一化）
        du1 = (xu * sxi).sum(1).unsqueeze(1) / dui - b_neg + self.alpha
        du2 = -(xu * sxj).sum(1).unsqueeze(1) / duj + b_pos + self.alpha

        # 物品损失权重项 d1/d2，基于邻居聚合的加权向量
        di1 = (xi * sparse_sum(xu / dui, u, i, m)).sum(1).unsqueeze(1) + \
              sparse_sum((-b_neg + self.alpha) / dui, u, i, m)
        di2 = -(xi * rest_sum(xu / duj, u, i, m)).sum(1).unsqueeze(1) + \
              rest_sum((b_pos + self.alpha) / duj, u, i, m)

        # 用户表示增强（正负分支）
        A = sparse_sum(xui * vi[i], None, u, n)  # 用户聚合的打分增强项
        zu1 = A / dui - svi * (b_neg - self.alpha) / dui
        zu2 = (torch.mm(xu, xvi.sum(0)) - A) / duj - svj * (b_pos + self.alpha) / duj

        # 物品表示增强（正负分支）
        zi1 = sparse_sum(xui * vu[u] / dui[u], None, i, m) - \
              sparse_sum(vu * (b_neg - self.alpha) / dui, u, i, m)
        zi2 = torch.mm(xi, ((xu / duj).unsqueeze(2) * vu.unsqueeze(1)).sum(0)) - \
              sparse_sum(xui * (vu / duj)[u], None, i, m) - \
              rest_sum(vu * (b_pos + self.alpha) / duj, u, i, m)

        # 拼接正负更新向量
        z1 = torch.concat([zu1, zi1], 0)  # 正反馈
        z2 = torch.concat([zu2, zi2], 0)  # 负反馈

        # 拼接正负权重项
        d1 = torch.concat([du1, di1], 0).clamp(args.rankformer_clamp_value)  # 限幅防梯度爆炸
        d2 = torch.concat([du2, di2], 0).clamp(args.rankformer_clamp_value)

        # 可选：是否删除负反馈分支（例如只使用正反馈构造更新）
        if args.del_neg:
            z2, d2 = 0, 0

        # 合并表示与归一化项
        z = z1 + z2
        d = d1 + d2

        # 是否使用最终归一化（Ω范数）
        if args.del_omega_norm:
            return z
        return z / d  # 返回归一化后的嵌入增强结果



