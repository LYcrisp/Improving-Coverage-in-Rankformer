原始代码[Rankformer (GitHub)](https://github.com/StupidThree/Rankformer)，论文[Rankformer: A Graph Transformer for Recommendation based on Ranking Objective (ACM WWW ’25)](https://doi.org/10.1145/3696410.3714547)

## 运行方法

### 改进后的模型

第一个数据集

```cmd
python -u code/main.py --data=Ali-Display --use_gcn --gcn_layers=2 --gcn_left=0.5 --gcn_right=0.5 --learning_rate=1e-3 --loss_batch_size=2048 --valid_interval=1 --sgl_lambda=0.018 --use_rankformer --rankformer_layers=2 --rankformer_tau=0.14  --use_sgl --tail_lambda=1.5 --use_coverage --use_tail_loss 
```
第二个数据集
```cmd
python -u code/main.py --data=Epinions --use_gcn --gcn_layers=2 --gcn_left=0.5 --gcn_right=0.5 --learning_rate=1e-3 --loss_batch_size=2048 --valid_interval=1 --sgl_lambda=0.015 --use_rankformer --rankformer_layers=1 --rankformer_tau=0.14 --use_sgl --tail_lambda=1.2 --use_coverage --use_tail_loss
```

### 论文的原始模型

```cmd
python -u code/main.py --data=Ali-Display --use_cl --use_gcn --gcn_layers=2 --gcn_left=0.5 --gcn_right=0.5 --use_rankformer --rankformer_layers=5 --rankformer_tau=0.1 --learning_rate=1e-3 --loss_batch_size=2048 --valid_interval=1 --use_coverage
```

```cmd
python -u code/main.py --data=Epinions --use_cl --use_gcn --gcn_layers=2 --gcn_left=0.5 --gcn_right=0.5 --use_rankformer --rankformer_layers=3 --rankformer_tau=0.2 --learning_rate=1e-3 --loss_batch_size=2048 --valid_interval=1
```

### 论文的原始模型加入长尾损失

```cmd
python -u code/main.py --data=Ali-Display --use_cl --use_gcn --gcn_layers=2 --gcn_left=0.5 --gcn_right=0.5 --use_rankformer --rankformer_layers=5 --rankformer_tau=0.1 --learning_rate=1e-3 --loss_batch_size=2048 --valid_interval=1 --use_coverage --tail_lambda=2.5 --use_tail_loss
```

```cmd
python -u code/main.py --data=Epinions --use_cl --use_gcn --gcn_layers=2 --gcn_left=0.5 --gcn_right=0.5 --use_rankformer --rankformer_layers=3 --rankformer_tau=0.2 --learning_rate=1e-3 --loss_batch_size=2048 --valid_interval=1 --use_coverage --tail_lambda=1.2 --use_tail_loss

```

## 运行结果
####  Ali-Display 数据集

| 方法                  | NDCG@20  | Recall@20 | Precision@20 | Coverage@20 | Tail-Coverage@20 |
| --------------------- | -------- | --------- | ------------ | ----------- | ---------------- |
| 原论文模型            | 0.067911 | 0.125297  | 0.013703     | 0.4326      | 0.1580           |
| 原论文 + 长尾损失加权 | 0.066489 | 0.122714  | 0.013467     | ↑0.5322     | ↑0.3809          |
| 本方法                | 0.067115 | 0.123634  | 0.013514     | ↑0.5889     | ↑0.3911          |



#### Epinions 数据集

| 方法                  | NDCG@20  | Recall@20 | Precision@20 | Coverage@20 | Tail-Coverage@20 |
| --------------------- | -------- | --------- | ------------ | ----------- | ---------------- |
| 原论文模型            | 0.063305 | 0.100399  | 0.018947     | 0.2775      | 0.0871           |
| 原论文 + 长尾损失加权 | 0.063121 | 0.100074  | 0.018895     | ↑0.3106     | ↑0.1265          |
| 本方法                | 0.062254 | 0.098887  | 0.018709     | ↑0.3658     | ↑0.1679          |

------