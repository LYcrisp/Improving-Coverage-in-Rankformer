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

