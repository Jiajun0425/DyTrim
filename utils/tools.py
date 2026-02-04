import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import random

class BatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, use_beta=True):
        super(BatchNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # 初始化可学习的参数 gamma 和 beta
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features)) if use_beta else None

        # 用于存储训练时计算的均值和方差
        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)
        
        # 控制 beta 是否启用
        self.use_beta = use_beta
        
       
        init.constant_(self.beta, 0)    # 将 BatchNorm 的 bias 初始化为 0
        

    def forward(self, x):
        # 训练阶段与推理阶段区分
        if self.training:
            # 计算当前批次的均值和方差
            self.batch_mean = x.mean(dim=0)
            self.batch_var = x.var(dim=0, unbiased=False)

            # 使用批量均值和方差进行标准化
            x_normalized = (x - self.batch_mean) / torch.sqrt(self.batch_var + self.eps)

            # 更新运行均值和方差
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * self.batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * self.batch_var

        else:
            # 推理阶段，使用训练时计算的均值和方差
            x_normalized = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)

        # 缩放
        out = self.gamma * x_normalized
        
        # 如果 use_beta 为 True，应用 beta
        if self.use_beta:
            out += self.beta
        
        return out
    

def set_seed(seed=42):
    # 设置 Python 的随机数种子
    random.seed(seed)
    
    # 设置 NumPy 的随机数种子
    np.random.seed(seed)
    
    # 设置 PyTorch 的随机数种子
    torch.manual_seed(seed)
    
    # 设置 GPU 的随机数种子（如果使用 CUDA）
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果有多个 GPU
    
    # 设置 CUDNN 的确定性算法，确保每次计算结果一致
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False