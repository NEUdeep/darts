import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype


#本文件是为了构建超网

class MixedOp(nn.Module): # 定义了一个cell中两个节点可选择的op和他们对应的权重，目的是组成一个cell所需要的节点链接，假设每个节点之间都有多个op，如何选择其中一个op，这应该产生子网络的关键

  def __init__(self, C, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES:
      op = OPS[primitive](C, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)

  def forward(self, x, weights):
    return sum(w * op(x) for w, op in zip(weights, self._ops)) # weights就是alpha 输出是对各操作的加权相加


class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    self.reduction = reduction

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps): # i = 0,1,2,3
      for j in range(2+i): # j = 2,3,4,5 因为每处理一个节点，该节点就变为下一个节点的前继
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride) #初始化得到一个节点到另一个节点的操作集合
        self._ops.append(op)  # len = 14
        '''
        self._ops[0,1]代表的是内部节点0的前继操作
        self._ops[2,3,4]代表的是内部节点1的前继操作
        self._ops[5,6,7,8]代表的是内部节点2的前继操作
        self._ops[9,10,11,12,13]代表的是内部节点3的前继操作
        '''

  def forward(self, s0, s1, weights):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      #因为先将该节点到另一个节点各操作后的输出相加，再把该节点与所有前继节点的操作相加 所以输出维度不变
      s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)
      '''
      i=1:  s2 = self._ops[0](s0,weights[0]) + self._ops[1](s1,weights[1])即内部节点0
      i=2:  s3 = self._ops[2](s0,weights[2]) + self._ops[3](s1,weights[3]) + self._ops[4](s2,weights[4])即内部节点1
      i=3、4依次计算得到s4，s5
      由此可知len(weights)也等于14，因为有8个操作，所以weight[i]有8个值
      '''

    return torch.cat(states[-self._multiplier:], dim=1)  #将后4个节点的输出按channel拼接 channel变为4倍
    # eg:(10,3,2,2)+(10,3,2,2)-->(10,6,2,2)


# （只解析了初始化部分以及genotype部分）
class Network(nn.Module):

  def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
    # c =16 num_class = 10,layers = 8 ,criterion = nn.CrossEntropyLoss().cuda()
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps # 4
    self._multiplier = multiplier # 4 通道数乘数因子 因为有4个中间节点 代表通道数要扩大4倍

    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
    # 对于第一个cell来说，stem即是s0，也是s1
    # C_prev_prev, C_prev是输出channel 48
    # C_curr 现在是输入channel 16
 
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      # 在 1/3 和 2/3 层减小特征size并且加倍通道.
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr
    '''
    layers = 8, 第2和5个cell是reduction_cell
    cells[0]: cell = Cell(4, 4, 48,  48,  16, false,  false) 输出[N,16*4,h,w]
    cells[1]: cell = Cell(4, 4, 48,  64,  16, false,  false) 输出[N,16*4,h,w]
    cells[2]: cell = Cell(4, 4, 64,  64,  32, True,   false) 输出[N,32*4,h/2,w/2]
    cells[3]: cell = Cell(4, 4, 64,  128, 32, false,  false) 输出[N,32*4,h/2,w/2]
    cells[4]: cell = Cell(4, 4, 128, 128, 32, false,  false) 输出[N,32*4,h/2,w/2]
    cells[5]: cell = Cell(4, 4, 128, 128, 64, True,   false) 输出[N,64*4,h/4,w/4]
    cells[6]: cell = Cell(4, 4, 128, 256, 64, false,  false) 输出[N,64*4,h/4,w/4]
    cells[7]: cell = Cell(4, 4, 256, 256, 64, false,  false) 输出[N,64*4,h/4,w/4]
    '''
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

    self._initialize_alphas()

  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input):
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        weights = F.softmax(self.alphas_reduce, dim=-1)
      else:
        weights = F.softmax(self.alphas_normal, dim=-1)
      s0, s1 = s1, cell(s0, s1, weights)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

  def _loss(self, input, target):
    logits = self(input)
    return self._criterion(logits, target) 

  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(PRIMITIVES)

    self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
    ]

  def arch_parameters(self):
    return self._arch_parameters


  def genotype(self):
    # weights [14,8]
    def _parse(weights):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps): #[0,1,2,3] 代表了中间节点
        end = start + n
        W = weights[start:end].copy() #获取当前中间节点至前继节点的权重
        #sorted返回按alpha降序排列的前继节点的索引
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):# [0~7]
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k #找出与前继节点不包含none的操作中alpha值最大的操作索引,为什么要除去none操作，文中给了两点解释:https://img-blog.csdnimg.cn/20191024204040433.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2NTMwOTky,size_16,color_FFFFFF,t_70#pic_center
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene

    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy()) #按行进行softmax
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

