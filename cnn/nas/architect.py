import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


# 该函数是为了求alpha的导数而设置,

def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])


class Architect(object):
  #计算alpha的梯度

  def __init__(self, model, args):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model = model
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(), #训练路径参数alpha
        lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

  def _compute_unrolled_model(self, input, target, eta, network_optimizer):
    loss = self.model._loss(input, target) #首先计算一下前向损失
    theta = _concat(self.model.parameters()).data #将所有权重copy一下
    try:
      moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum) #替代了optimizer.step() 从而不改变网络权重
    except:
      moment = torch.zeros_like(theta)
    '''
    带权重衰减和动量的sgd optimizer.step()会使momentum变化 
    network_optimizer.state[v]['momentum_buffer'] 会变成计算之后的moment

    v_{t+1} = mu * v_{t} + g_{t+1} + weight_decay * w  其中weight_decay*w是正则项 
    w_{t+1} = w_{t} - lr * v_{t+1}
    所以：
    moment = mu * v_{t} 
    dw = g_{t+1} + weight_decay * w 
    v_{t+1} = moment + dw
    w'= w - lr * v_{t+1}
    '''
    dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay*theta
    unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta))
    return unrolled_model

  def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):
    # eta是网络的学习率
    self.optimizer.zero_grad() #将该batch的alpha的梯度设置为0
    if unrolled: #展开要求w‘
        self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
    else: #不展开的话首先将valid计算一次前向传播，再计算损失
        self._backward_step(input_valid, target_valid)
    self.optimizer.step()  #alpha梯度更新

  def _backward_step(self, input_valid, target_valid):
    loss = self.model._loss(input_valid, target_valid)
    loss.backward()

  def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
    unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
    unrolled_loss = unrolled_model._loss(input_valid, target_valid) #L_val(w')

    unrolled_loss.backward()#得到了新的权重
    dalpha = [v.grad for v in unrolled_model.arch_parameters()] #dalpha{L_val(w', alpha)}
    vector = [v.grad.data for v in unrolled_model.parameters()]  #dw'{L_val(w', alpha)}
    implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

     #更新最后的梯度gradient = dalpha - lr*hessian
    for g, ig in zip(dalpha, implicit_grads):
      g.data.sub_(eta, ig.data)

    for v, g in zip(self.model.arch_parameters(), dalpha):
      if v.grad is None:
        v.grad = Variable(g.data)
      else:
        v.grad.data.copy_(g.data)

  def _construct_model_from_theta(self, theta): #theta=w'
    model_new = self.model.new()
    model_dict = self.model.state_dict()

    params, offset = {}, 0
    for k, v in self.model.named_parameters():
      v_length = np.prod(v.size())
      params[k] = theta[offset: offset+v_length].view(v.size())
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    model_new.load_state_dict(model_dict)
    return model_new.cuda()

  def _hessian_vector_product(self, vector, input, target, r=1e-2):
    R = r / _concat(vector).norm()
    # w+ = w + eps*dw'
    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)
    loss = self.model._loss(input, target)
    grads_p = torch.autograd.grad(loss, self.model.arch_parameters())
    # w- = w - eps*dw'

    for p, v in zip(self.model.parameters(), vector):
      p.data.sub_(2*R, v)
    loss = self.model._loss(input, target)
    grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

