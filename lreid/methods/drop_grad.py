import numpy as np
import torch

class DropGrad(torch.nn.Module):
  def __init__(self, method='gaussian', rate=0.1, schedule='constant'):
    super(DropGrad, self).__init__()
    self.method = method
    self.rate = rate if self.method != 'gaussian' else np.sqrt(rate/(1 - rate))
    self.schedule = schedule

  def update_rate(self, epoch ,stop_epoch):
    if self.schedule == 'constant':
      self.cur_rate = self.rate
    elif self.schedule == 'linear':
      self.cur_rate = self.rate * epoch  / (stop_epoch - 1)
    else:
      raise Exception('no such DropGrad schedule')

  def forward(self, input):
    if self.method == 'binary':
      output = input * (torch.gt(torch.rand_like(input), self.cur_rate).float() * (1 / (1 - self.cur_rate)))
    elif self.method == 'gaussian':
      output = input * torch.normal(mean=torch.ones_like(input), std=torch.ones_like(input)*self.cur_rate)
    elif self.method == 'none':
      output = input
    else:
      raise Exception('no such DropGrad method')
    return output
