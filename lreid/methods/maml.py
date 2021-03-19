# This code is modified from https://github.com/dragen1860/MAML-Pytorch and https://github.com/katerakelly/pytorch-maml

from methods import backbone
import torch
import torch.nn as nn
import numpy as np
from methods.meta_template import MetaTemplate
from methods.drop_grad import DropGrad

class MAML(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support, tf_path=None, approx=False, dropout_method='none', dropout_rate=0., dropout_schedule='constant'):
        super(MAML, self).__init__( model_func,  n_way, n_support, tf_path=tf_path, change_way=False)

        self.loss_fn = nn.CrossEntropyLoss()
        self.classifier = backbone.Linear_fw(self.feat_dim, n_way)
        self.classifier.bias.data.fill_(0)

        self.batch_size = 4
        self.task_update_num = 5 # first-order grad learning rate
        self.train_lr = 0.01 # meta grad learning rate * task_update_num = final meta grad learning rate
        self.approx = approx # first order approx.

        self.dropout = DropGrad(dropout_method, dropout_rate, dropout_schedule)
        self.optimizer = torch.optim.Adam(self.parameters())

    def forward(self, x):
        out = self.feature.forward(x)
        scores = self.classifier.forward(out)
        return scores

    def set_forward(self, x):
        # x 5,21,3,224,224
        x = x.cuda()
        # x_a_i 25,3,224,224
        x_a_i = x[:, :self.n_support, :, :, :].contiguous().view(self.n_way * self.n_support, *x.size()[2:])
        # x_b_i 80,3,224,224
        x_b_i = x[:, self.n_support:, :, :, :].contiguous().view(self.n_way * self.n_query,   *x.size()[2:])
        # y_a_i 25
        y_a_i = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).cuda()

        fast_parameters = list(self.parameters())
        for weight in self.parameters():
            weight.fast = None
        self.zero_grad()
        # task_update_num = 5
        for task_step in range(self.task_update_num):
            # forward and get grad on support data
            scores = self.forward(x_a_i)
            set_loss = self.loss_fn(scores, y_a_i)
            grad = torch.autograd.grad(set_loss, fast_parameters, create_graph=True)

            # first order approx
            if self.approx:
                grad = [g.detach() for g in grad]

            # update
            fast_parameters = []
            for k, (name, weight) in enumerate(self.named_parameters()):
                # regularization
                if self.training:
                    grad[k] = self.dropout(grad[k])
                # update
                if weight.fast is None:
                    weight.fast = weight - self.train_lr * grad[k] # link fast weight to weight
                else:
                    weight.fast = weight.fast - self.train_lr * grad[k]
                fast_parameters.append(weight.fast)
        # forward and get loss on query data
        scores = self.forward(x_b_i)
        return scores

    def set_forward_adaptation(self,x, is_feature = False): #overwrite parrent function
        raise ValueError('MAML performs further adapation simply by increasing task_update_num')

    def set_forward_loss(self, x):
        scores = self.set_forward(x)
        y_b_i = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query)).cuda()
        loss = self.loss_fn(scores, y_b_i)
        return scores, loss

    def train_loop(self, epoch, stop_epoch, train_loader, total_it): #overwrite parrent function
        print_freq = len(train_loader) // 5
        avg_loss = 0
        task_count = 0
        loss_all = []
        self.optimizer.zero_grad()

        # update dropout rate
        self.dropout.update_rate(epoch, stop_epoch) ## epoch / (stop_epoch - 1) * self.dropout_p

        # train loop
        for i, (x,_) in enumerate(train_loader):
            self.n_query = x.size(1) - self.n_support
            assert(self.n_way==x.size(0))

            # get loss
            self.optimizer.zero_grad()
            _, loss = self.set_forward_loss(x)
            avg_loss = avg_loss+loss.item()
            loss_all.append(loss)

            # batch update
            task_count += 1 # 5 class 5 images for 1 task
            if task_count == self.batch_size:
                loss_q = torch.stack(loss_all).sum(0)
                loss_q.backward()
                self.optimizer.step() # 5*5*self.batch_size (4) =100 image for 1 batch
                task_count = 0
                loss_all = []

            # print out
            if (i + 1) % print_freq == 0:
                print('Epoch {:d}/{:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch + 1, stop_epoch, i + 1, len(train_loader), avg_loss/float(i+1)))
            if (total_it + 1) % 10 == 0 and self.tf_writer is not None:
                self.tf_writer.add_scalar('maml/query_loss', loss.item(), total_it + 1)
            total_it += 1

        return total_it, avg_loss/float(i+1)

    def test_loop(self, test_loader, epoch=None, return_std=False): #overwrite parrent function
        loss = 0.
        count = 0
        acc_all = []

        iter_num = len(test_loader)
        for i, (x,_) in enumerate(test_loader):
            self.n_query = x.size(1) - self.n_support
            assert(self.n_way==x.size(0))
            correct_this, count_this, loss_this = self.correct(x)
            acc_all.append(correct_this/ count_this *100 )
            loss += loss_this
            count += count_this

        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        print('--- %d Loss = %.6f ---' %(iter_num,  loss/count))
        print('--- %d Test Acc = %4.2f%% +- %4.2f%% ---' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))

        if self.tf_writer is not None:
            assert(epoch is not None)
            self.tf_writer.add_scalar('maml/val_loss', loss/count, epoch)

        if return_std:
            return acc_mean, acc_std
        else:
            return acc_mean
