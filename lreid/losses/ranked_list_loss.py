import torch

def normalize_rank(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist_rank(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def rank_loss(dist_mat, labels, margin, alpha, tval):
    """
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]

    """
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    total_loss = 0.0
    for ind in range(N):
        is_pos = labels.eq(labels[ind])
        is_pos[ind] = 0
        is_neg = labels.ne(labels[ind])

        dist_ap = dist_mat[ind][is_pos]
        dist_an = dist_mat[ind][is_neg]

        ap_is_pos = torch.clamp(torch.add(dist_ap, margin - alpha), min=0.0)
        ap_pos_num = ap_is_pos.size(0) + 1e-5
        ap_pos_val_sum = torch.sum(ap_is_pos)
        loss_ap = torch.div(ap_pos_val_sum, float(ap_pos_num))

        an_is_pos = torch.lt(dist_an, alpha)
        an_less_alpha = dist_an[an_is_pos]
        an_weight = torch.exp(tval * (-1 * an_less_alpha + alpha))
        an_weight_sum = torch.sum(an_weight) + 1e-5
        an_dist_lm = alpha - an_less_alpha
        an_ln_sum = torch.sum(torch.mul(an_dist_lm, an_weight))
        loss_an = torch.div(an_ln_sum, an_weight_sum)

        total_loss = total_loss + loss_ap + loss_an
    total_loss = total_loss * 1.0 / N
    return total_loss


class RankedLoss(torch.nn.Module):
    "Ranked_List_Loss_for_Deep_Metric_Learning_CVPR_2019_paper"

    def __init__(self, margin=None, alpha=None, tval=None, normalize_feature=False):
        super(RankedLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.tval = tval
        self.normalize_feature = normalize_feature

    def forward(self, global_feat1, global_feat2, global_feat3, labels1, labels2, labels3):
        if self.normalize_feature:
            global_feat1 = normalize_rank(global_feat1, axis=-1)
        dist_mat = euclidean_dist_rank(global_feat1, global_feat1)
        total_loss = rank_loss(dist_mat, labels1, self.margin, self.alpha, self.tval)

        return total_loss