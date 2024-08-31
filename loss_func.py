import torch
import torch.nn as nn
import torch.nn.functional as F


class CELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.xent_loss = nn.CrossEntropyLoss()
    def convert_vec_to_int(self, label_vec):
        label_int = torch.argmax(label_vec, -1)
        return label_int
    def forward(self, embeddings, output_vec, targets):
        targets = self.convert_vec_to_int(targets)
        return self.xent_loss(output_vec, targets)


class SupConLoss(nn.Module):
    def __init__(self, alpha, temp):
        super().__init__()
        self.xent_loss = nn.CrossEntropyLoss()
        self.alpha = alpha
        self.temp = temp

    def nt_xent_loss(self, anchor, target, labels):
        with torch.no_grad():
            labels = labels.unsqueeze(-1)
            mask = torch.eq(labels, labels.transpose(0, 1))      # require pytorch version >= 1.10
            # delete diag elem
            mask = mask ^ torch.diag_embed(torch.diag(mask))
        # compute logits
        anchor_dot_target = torch.einsum('bd,cd->bc', anchor, target) / self.temp
        # delete diag elem
        anchor_dot_target = anchor_dot_target - torch.diag_embed(torch.diag(anchor_dot_target))
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_target, dim=1, keepdim=True)
        logits = anchor_dot_target - logits_max.detach()
        # compute log prob
        exp_logits = torch.exp(logits)
        # mask out positives
        logits = logits * mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
        # in case that mask.sum(1) is zero
        mask_sum = mask.sum(dim=1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        # compute log-likelihood
        pos_logits = (mask * log_prob).sum(dim=1) / mask_sum.detach()
        loss = -1 * pos_logits.mean()
        return loss

    def convert_vec_to_int(self, label_vec):
        label_int = torch.argmax(label_vec, -1)
        return label_int

    def forward(self, embeddings, output_vec, targets):
        targets = self.convert_vec_to_int(targets)
        normed_cls_feats = F.normalize(embeddings, dim=-1)
        ce_loss = (1 - self.alpha) * self.xent_loss(output_vec, targets)
        cl_loss = self.alpha * self.nt_xent_loss(normed_cls_feats, normed_cls_feats, targets)
        return ce_loss + cl_loss
