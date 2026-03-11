import torch
import torch.nn as nn
import torch.nn.functional as F


class ATLoss(nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        # TH label
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, 0] = 1.0
        labels[:, 0] = 0.0

        p_mask = labels + th_label
        n_mask = 1 - labels

        # Rank positive classes to TH
        logit1 = logits - (1 - p_mask) * 1e30
        loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1)
        # Rank TH to negative classes
        logit2 = logits - (1 - n_mask) * 1e30
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)
        # Sum two parts
        loss = loss1 + loss2
        loss = loss.mean()
        return loss

    def get_label(self, logits, num_labels=-1):

        th_logit = logits[:, 0].unsqueeze(1)  # theshold is norelation
        output = torch.zeros_like(logits).to(logits)
        mask = (logits > th_logit)
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1] # smallest logits among the num_labels
            # predictions are those logits > thresh and logits >= smallest
            mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        # if no such relation label exist: set its label to 'Nolabel'
        output[:, 0] = (output.sum(1) == 0.).to(logits)
        return output

    def get_score(self, logits, num_labels=-1):

        if num_labels > 0:
            return torch.topk(logits, num_labels, dim=1)
        else:
            return logits[:,1] - logits[:,0], 0


import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalHierarchicalKLLoss(nn.Module):
    def __init__(self, offset: int = 1,alpha: float = 0.05, beta: float = 0.01, eps: float = 1e-30):
        super().__init__()
        self.offset = offset
        self.alpha = alpha
        self.beta   = beta
        self.eps    = eps

    def forward(
        self,
        doc_attn: torch.Tensor,      # (N_pairs, doc_len) raw attention scores
        sent_labels: torch.Tensor,   # (N_pairs, S) sentence-level gold counts
        token_labels: torch.Tensor,  # (N_pairs, doc_len) token-level gold (0/1)
        sent_pos: list               # list of list of (start, end) per sentence
    ) -> torch.Tensor:
        """
        Compute token-level KL loss only over evidence sentence tokens.
        """
        N, _ = doc_attn.shape
        offset = self.offset
        eps    = self.eps
        alpha  = self.alpha

        total_loss = 0.0
        count = 0

        for i in range(N):
            ev_sents = (sent_labels[i] > 0).nonzero(as_tuple=False).view(-1)
            if ev_sents.numel() == 0:
                continue

            toks = []
            for sidx in ev_sents:
                start, end = sent_pos[i][sidx.item()]
                toks.extend(range(start + offset, end + offset))
            toks = torch.tensor(toks, device=doc_attn.device, dtype=torch.long)

            scores = doc_attn[i, toks].clamp(min=eps)      # (L_loc,)
            p_loc  = scores / (scores.sum() + eps)         # (L_loc,)
            log_p  = torch.log(p_loc + eps).unsqueeze(0)   # (1, L_loc)
            tl     = token_labels[i, toks].float()         # (L_loc,)
            one_hot = tl / (tl.sum() + eps)                # (L_loc,)
            # label smoothing: mix with uniform distribution
            uniform = torch.full_like(one_hot, 1.0 / one_hot.numel())
            q_loc = (1 - alpha) * one_hot + alpha * uniform
            q_loc = q_loc.unsqueeze(0)                    # (1, L_loc)

            loss = F.kl_div(log_p, q_loc, reduction="batchmean")

            total_loss += loss
            count += 1

        return total_loss / max(1, count)
