import torch
import os
import random
import numpy as np
from torch.nn.utils.rnn import pad_sequence

def create_directory(d):
    if d and not os.path.exists(d):
        os.makedirs(d)
    return d

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def collate_fn(batch):
    import numpy as np
    import torch

    max_len = max([len(f["input_ids"]) for f in batch])
    max_sent = max([len(f["sent_pos"]) for f in batch])

    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] for f in batch]
    entity_pos = [f["entity_pos"] for f in batch]
    hts = [f["hts"] for f in batch]
    if "hts_sent_pos" in batch[0]:
        hts_sent_pos = [f["hts_sent_pos"] for f in batch]
    else:
        hts_sent_pos = [0]
    sent_pos = [f["sent_pos"] for f in batch]
    sent_labels = [f["sent_labels"] for f in batch if "sent_labels" in f]
    attns = [f["attns"] for f in batch if "attns" in f]

    if "token_labels" in batch[0] and batch[0]["token_labels"] is not None:
        max_doc_len = max(len(f["token_labels"][0]) for f in batch)
        token_labels_list = []
        for f in batch:
            tl = np.array(f["token_labels"], dtype=float)  # shape (P_i, L_i)
            pad_width = ((0, 0), (0, max_doc_len - tl.shape[1]))
            tl_padded = np.pad(tl, pad_width, mode="constant", constant_values=0.0)
            token_labels_list.append(torch.from_numpy(tl_padded))  # tensor (P_i, max_doc_len)
        token_labels = torch.cat(token_labels_list, dim=0).float()
    else:
        token_labels = None

    if "trigger_sent_labels" in batch[0] and batch[0]["trigger_sent_labels"] is not None:
        num_rels = len(batch[0]["trigger_sent_labels"])
        trigger_sent_labels_list = []
        for f in batch:
            tsl = np.array(f["trigger_sent_labels"], dtype=float)  # shape (num_rels, sent_len)
            pad_width = ((0, 0), (0, max_sent - tsl.shape[1]))  # pad sentence axis
            tsl_padded = np.pad(tsl, pad_width, mode="constant", constant_values=0.0)
            trigger_sent_labels_list.append(torch.from_numpy(tsl_padded))  # (num_rels, max_sent)
        trigger_sent_labels = torch.stack(trigger_sent_labels_list, dim=0)  # (B, num_rels, max_sent)
    else:
        trigger_sent_labels = None

    

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)

    labels = [torch.tensor(label) for label in labels]
    labels = torch.cat(labels, dim=0)

    na_labels = (labels[:, 0] == 0).long()  

    
    if sent_labels != [] and None not in sent_labels:
        sent_labels_tensor = []
        for sent_label in sent_labels:
            sent_label = np.array(sent_label)
            row_all_zero = np.all(sent_label == 0, axis=1)   # shape: (num_sent,)

            sent_labels_tensor.append(np.pad(sent_label, ((0, 0), (0, max_sent - sent_label.shape[1]))))
        sent_labels_tensor = torch.from_numpy(np.concatenate(sent_labels_tensor, axis=0))
    else:
        sent_labels_tensor = None


    if attns != []:
        attns = [np.pad(attn, ((0, 0), (0, max_len - attn.shape[1]))) for attn in attns]
        attns = torch.from_numpy(np.concatenate(attns, axis=0))
    else:
        attns = None

    output = (
        input_ids, input_mask, labels, entity_pos, hts, sent_pos,
        hts_sent_pos, sent_labels_tensor, attns, token_labels, trigger_sent_labels, na_labels
    )
    return output

def create_directory(d):
    if d and not os.path.exists(d):
        os.makedirs(d)
    return d

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
