import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from graph import AttentionGCNLayer
from opt_einsum import contract
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from long_seq import process_long_input
from losses import ATLoss
import json
from collections import defaultdict


class DocREModel(nn.Module):

    def __init__(self, config, model, tokenizer,device,
                emb_size=768, block_size=64, num_labels=-1,
                max_sent_num=25, evi_thresh=0.2, rel_evi_thresh = 0.3):
        '''
        Initialize the model.
        :model: Pretrained langage model encoder;
        :tokenizer: Tokenzier corresponding to the pretrained language model encoder;
        :emb_size: Dimension of embeddings for subject/object (head/tail) representations;
        :block_size: Number of blocks for grouped bilinear classification;
        :num_labels: Maximum number of relation labels for each entity pair;
        :max_sent_num: Maximum number of sentences for each document;
        :evi_thresh: Threshold for selecting evidence sentences.
        '''
        
        super().__init__()
        self.config = config
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.hidden_size = config.hidden_size
        


        self.edges = ['self-loop', 'pair-pair', 'pair-proto', 'proto-label','label-label']

        with open("./docred/meta/rel2id.json", "r") as f:
            raw_rel2id = json.load(f)
        self.rel2id = {rel: idx - 1 for rel, idx in raw_rel2id.items()}
        self.raw_rel2id = raw_rel2id
        with open("./dataset/meta/rel_info.json", "r") as f:
            self.rel_info = json.load(f)

        self.cluster_info = pd.read_csv("./dataset/docred/flattened_cluster_data.csv")
        
        
        with open("./docred/trigger_term_set.txt", "r", encoding="utf-8") as f:
            self.trigger_term_set = set(line.strip() for line in f if line.strip())
        
        self.proto_sent_df = pd.read_csv("./docred/max_proto_sents.csv")
        self.rel2proto = (
            self.proto_sent_df.groupby("Property")["Sentence"]
            .apply(lambda terms: set(t.lower() for t in terms))
            .to_dict()
        )

        self.relation_correlation = torch.load("./dataset/docred/label_correlation.pt", map_location=device)
        self.cached_label_label_edges = self.build_inter_prototype_label_edges(self.relation_correlation['B_tensor'],self.relation_correlation['relation_list'])
        
        self.interior_protos = torch.nn.ParameterDict({
            rel: torch.nn.Parameter(torch.empty(self.hidden_size)) for rel in self.rel2proto
        })
        for p in self.interior_protos.values():
            torch.nn.init.uniform_(p.data, a=-0.1, b=0.1)

        self.loss_fnt = ATLoss()
        self.loss_fnt_evi = nn.KLDivLoss(reduction="batchmean")

        self.head_extractor = nn.Linear(self.hidden_size * 3, emb_size)
        self.tail_extractor = nn.Linear(self.hidden_size * 3, emb_size)   

        self.na_head_extractor = nn.Linear(self.hidden_size * 2, emb_size)
        self.na_tail_extractor = nn.Linear(self.hidden_size * 2, emb_size)    

        self.bilinear = nn.Linear(emb_size * block_size, config.num_labels)
        self.na_classifier = nn.Linear(emb_size * block_size, 2)

        self.graph_layers = nn.ModuleList(
            AttentionGCNLayer(self.edges,self.hidden_size, nhead=2, iters=2) for _ in
            range(2))

        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels
        self.total_labels = config.num_labels
        self.rel_types = config.num_labels-1
        self.max_sent_num = max_sent_num
        self.evi_thresh = evi_thresh
        self.rel_evi_thresh = rel_evi_thresh

        

            

    def encode(self, input_ids, attention_mask):
        
        '''
        Get the embedding of each token. For long document that has more than 512 tokens, split it into two overlapping chunks.
        Inputs:
            :input_ids: (batch_size, doc_len)
            :attention_mask: (batch_size, doc_len)
        Outputs:
            :sequence_output: (batch_size, doc_len, hidden_dim)
            :attention: (batch_size, num_attn_heads, doc_len, doc_len)
        '''
        
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [self.tokenizer.cls_token_id]
            end_tokens = [self.tokenizer.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [self.tokenizer.cls_token_id]
            end_tokens = [self.tokenizer.sep_token_id, self.tokenizer.sep_token_id]
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
        
        return sequence_output, attention

    def get_hrt(self, sequence_output, attention, entity_pos, hts, offset):

        '''
        Get head, tail, context embeddings from token embeddings.
        Inputs:
            :sequence_output: (batch_size, doc_len, hidden_dim)
            :attention: (batch_size, num_attn_heads, doc_len, doc_len)
            :entity_pos: list of list. Outer length = batch size, inner length = number of entities each batch.
            :hts: list of list. Outer length = batch size, inner length = number of combination of entity pairs each batch.
            :offset: 1 for bert and roberta. Offset caused by [CLS] token.
        Outputs:
            :hss: (num_ent_pairs_all_batches, emb_size)
            :tss: (num_ent_pairs_all_batches, emb_size)
            :rss: (num_ent_pairs_all_batches, emb_size)
            :ht_atts: (num_ent_pairs_all_batches, doc_len)
            :rels_per_batch: list of length = batch size. Each entry represents the number of entity pairs of the batch.
        '''
        
        n, h, _, c = attention.size()
        hss, tss, rss = [], [], []
        ht_atts = []

        for i in range(len(entity_pos)): # for each batch
            entity_embs, entity_atts = [], []
            
            for eid, e in enumerate(entity_pos[i]): # for each entity
                if len(e) > 1:
                    e_emb, e_att = [], []
                    for mid, (start, end) in enumerate(e): # for every mention
                        if start + offset < c:
                            # In case the entity mention is truncated due to limited max seq length.
                            e_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset])

                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)

                entity_embs.append(e_emb)
                entity_atts.append(e_att)
                
            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]

            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)

            # obtain subject/object (head/tail) embeddings from entity embeddings.
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])
                
            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])

            ht_att = (h_att * t_att).mean(1) # average over all heads        
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-30) 
            ht_atts.append(ht_att)

            rs = contract("ld,rl->rd", sequence_output[i], ht_att)

            hss.append(hs)
            tss.append(ts)
            rss.append(rs)
        
        rels_per_batch = [len(b) for b in hss]
        hss = torch.cat(hss, dim=0) # (num_ent_pairs_all_batches, emb_size)
        tss = torch.cat(tss, dim=0) # (num_ent_pairs_all_batches, emb_size)
        rss = torch.cat(rss, dim=0) # (num_ent_pairs_all_batches, emb_size)
        ht_atts = torch.cat(ht_atts, dim=0) # (num_ent_pairs_all_batches, max_doc_len)

        return hss, rss, tss, ht_atts, rels_per_batch


    def forward_rel(self, hs, ts, rs,gs):
        '''
        Forward computation for RE.
        Inputs:
            :hs: (num_ent_pairs_all_batches, emb_size)
            :ts: (num_ent_pairs_all_batches, emb_size)
            :rs: (num_ent_pairs_all_batches, emb_size)
        Outputs:
            :logits: (num_ent_pairs_all_batches, num_rel_labels)
        '''
        
        hs = torch.tanh(self.head_extractor(torch.cat([hs,rs,gs], dim=-1)))
        ts = torch.tanh(self.tail_extractor(torch.cat([ts,rs,gs], dim=-1)))
        # split into several groups.
        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)

        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        logits = self.bilinear(bl)
        na_logits = self.na_classifier(bl)
        
        return logits, na_logits

    def forward_na_rel(self, hs, ts, rs):
        '''
        Forward computation for RE.
        Inputs:
            :hs: (num_ent_pairs_all_batches, emb_size)
            :ts: (num_ent_pairs_all_batches, emb_size)
            :rs: (num_ent_pairs_all_batches, emb_size)
        Outputs:
            :logits: (num_ent_pairs_all_batches, num_rel_labels)
        '''
        
        hs = torch.tanh(self.na_head_extractor(torch.cat([hs, rs], dim=-1)))
        ts = torch.tanh(self.na_tail_extractor(torch.cat([ts, rs], dim=-1)))
        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)

        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        na_logits = self.na_classifier(bl)
        
        return na_logits


    def forward_evi(self, doc_attn, sent_pos, batch_rel, offset):
        '''
        Forward computation for ER.
        Inputs:
            :doc_attn: (num_ent_pairs_all_batches, doc_len), attention weight of each token for computing localized context pooling.
            :sent_pos: list of list. The outer length = batch size. The inner list contains (start, end) position of each sentence in each batch.
            :batch_rel: list of length = batch size. Each entry represents the number of entity pairs of the batch.
            :offset: 1 for bert and roberta. Offset caused by [CLS] token.
        Outputs:
            :s_attn:  (num_ent_pairs_all_batches, max_sent_all_batch), sentence-level evidence distribution of each entity pair.
        '''
        
        max_sent_num = max([len(sent) for sent in sent_pos])
        rel_sent_attn = []
        for i in range(len(sent_pos)): # for each batch
            curr_attn = doc_attn[sum(batch_rel[:i]):sum(batch_rel[:i+1])]
            curr_sent_pos = [torch.arange(s[0], s[1]).to(curr_attn.device) + offset for s in sent_pos[i]] # + offset

            curr_attn_per_sent = [curr_attn.index_select(-1, sent) for sent in curr_sent_pos]
            curr_attn_per_sent += [torch.zeros_like(curr_attn_per_sent[0])] * (max_sent_num - len(curr_attn_per_sent))
            sum_attn = torch.stack([attn.sum(dim=-1) for attn in curr_attn_per_sent], dim=-1) 
            rel_sent_attn.append(sum_attn)

        s_attn = torch.cat(rel_sent_attn, dim=0)
        return s_attn


    def make_proto_label_emb_with_exterior_cluster(
        self, model, tokenizer, cluster_info, rel2proto, device, return_homo_nodes=True
    ):
        model = model.to(device).eval()

        sentences = cluster_info["sentence"].tolist()
        rels = sorted(cluster_info["rel"].unique().tolist())
        term2rel = cluster_info["rel"].to_list()
        term2cluster = cluster_info["cluster"].to_list()
        rel_texts = [self.rel_info[rel] if rel in self.rel_info else rel for rel in rels]
        all_texts = sentences + rel_texts

        BATCH_SIZE = 512
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(all_texts), BATCH_SIZE):
                batch = all_texts[i:i + BATCH_SIZE]
                encoded = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    return_attention_mask=True
                ).to(device)
                out = model(**encoded)

                hs = out.last_hidden_state  # [B, L, H]
                input_ids = encoded["input_ids"]
                attn_mask = encoded["attention_mask"].bool()
                cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
                mask = (input_ids != cls_id) & (input_ids != sep_id) & attn_mask
              
                masked_hs = hs.masked_fill(~mask.unsqueeze(-1), float("-inf"))
                batch_vecs, _ = masked_hs.max(dim=1)  # [B, H]

                all_embs.append(batch_vecs.cpu())
              


        all_embs = torch.cat(all_embs, dim=0)  # [T+R, H]
        term_embs = all_embs[:len(sentences)]
        relname_embs = all_embs[len(sentences):]

        cluster_to_indices = defaultdict(list)
        for idx, cl in enumerate(term2cluster):
            cluster_to_indices[cl].append(idx)

        rel2proto_clustered = defaultdict(lambda: {"border": []})
        homo_node_embs = []
        homo_labels = []

        for indices in cluster_to_indices.values():
            cluster_rels = set(term2rel[i] for i in indices)
            if len(cluster_rels) > 1:
                for rel in cluster_rels:
                    rel_terms = [i for i in indices if term2rel[i] == rel]
                    if rel_terms:
                        rel_embs = term_embs[rel_terms]
                        rel_mean = rel_embs.mean(dim=0)
                        rel2proto_clustered[rel]["border"].append(rel_mean)
            else:
                if return_homo_nodes:
                    rel = list(cluster_rels)[0]
                    for i in indices:
                        homo_node_embs.append(term_embs[i].unsqueeze(0))
                        rel_id = self.rel2id[rel] if hasattr(self, 'rel2id') and rel in self.rel2id else rels.index(rel)
                        homo_labels.append(rel_id)

        label_embs = {}
        for i, rel in enumerate(rels):
            if hasattr(self, "interior_protos") and rel in self.interior_protos:
                interior = self.interior_protos[rel]
            else:
                print("There is no interior cluster. :", rel)
                interior = torch.nn.Parameter(torch.zeros(term_embs.size(1), device=device))

            border = rel2proto_clustered[rel]["border"]
            label_embs[rel] = {
                "original_emb": relname_embs[i].to(device),
                "interior": interior,
                "border": [v.to(device) for v in border],
            }
      
        if return_homo_nodes:
            homo_node_embs = torch.cat(homo_node_embs, dim=0) if homo_node_embs else torch.empty(0, term_embs.size(1)).to(device)
            homo_labels = torch.tensor(homo_labels, dtype=torch.long, device=device)
         
            return label_embs, homo_node_embs, homo_labels

    
        return label_embs


    def create_graph_nodes_with_proto_labels(self, rs, label_embs):
        pair_feats = rs  # [P, D]

        proto_list = []
        label_node_map = []
        rel_ordered = sorted(
            [(rel, idx) for rel, idx in self.raw_rel2id.items() if rel != "Na"],
            key=lambda x: x[1]
        )
        ordered_rels = [rel for rel, _ in rel_ordered]

        for rel in ordered_rels:
            interior_emb = label_embs[rel]["interior"]
            proto_list.append(interior_emb)
            label_node_map.append((rel, "interior"))

            for b_emb in label_embs[rel]["border"]:
                proto_list.append(b_emb)
                label_node_map.append((rel, "border"))

        orig_label_list = []
        for rel in ordered_rels:
            orig_emb = label_embs[rel]["original_emb"]
            orig_label_list.append(orig_emb)
            label_node_map.append((rel, "original"))

        if len(proto_list) > 0:
            proto_feats = torch.stack(proto_list, dim=0)       # [T, D]
        else:
            proto_feats = torch.zeros(0, pair_feats.size(1), device=pair_feats.device)

        if len(orig_label_list) > 0:
            orig_feats = torch.stack(orig_label_list, dim=0)   # [R, D]
        else:
            orig_feats = torch.zeros(0, pair_feats.size(1), device=pair_feats.device)

        node_feats = torch.cat([pair_feats, proto_feats, orig_feats], dim=0)  # [P+T+R, D]

        num_pair_nodes = pair_feats.size(0)
        num_proto_nodes = len(proto_list)
        num_orig_nodes = len(orig_label_list)

        node_index_info = {
            "pair": (0, num_pair_nodes),
            "proto": (num_pair_nodes, num_proto_nodes),
            "original_label": (num_pair_nodes + num_proto_nodes, num_orig_nodes),
        }

        return node_feats, node_index_info, label_node_map




    def build_inter_prototype_label_edges(self, B_tensor, rel_list): 
        # Initialize dictionaries for proto to label and label to proto mappings
        label_edges = []
        for i in range(len(rel_list)):
            for j in range(len(rel_list)):
                if i != j and B_tensor[i, j] == 1:
                    label_edges.append((i, j))  # i → j
        return torch.tensor(label_edges, dtype=torch.long)  # [E, 2]

    def create_full_connected_edges(
            self,
            node_lengths,
            node_feats,
            hts,
            hts_sent_pos,
            node_index_info_list,
            label_node_map,
            na_logits = None,
            na_labels = None,
            unconfident_mask = None,
            confident_mask = None,
        ):
        B = len(node_lengths)
        N_max = max(node_lengths)
        edge_types = torch.zeros(B, N_max, N_max, dtype=torch.int64)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        total_TP, total_TN, total_FP, total_FN, total_pairs = 0, 0, 0, 0, 0

        for b in range(B):
            N = node_lengths[b]
            node_info = node_index_info_list[b]
            label_map = label_node_map

            P_start, P_len = node_info["pair"]
            L_start, L_len = node_info["proto"]
            OL_start, OL_len = node_info["original_label"]

            idx = torch.arange(N)
            edge_types[b, idx, idx] = 1


            hts_b = torch.tensor(hts[b], dtype=torch.long, device=device)
            E = torch.max(hts_b) + 1
            entity_pair_map = torch.zeros(E, P_len, dtype=torch.float, device=device)
            entity_pair_map[hts_b[:, 0], torch.arange(P_len)] = 1
            entity_pair_map[hts_b[:, 1], torch.arange(P_len)] = 1

            pair_pair_mask = (torch.matmul(entity_pair_map.T, entity_pair_map) > 0)
            pair_pair_mask.fill_diagonal_(False)

            edge_types[b, P_start:P_start+P_len, P_start:P_start+P_len][pair_pair_mask] = 2

            pair_vecs  = node_feats[b][P_start : P_start + P_len]
            label_vecs = node_feats[b][L_start : L_start + L_len]

            pair_vecs  = F.normalize(pair_vecs, dim=-1)
            label_vecs = F.normalize(label_vecs, dim=-1)

            pair_lengths = [info["pair"][1] for info in node_index_info_list]
            pair_offset = sum(pair_lengths[:b])
            Pair_len = node_index_info_list[b]["pair"][1]

            sim_matrix = torch.matmul(pair_vecs, label_vecs.T)
            threshold = torch.quantile(sim_matrix, 0.9).item()
            high_sim_mask = sim_matrix > threshold  

            k = min(3, label_vecs.size(0))
            topk_sim, topk_idx = torch.topk(sim_matrix, k=k, dim=-1)

            pair_idx  = torch.arange(P_len, device=device).unsqueeze(1).expand(-1, k)
            label_idx = L_start + topk_idx 

            
            pair_idx_flat  = P_start + pair_idx.reshape(-1)
            label_idx_flat = label_idx.reshape(-1) 



            na_logits_b = na_logits[pair_offset : pair_offset + Pair_len]  
            na_probs = F.softmax(na_logits_b, dim=-1)
            is_non_na = na_probs[:, 1] > 0.5                   
            is_na = ~is_non_na
            valid_indices = torch.nonzero(is_non_na, as_tuple=False).squeeze(-1)  
            unconfident_mask_b = unconfident_mask[pair_offset : pair_offset + P_len] 
            confident_mask_b = confident_mask[pair_offset : pair_offset + P_len]  #
            
            unconfident_alive = (unconfident_mask_b == 1)                       

            valid_mask =  unconfident_alive & is_non_na #& is_non_na
            valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)

            edge_types[b, pair_idx_flat, label_idx_flat] = 3
            edge_types[b, label_idx_flat, pair_idx_flat] = 3
 
            na_labels_b = na_labels[pair_offset : pair_offset + P_len].squeeze(-1)  
            entropy_mask_b = unconfident_mask[pair_offset : pair_offset + Pair_len]        
            pred_non_na = high_sim_mask.any(dim=1).long()  

            TP = ((na_labels_b == 1) & (pred_non_na == 1)).sum().item()  
            TN = ((na_labels_b == 0) & (pred_non_na == 0)).sum().item()  
            FP = ((na_labels_b == 0) & (pred_non_na == 1)).sum().item() 
            FN = ((na_labels_b == 1) & (pred_non_na == 0)).sum().item()  


            total = len(na_labels_b)
            acc = (TP + TN) / total if total > 0 else 0
            precision = TP / (TP + FP + 1e-8)
            recall = TP / (TP + FN + 1e-8)

            total_TP += TP
            total_TN += TN
            total_FP += FP
            total_FN += FN
            total_pairs += total

            rel2orig = {}
            for j in range(OL_len):
                rel, typ = label_map[L_len + j]
                if typ == "original":
                    rel2orig[rel] = OL_start + j

            rel_to_proto = defaultdict(list)
            proto_label_edges = []
            proto_proto_edges = []

            for j in range(L_len):
                rel, typ = label_map[j]
                if typ in ("interior", "border"):
                    proto_idx = L_start + j
                    rel_to_proto[rel].append(proto_idx)

                    if rel in rel2orig:
                        orig_idx = rel2orig[rel]
                        proto_label_edges.append((proto_idx, orig_idx))

            for proto_list in rel_to_proto.values():
                if len(proto_list) >= 2:
                    proto_tensor = torch.tensor(proto_list)
                    idx_pairs = torch.combinations(proto_tensor, r=2)
                    proto_proto_edges.extend(idx_pairs.tolist())

            for i, j in proto_label_edges:
                edge_types[b, i, j] = 4
                edge_types[b, j, i] = 4

            edges = self.cached_label_label_edges  # [E, 2]
            src = OL_start + edges[:, 0]
            dst = OL_start + edges[:, 1]

            edge_types[b, src, dst] = 5
            
        batch_metrics = {
        "TP": total_TP,
        "TN": total_TN,
        "FP": total_FP,
        "FN": total_FN,
        "total": total_pairs,
        }
        

        return edge_types, batch_metrics


    def create_graph(self, rs, label_embs, rels_per_batch, hts, hts_sent_pos,na_logits = None, na_labels=None,unconfident_mask = None,confident_mask = None):
        rs_per_batch = torch.split(rs, rels_per_batch, dim=0)

        node_feats = []
        label_node_map_list = []     
        node_index_info_list = []     

        for b in range(len(rs_per_batch)):
            node_feats_doc, node_index_info, label_node_map = self.create_graph_nodes_with_proto_labels(
                rs_per_batch[b], label_embs
            )
            node_feats.append(node_feats_doc)
            label_node_map_list.append(label_node_map)             
            node_index_info_list.append(node_index_info)             

        node_lengths = [nf.size(0) for nf in node_feats]

        batch_node_feats = pad_sequence(
            node_feats,
            batch_first=True,
            padding_value=0.0
        )
        B, max_node, D = batch_node_feats.shape

        graph_edges, edge_results = self.create_full_connected_edges(
            node_lengths=node_lengths,
            node_feats=node_feats,
            hts=hts,
            hts_sent_pos=hts_sent_pos,
            node_index_info_list=node_index_info_list,
            label_node_map= label_node_map,
            na_logits = na_logits,
            na_labels = na_labels,
            unconfident_mask = unconfident_mask,
            confident_mask = confident_mask
        )


        graph_fea = torch.zeros(B, max_node, D, device=rs.device)
        graph_adj = torch.zeros(B, max_node, max_node, dtype=torch.int32, device=rs.device)

        for i, nf in enumerate(node_feats):
            n_nodes = nf.size(0)
            graph_fea[i, :n_nodes, :] = nf
            graph_adj[i, :n_nodes, :n_nodes] = graph_edges[i, :n_nodes, :n_nodes]

        return graph_fea, graph_adj, label_node_map_list, node_index_info_list, edge_results

    
    def forward_graph(self, graph_feats, graph_adj, rels_per_batch, node_index_info_list, supervise_edge_type=None):
        """
        graph_feats: Tensor, (B, N, D)
        graph_adj:   Tensor or Sparse, (B, N, N)
        rels_per_batch: List[int], sum = total_pairs
        node_index_info_list: List[dict], 각 문서별 node index 정보
        supervise_edge_type: Optional Tensor (B, N, N)
        """
        h = graph_feats  # (B, N, D)
        for graph_layer in self.graph_layers:
            h, attn, supervised_attn = graph_layer(h, graph_adj, supervise_edge_type)  # (B, N, D)

        last_attn = attn.permute(1, 0, 2, 3)  # (B, H, N, N)
        attn_avg = last_attn.mean(dim=1)     # (B, N, N)

        outs = []
        sent_outs = []  
        rel_outs = []

        for i in range(len(rels_per_batch)):

            node_idx = node_index_info_list[i]

            pair_start, pair_len = node_idx["pair"]  
            outs.append(h[i, pair_start:pair_start + pair_len, :])  

            l_start, l_len = node_idx["proto"]
            rel_outs.append(h[i, l_start:l_start + l_len, :])   

        out = torch.cat(outs, dim=0)  # (sum(rels_per_batch), D)
        

        return out, attn_avg, supervised_attn,rel_outs

    def infer_sent_lens_from_s_attn(self, s_attn: torch.Tensor, pair_counts: list) -> list:
                    sent_lens = []
                    offset = 0
                    for count in pair_counts:
                        s_attn_doc = s_attn[offset : offset + count]  
                        sent_mask = (s_attn_doc.sum(dim=0) > 0)       
                        sent_lens.append(sent_mask.sum().item())
                        offset += count
                    return sent_lens

    def compute_masked_entropy(self,s_attn, sent_lens, pair_counts, eps=1e-9):
                    entropies = []
                    offset = 0
                    for sent_len, pair_count in zip(sent_lens, pair_counts):
                        s_attn_doc = s_attn[offset:offset + pair_count, :sent_len]  
                        attn = s_attn_doc / (s_attn_doc.sum(dim=-1, keepdim=True) + eps)
                        entropy = -(attn * (attn + eps).log()).sum(dim=-1) 
                        entropies.append(entropy)
                        offset += pair_count
                    return torch.cat(entropies, dim=0)  

    def mask_top_entropy_pairs(self,entropy: torch.Tensor, ratio: float = 0.2) -> torch.Tensor:
        P = entropy.size(0)
        k = int(P * ratio)

        topk_indices = torch.argsort(entropy, descending=True)[:k]
        mask = torch.ones(P, dtype=torch.long, device=entropy.device)
        mask[topk_indices] = 0

        return mask  
    
    def mask_under_entropy_pairs(self,entropy: torch.Tensor, ratio: float = 0.2) -> torch.Tensor:
        P = entropy.size(0)
        k = int(P * ratio)

        bottomk_indices = torch.argsort(entropy, descending=False)[:k]
        mask = torch.ones(P, dtype=torch.long, device=entropy.device)
        mask[bottomk_indices] = 0

        return mask  # (P,)
    
    def mix_with_top3_relations(self, gs_b, rel_outs_b):
        gs_new_list = []

        for b in range(len(gs_b)):
            gs = gs_b[b]         
            rel_outs = rel_outs_b[b] 
            P, D = gs.shape
            R, _ = rel_outs.shape
            sim = F.cosine_similarity(gs.unsqueeze(1), rel_outs.unsqueeze(0), dim=-1)  # (P, R)

            k = min(3, R)
            topk_vals, topk_idx = torch.topk(sim, k=k, dim=-1)

            top3_rels = torch.gather(
                rel_outs.unsqueeze(0).expand(P, R, D),
                1,
                topk_idx.unsqueeze(-1).expand(-1, -1, D)
            )  
            top3_mean = top3_rels.mean(dim=1)  
            gs_new = 0.5 * gs + 0.5 * top3_mean
            gs_new_list.append(gs_new)

        gs_new_flat = torch.cat(gs_new_list, dim=0)

        return gs_new_flat

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,             # relation labels
        entity_pos=None,
        hts=None,                # entity pairs
        sent_pos=None,
        hts_sent_pos = None,
        sent_labels=None,        # evidence labels (0/1)
        teacher_attns=None,      # evidence distribution from teacher model
        token_labels=None,
        trigger_sent_labels = None,
        na_labels = None,
        graph=True,
        tag="train",
    ):
        trigger_sent_labels = None
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        output = {}

      
        sequence_output, attention = self.encode(input_ids, attention_mask)
        
        # --------------------------------------------------
      
        label_embs, homo_node_embs, homo_labels = self.make_proto_label_emb_with_exterior_cluster(
            self.model, self.tokenizer, self.cluster_info,
            self.rel2proto, sequence_output.device
            )

        hs, rs, ts, doc_attn, batch_rel = self.get_hrt(
            sequence_output, attention, entity_pos, hts, offset
            )
        
        na_logits = self.forward_na_rel(hs.detach(), ts.detach(), rs.detach())
        probs = F.softmax(na_logits, dim=-1)

        s_attn = self.forward_evi(doc_attn, sent_pos, batch_rel, offset)
        sent_lens = self.infer_sent_lens_from_s_attn(s_attn, batch_rel)
        entropy = self.compute_masked_entropy(s_attn, sent_lens, batch_rel)

        unconfident_entropy_mask = self.mask_top_entropy_pairs(entropy, ratio=0.7)
        confident_entropy_mask = self.mask_under_entropy_pairs(entropy,ratio=0.2)

        if graph:
            graph_feats, graph_adj, label_node_map, node_index_info_list, edge_results = self.create_graph(rs, label_embs, batch_rel, hts,hts_sent_pos,na_logits, na_labels, unconfident_entropy_mask,confident_entropy_mask)
            output['edge_results'] = edge_results
            if trigger_sent_labels is not None:
                gs, g_attn, supervised_attn, rel_outs = self.forward_graph(graph_feats, graph_adj, batch_rel,node_index_info_list)
                
                supervised_attn = supervised_attn.mean(dim=0)
            else:
                gs, g_attn, _ , rel_outs= self.forward_graph(graph_feats, graph_adj,batch_rel,node_index_info_list)
    
        logits, dont_use_na_logits = self.forward_rel(hs, ts, rs, gs)
        output["rel_pred"] = self.loss_fnt.get_label(logits, num_labels=self.num_labels)
          
        if sent_labels is not None:
            sent_labels_tf = (sent_labels > 0).float()   
            cur_len = sent_labels_tf.size(1)
            pad_len = self.max_sent_num - cur_len

            if pad_len > 0:
                sent_labels_tf = F.pad(sent_labels_tf, (0, pad_len))  
            elif pad_len < 0:
                sent_labels_tf = sent_labels_tf[:, :self.max_sent_num]  

            rel_pred = output["rel_pred"]              
            sent_labels_bool = sent_labels_tf.bool()   
            rel_logits = logits
            probs = F.softmax(rel_logits, dim=-1)          

            top2_vals, top2_idx = torch.topk(probs, k=2, dim=-1)
            margin = top2_vals[:, 0] - top2_vals[:, 1]
            raw_top2_vals, raw_top2_idx = torch.topk(rel_logits, k=2, dim=-1)
            raw_margin = raw_top2_vals[:, 0] - raw_top2_vals[:, 1]

            is_na = (rel_pred[..., 0] == 1) & (rel_pred[..., 1:].sum(dim=-1) == 0)

            c1 = top2_idx[:, 0]
            c2 = top2_idx[:, 1]

            na_delta = 0.5  
            non_na_delta = 0.5

            confident_na = (c1 == 0) & (c2 != 0) & (margin > na_delta)
            confident_rel = (c1 != 0) & (c2 == 0) & (margin > non_na_delta)

            is_confident = confident_na

            false_mask = torch.zeros_like(sent_labels_bool[0], dtype=torch.bool)  
            
            evi_pred = F.pad(
                s_attn > self.evi_thresh,
                (0, self.max_sent_num - s_attn.shape[-1])
            ) 
            output["evi_pred_ori"] = evi_pred.clone()
            evi_pred[is_confident] = false_mask
            

            output["evi_pred"] = evi_pred      
            output["confident_non_na"] = confident_rel.long()  
            output["confident_na"] = confident_na.long()         
            output['margin'] = raw_margin


        if tag in ["test", "dev"]:
            scores_topk = self.loss_fnt.get_score(logits, self.num_labels)
            output["scores"] = scores_topk[0] 
            output["topks"] = scores_topk[1] 

        if tag == "infer": # teacher model inference
            output["attns"] = doc_attn.split(batch_rel)

        else:
            loss = self.loss_fnt(logits.float(), labels.float())
            output["loss"] = {"rel_loss": loss.to(sequence_output)}

            x = homo_node_embs
            y = homo_labels.to(torch.long).to(x.device)

            interior_weight = torch.stack(
                [self.interior_protos[rel] for rel in self.rel2proto.keys()],
                dim=0
            )
            proto_logits = F.linear(x, interior_weight.to(x.device))
            proto_cls_loss = F.cross_entropy(proto_logits, y)
            output["loss"]["proto_cls_loss"] = proto_cls_loss.to(sequence_output)

            if sent_labels != None:
                sent_labels = sent_labels.float()
                
                is_na = (labels[:, 0] == 1)  
                is_non_na = ~is_na

                na_indices = torch.nonzero(is_na).view(-1)
                non_na_indices = torch.nonzero(is_non_na).view(-1)

                num_non_na = len(non_na_indices)
                num_sample = min(len(na_indices), num_non_na)

                sampled_na_indices = na_indices[torch.randperm(len(na_indices))[:num_sample]]

                idx_used = torch.cat([sampled_na_indices, non_na_indices], dim=0)
                idx_used = idx_used[torch.randperm(len(idx_used))]  # 섞기
                s_attn = s_attn[idx_used]
                sent_labels = sent_labels[idx_used]
                norm_s_labels = sent_labels/(sent_labels.sum(dim=-1, keepdim=True) + 1e-30)
                norm_s_labels[norm_s_labels == 0] = 1e-30
                norm_s_labels = norm_s_labels.float()

                s_attn[s_attn == 0] = 1e-30
                evi_loss = self.loss_fnt_evi(s_attn.log(), norm_s_labels)
                
                output["loss"]["evi_loss"] = evi_loss.to(sequence_output)

            if na_labels is not None:
                pos_weight = torch.tensor([1.0, 40.0]).to(sequence_output)  

                criterion = nn.CrossEntropyLoss(weight=pos_weight)
                na_loss= criterion(na_logits, na_labels)
                output["loss"]["na_loss"] = na_loss.to(sequence_output)

            elif teacher_attns != None:
                doc_attn[doc_attn == 0] = 1e-30
                teacher_attns[teacher_attns == 0] = 1e-30
                attn_loss = self.loss_fnt_evi(doc_attn.log(), teacher_attns)
                output["loss"]["attn_loss"] = attn_loss.to(sequence_output)

        return output
