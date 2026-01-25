import math
import torch
import torch.nn.functional as F
import numpy as np
import os


def rand(prune_num, important_indices, h_vis, h_text):
    mi_proxy = torch.rand(h_vis.shape[0])
    # prune_idx = torch.topk(mi_proxy, k=prune_num, largest=False).indices
    prune_idx = torch.argsort(mi_proxy, descending=False)[:prune_num]
    return prune_idx


def cosine_mean(prune_num, important_indices, h_vis, h_text):
    sim = h_vis @ h_text.T  # [260, 10]
    mi_proxy = sim.mean(dim=1)  # avg_similarity -> abs(sim)??

    prune_idx = torch.argsort(mi_proxy, descending=False)[:prune_num]
    # prune_idx = torch.topk(mi_proxy, k=prune_num, largest=False).indices
    return prune_idx

def cosine_mean_trim(prune_num, important_indices, h_vis, h_text):
    sim = h_vis @ h_text.T  # [260, 10]
    sim_trim = torch.topk(sim, k=5, dim=-1).values  # only consider top5-highest-sim token
    mi_proxy = sim_trim.mean(dim=1)  # avg_similarity -> abs(sim)??
    prune_idx = torch.argsort(mi_proxy, descending=False)[:prune_num]
    return prune_idx

def cosine_maxmin(prune_num, important_indices, h_vis, h_text):
    sim = h_vis @ h_text.T  # [260, 10]
    mi_proxy_max = sim.max(dim=1).values  # max_similarity
    mi_proxy_min = sim.min(dim=1).values   # min
    round_1 = prune_num // 2
    round_2 = prune_num - round_1

    prune_idx_1 = torch.argsort(mi_proxy_max, descending=False)[:round_1]
    mi_proxy_min[prune_idx_1] = torch.inf # not considered any more
    prune_idx_2 = torch.argsort(mi_proxy_min, descending=False)[:round_2]
    prune_idx = torch.cat([prune_idx_1, prune_idx_2], dim=0)
    return prune_idx


def cosine_minmax(prune_num, important_indices, h_vis, h_text):
    sim = h_vis @ h_text.T  # [260, 10]
    keep_mask = torch.ones(h_vis.shape[0], device=h_vis.device, dtype=torch.bool)
    mi_proxy_max = sim.max(dim=1).values  # max_similarity
    mi_proxy_min = sim.min(dim=1).values   # min


    round_1 = prune_num // 2
    round_2 = prune_num - round_1

    prune_idx_1 = torch.argsort(mi_proxy_min, descending=False)[:round_1]
    mi_proxy_max[prune_idx_1] = torch.inf # not considered any more
    prune_idx_2 = torch.argsort(mi_proxy_max, descending=False)[:round_2]
    prune_idx = torch.cat([prune_idx_1, prune_idx_2], dim=0)
    return prune_idx

def cosine_max(prune_num, important_indices, h_vis, h_text):
    sim = h_vis @ h_text.T  # [260, 10]
    mi_proxy_max = sim.max(dim=1).values  # max_similarity
    prune_idx = torch.argsort(mi_proxy_max, descending=False)[:prune_num]
    return prune_idx

def cosine_min(prune_num, important_indices, h_vis, h_text):
    sim = h_vis @ h_text.T  # [260, 10]
    mi_proxy_min = sim.min(dim=1).values  # min_similarity
    prune_idx = torch.argsort(mi_proxy_min, descending=False)[:prune_num]
    return prune_idx

def entropy_v(prune_num, important_indices, h_vis, h_text):
    temperature = STATION["temperature"]
    sim = (h_vis @ h_text.T)/ temperature # sharper

    # conditional p(text | patch)
    pt_on_pv = F.softmax(sim, dim=-1)  # [576, 16]

    # entropy H(text | patch)
    mi_proxy = torch.sum(pt_on_pv * torch.log(pt_on_pv + 1e-8), dim=-1)  # -entropy: [576] for the whole sentence
    prune_idx = torch.argsort(mi_proxy, descending=False)[:prune_num]
    return prune_idx

def mi_mean(prune_num, important_indices, h_vis, h_text):
    temperature = STATION["temperature"]
    sim = (h_vis @ h_text.T)/ temperature # sharper
    pt_on_pv = F.softmax(sim, dim=-1)           # conditional p(text | patch)  [576, 16]
    p_t = pt_on_pv.mean(dim=0, keepdim=True)   # t_prior p(avg_text | patch) [1, 16]

    # MI score
    mi_per_token = pt_on_pv * (torch.log(pt_on_pv + 1e-8) - torch.log(p_t + 1e-8)) # [576, 16]
    mi_proxy = torch.mean(mi_per_token,dim=-1)  # [576]
    prune_idx = torch.argsort(mi_proxy, descending=False)[:prune_num]
    return prune_idx

def mi_mean_trim(prune_num, important_indices, h_vis, h_text):
    temperature = STATION["temperature"]
    sim = (h_vis @ h_text.T)/ temperature # sharper
    pt_on_pv = F.softmax(sim, dim=-1)           # conditional p(text | patch)  [576, 16]
    p_t = pt_on_pv.mean(dim=0, keepdim=True)   # t_prior p(avg_text | patch) [1, 16]

    # MI score
    mi_per_token = pt_on_pv * (torch.log(pt_on_pv + 1e-8) - torch.log(p_t + 1e-8))
    mi_per_token_trim = torch.topk(mi_per_token, k=3, dim=-1).values # only consider top5-highest-mi token
    mi_proxy = torch.mean(mi_per_token_trim,dim=-1)
    prune_idx = torch.argsort(mi_proxy, descending=False)[:prune_num]
    return prune_idx

def mi_max(prune_num, important_indices, h_vis, h_text):
    temperature = STATION["temperature"]
    sim = (h_vis @ h_text.T)/ temperature # sharper
    pt_on_pv = F.softmax(sim, dim=-1)           # conditional p(text | patch)  [576, 16]
    p_t = pt_on_pv.mean(dim=0, keepdim=True)   # t_prior p(avg_text | patch) [1, 16]

    # MI score
    mi_per_token = pt_on_pv * (torch.log(pt_on_pv + 1e-8) - torch.log(p_t + 1e-8))
    mi_proxy = mi_per_token.max(dim=-1).values
    cls_idx = torch.arange(0, h_vis.shape[0], 257, device=h_vis.device)
    mi_proxy[cls_idx] = torch.inf  # [cls]: not considered
    prune_idx = torch.argsort(mi_proxy, descending=False)[:prune_num]

    return prune_idx


def mi_max_full(prune_num, important_indices, h_vis, h_text):
    temperature = STATION["temperature"]
    vis_len = h_vis.shape[0]
    weight = 1
    keep_round1 = 5
    ## CLS ##
    cls_idx = torch.arange(0, vis_len, 257, device=h_vis.device)
    is_cls = torch.zeros(vis_len, dtype=torch.bool, device=h_vis.device)
    is_cls[cls_idx] = True
    valid_indices = torch.where(~is_cls)[0]

    # 实际可参与选择的 token 数
    valid_len = vis_len - cls_idx.numel()
    keep_total = valid_len - prune_num

    # p(t|v)
    sim_vt = (h_vis @ h_text.T) / temperature
    pt_on_pv = F.softmax(sim_vt, dim=-1)

    # p(v|v)
    sim_vv = (h_vis @ h_vis.T) / temperature
    pv_on_pv = F.softmax(sim_vv, dim=-1)

    p_t = pt_on_pv.mean(dim=0, keepdim=True)
    p_v = torch.tensor(1.0 / vis_len, device=h_vis.device)

    # PMI
    pmi_vt_mat = torch.log(pt_on_pv + 1e-8) - torch.log(p_t + 1e-8)
    pmi_vv_mat = torch.log(pv_on_pv + 1e-8) - torch.log(p_v + 1e-8)


    important_indices = torch.full(
        (keep_total,), -1, dtype=torch.long, device=h_vis.device
    )

    # --------------------
    # round 1: relevance (MI_vt), top-k
    # --------------------
    relevance = pmi_vt_mat[valid_indices].max(dim=-1).values
    topk = torch.argsort(relevance, descending=True)[:keep_round1]
    important_indices[:keep_round1] = valid_indices[topk]

    # --------------------
    # round 2: greedy
    # --------------------
    for counter in range(keep_round1, keep_total):
        mask = torch.ones(vis_len, dtype=torch.bool, device=h_vis.device)
        mask[important_indices[:counter]] = False
        mask[cls_idx] = False  # ❗ CLS 永远不参与 greedy

        remain_indices = torch.where(mask)[0]
        selected_indices = important_indices[:counter]

        relevance = pmi_vt_mat[remain_indices].max(dim=-1).values
        redundancy = pmi_vv_mat[remain_indices][:, selected_indices].max(dim=-1).values

        score = relevance - weight * redundancy
        important_indices[counter] = remain_indices[torch.argmax(score)]

    # --------------------
    # final prune index
    # --------------------
    final_mask = torch.ones(vis_len, dtype=torch.bool, device=h_vis.device)
    final_mask[important_indices] = False
    final_mask[cls_idx] = False  # keep  CLS

    prune_idx = torch.where(final_mask)[0]
    return prune_idx


def mi_max_round2(prune_num, important_indices, h_vis, h_text, ):
    temperature = STATION["temperature"]
    sim = (h_vis @ h_text.T)/ temperature # sharper
    pt_on_pv = F.softmax(sim, dim=-1)           # conditional p(text | patch)  [576, 16]
    p_t = pt_on_pv.mean(dim=0, keepdim=True)   # t_prior p(avg_text | patch) [1, 16]

    # MI score
    mi_per_token = pt_on_pv * (torch.log(pt_on_pv + 1e-8) - torch.log(p_t + 1e-8))
    mi_proxy = mi_per_token.max(dim=-1).values
    mi_proxy[important_indices] = torch.inf ## round1: vis_attn
    prune_idx = torch.argsort(mi_proxy, descending=False)[:prune_num]
    return prune_idx

def mi_minmax(prune_num, important_indices, h_vis, h_text):
    round_1 = prune_num // 2
    round_2 = prune_num - round_1
    temperature = STATION["temperature"]

    sim = (h_vis @ h_text.T)/ temperature # sharper
    pt_on_pv = F.softmax(sim, dim=-1)           # conditional p(text | patch)  [576, 16]
    p_t = pt_on_pv.mean(dim=0, keepdim=True)   # t_prior p(avg_text | patch) [1, 16]

    # MI score
    mi_per_token = pt_on_pv * (torch.log(pt_on_pv + 1e-8) - torch.log(p_t + 1e-8))
    mi_proxy_max = mi_per_token.max(dim=-1).values
    mi_proxy_min = mi_per_token.min(dim=-1).values
    prune_idx_1 = torch.argsort(mi_proxy_max, descending=False)[:round_1]
    mi_proxy_min[prune_idx_1] = torch.inf # not considered any more
    prune_idx_2 = torch.argsort(mi_proxy_min, descending=False)[:round_2]
    prune_idx = torch.cat([prune_idx_1, prune_idx_2], dim=0)
    return prune_idx

# similarity_measure
SIM_FUNC = {"w_o": None,
           "rand": rand,
           "cos_mean": cosine_mean,
           "cos_mean_trim": cosine_mean_trim,
           "cos_max": cosine_max,
           "cos_min": cosine_min,
           "cos_maxmin": cosine_maxmin,
           "cos_minmax": cosine_minmax,
           "mi_max": mi_max,
            "mi_max_full":mi_max_full,
           "mi_minmax": mi_minmax,
           "mi_mean": mi_mean,
           "mi_mean_trim": mi_mean_trim,
           "vis_pruner_mi": mi_max_round2,
            }

## "_trim" consider only the top5 tokens, which is for long questions.
STATION={"modify": "w_o",
         "temperature": 0.1}
