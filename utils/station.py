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
    # import pdb;pdb.set_trace()
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
    # print("cls_idx: ", cls_idx)
    mi_proxy[cls_idx] = torch.inf  # [cls]: not considered
    prune_idx = torch.argsort(mi_proxy, descending=False)[:prune_num]
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
           "mi_minmax": mi_minmax,
           "mi_mean": mi_mean,
           "mi_mean_trim": mi_mean_trim,
           "vis_pruner_mi": mi_max_round2,
            }

## "_trim" consider only the top5 tokens, which is for long questions.
STATION={"modify": "w_o",
         "temperature": 0.1}
