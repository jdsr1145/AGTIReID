import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_rbs(dict_feats_loss, pid, label_hat=None, tau=0.02, margin=0.1, loss_type='TAL', logit_scale=50, enable_loss_view=False):
    dict_loss = {}
    views = dict_feats_loss['views'] if 'views' in dict_feats_loss else None
    # print("DEBUG: views in loss", views)
    # BGE
    list_lb = []
    num=0
    i_feats = dict_feats_loss['i_feats']
    t_feats = dict_feats_loss['t_feats']
    loss_bge_gtgi, _ = compute_per_loss(i_feats, t_feats, pid, tau, margin, loss_type, logit_scale)
    list_lb.append(loss_bge_gtgi)
    num+=1
    # Set as tensor
    if 'i_feats_a_dualstream' in dict_feats_loss and 't_feats_a_dualstream' in dict_feats_loss:
        i_feats_a = dict_feats_loss['i_feats_a_dualstream']
        t_feats_a = dict_feats_loss['t_feats_a_dualstream']
        loss_bge_gtai, _ = compute_per_loss(i_feats_a, t_feats, pid, tau, margin, loss_type, logit_scale)
        list_lb.append(loss_bge_gtai)
        num+=1
        for item in t_feats_a:
            loss_bge_gtat, _ = compute_per_loss(i_feats, item, pid, tau, margin, loss_type, logit_scale)
            num+=1
            list_lb.append(loss_bge_gtat)
            # list_lb.append(loss_bge_gtat*(1/len(t_feats_a)))
        # if len(t_feats_a) >0:
            # num+=1
        # print("DEBUG: loss_bge_3")
    loss_bge = torch.zeros_like(loss_bge_gtgi)
    # print("DEBUG: list_lb length", len(list_lb))
    for item in list_lb:
        loss_bge += item
    loss_bge = loss_bge / num
    dict_loss.update({'loss_bge':loss_bge})
    
    # TSE
    if 'i_feats_tse' in dict_feats_loss and 't_feats_tse' in dict_feats_loss:
        list_lt = []
        i_tse_f = dict_feats_loss['i_feats_tse']
        t_tse_f = dict_feats_loss['t_feats_tse']
        loss_tse_gtgi, _ = compute_per_loss(i_tse_f, t_tse_f, pid, tau, margin, loss_type, logit_scale)
        list_lt.append(loss_tse_gtgi)
        if 'i_feats_tse_a_dualstream' in dict_feats_loss and 't_feats_tse_a_dualstream' in dict_feats_loss:
            i_tse_f_a = dict_feats_loss['i_feats_tse_a_dualstream']
            t_tse_f_a = dict_feats_loss['t_feats_tse_a_dualstream']
            loss_tse_gtai, _ = compute_per_loss(i_tse_f_a, t_tse_f, pid, tau, margin, loss_type, logit_scale)
            list_lt.append(loss_tse_gtai)
            # TODO: correct logic
            loss_tse_gtat, _ = compute_per_loss(t_tse_f, t_tse_f_a, pid, tau, margin, loss_type, logit_scale)
            list_lt.append(loss_tse_gtat)
        # print("DEBUG: list_lt length", len(list_lt))
        loss_tse = torch.zeros_like(loss_tse_gtgi)
        for item in list_lt:
            loss_tse += item
        loss_tse = loss_tse / len(list_lt)
        dict_loss.update({'loss_tse':loss_tse})

    # Re-Express
    # if 't_feats_a' in dict_feats_loss and 't_feats_g' in dict_feats_loss:
    if 't_feats_re' in dict_feats_loss:
        # t_feats_a = dict_feats_loss['t_feats_a']
        # t_feats_g = dict_feats_loss['t_feats_g']
        t_feats_reexpress = dict_feats_loss['t_feats_re']
        # t_feats_reexpress = []
        # for i, view in enumerate(views):
        #     if i==1:
        #         t_feats_reexpress.append(t_feats_a[i])
        #     else:
        #         t_feats_reexpress.append(t_feats_g[i])
        # t_feats_reexpress = torch.stack(t_feats_reexpress, dim=0)
        # TODO: Optimize loss?
        loss_reexpress, _ = compute_per_loss(i_feats, t_feats_reexpress, pid, tau, margin, loss_type, logit_scale)
        dict_loss.update({'loss_reexpress':loss_reexpress})

        if enable_loss_view:
            i_feats_vdt = dict_feats_loss['i_feats_view']
            loss_view, _ = compute_per_loss(i_feats_vdt, t_feats_reexpress, pid, tau, margin, loss_type, logit_scale)
            dict_loss.update({'loss_view':loss_view})

        if 'logits_reexpress_a' in dict_feats_loss and 'logits_reexpress_g' in dict_feats_loss:
            logits_reexpress_a = dict_feats_loss['logits_reexpress_a']
            logits_reexpress_g = dict_feats_loss['logits_reexpress_g']
            id_a = torch.full((logits_reexpress_a.shape[0],), 1, 
                      dtype=torch.long, device=logits_reexpress_a.device)
            id_g = torch.full((logits_reexpress_g.shape[0],), 0, 
                      dtype=torch.long, device=logits_reexpress_g.device)
            loss_classifier = (F.cross_entropy(logits_reexpress_a, id_a) + F.cross_entropy(logits_reexpress_g, id_g))/2
            dict_loss.update({'loss_classifier':loss_classifier})

    # VDT
    # TODO: Paired up orthogonal?
    if 'i_feats_view' in dict_feats_loss:
        i_feats_vdt = dict_feats_loss['i_feats_view']
        logits_view = dict_feats_loss['logits_view']

        # Orthogonal
        i_feats_vdt_or = i_feats_vdt / i_feats_vdt.norm(dim=-1, keepdim=True)
        i_feats_or = i_feats / i_feats.norm(dim=-1, keepdim=True)
        loss_orthogonal = torch.cosine_similarity(i_feats_vdt_or, i_feats_or).abs().mean()
        dict_loss.update({'loss_orthogonal':loss_orthogonal})

        # CE
        loss_ce = F.cross_entropy(logits_view, views)
        dict_loss.update({'loss_ce':loss_ce})

    for k,v in dict_loss.items():
        # print(k)
        dict_loss[k] = (label_hat * v).sum()

    return dict_loss

def compute_per_loss(image_features, text_features, pid, tau=0.02, margin=0.2, loss_type='TAL', logit_scale=50):
    
    # # normalized features
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)
    scores = text_norm @ image_norm.t()

    if 'TAL' in loss_type:
        per_loss = compute_TAL_per(scores, pid, tau, margin=margin)
    elif 'TRL' in loss_type:
        per_loss = compute_TRL_per(scores, pid, tau=tau, margin=margin)
    elif 'InfoNCE' in loss_type:
        per_loss = compute_InfoNCE_per(scores, logit_scale)
    elif 'SDM' in loss_type:
        per_loss = compute_sdm_per(scores, pid, logit_scale)
    else:
        exit()

    return per_loss, scores.diag()
 
def compute_sdm_per(scores, pid, logit_scale, epsilon=1e-8):
    """
    Similarity Distribution Matching
    """
    batch_size = scores.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    t2i_cosine_theta = scores
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.sum(i2t_loss, dim=1) + torch.sum(t2i_loss, dim=1)

    return loss

def compute_TRL_per(scores, pid, margin = 0.2, tau=0.02):       
    batch_size = scores.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float().cuda()
    mask = 1 - labels

    alpha_1 =((scores/tau).exp()* labels / ((scores/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()
    alpha_2 = ((scores.t()/tau).exp()* labels / ((scores.t()/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()

    pos_1 = (alpha_1 * scores).sum(1)
    pos_2 = (alpha_2 * scores.t()).sum(1)

    neg_1 = (mask*scores).max(1)[0]
    neg_2 = (mask*scores.t()).max(1)[0]

    cost_1 = (margin + neg_1 - pos_1).clamp(min=0)
    cost_2 = (margin + neg_2 - pos_2).clamp(min=0)
    return cost_1 + cost_2

 
def compute_InfoNCE_per(scores, logit_scale):
    
    # cosine similarity as logits
    logits_per_image = logit_scale * scores
    logits_per_text = logits_per_image.t()

    p1 = F.softmax(logits_per_image, dim=1)
    p2 = F.softmax(logits_per_text, dim=1)

    loss = (- p1.diag().log() - p2.diag().log())/2    
    return loss

def compute_TAL_per(scores, pid, tau, margin):
    batch_size = scores.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float().cuda()
    mask = 1 - labels

    alpha_i2t =((scores/tau).exp()* labels / ((scores/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()
    alpha_t2i = ((scores.t()/tau).exp()* labels / ((scores.t()/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()

    loss = (-  (alpha_i2t*scores).sum(1) + tau * ((scores / tau).exp() * mask).sum(1).clamp(max=10e35).log() + margin).clamp(min=0)  \
        +  (-  (alpha_t2i*scores.t()).sum(1) + tau * ((scores.t() / tau).exp() * mask).sum(1).clamp(max=10e35).log() + margin).clamp(min=0)
    
    return loss # Shape: [batch_size]
