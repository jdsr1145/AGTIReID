from prettytable import PrettyTable
import torch
import numpy as np
import os
import torch.nn.functional as F
import logging
 
from prettytable import PrettyTable
import torch
import numpy as np
import os
import torch.nn.functional as F
import logging


def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        # acclerate sort with topk
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )  # q * topk
    pred_labels = g_pids[indices.cpu()]  # q * k
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

    all_cmc = matches[:, :max_rank].cumsum(1) # cumulative sum
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    # all_cmc = all_cmc[topk - 1]

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k

    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices

def get_metrics(similarity, qids, gids, n_, retur_indices=False):
    t2i_cmc, t2i_mAP, t2i_mINP, indices = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
    t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()
    if retur_indices:
        return [n_, t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP, t2i_cmc[0]+ t2i_cmc[4]+ t2i_cmc[9]], indices
    else:
        return [n_, t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP, t2i_cmc[0]+ t2i_cmc[4]+ t2i_cmc[9]]


class Evaluator():
    def __init__(self, img_loader, txt_loader, text_txt_loader_aerial=None):
        self.img_loader = img_loader # gallery
        self.txt_loader = txt_loader # query
        self.txt_loader_aerial = text_txt_loader_aerial
        self.logger = logging.getLogger("RDE.eval")

    def _compute_embedding(self, model):
        model = model.eval()
        device = next(model.parameters()).device

        qids, gids, qfeats, gfeats = [], [], [], []
        qfeats_reexpress_a = []
        return_dict = {}
        # text
        # print(f"Query dataloader length: {len(self.txt_loader)}", flush=True)
        for pid, caption in self.txt_loader:
            caption = caption.to(device)
            with torch.no_grad():
                text_feat = model.encode_text(caption).cpu()
            qids.append(pid.view(-1)) # flatten 
            qfeats.append(text_feat)
        qids = torch.cat(qids, 0)
        qfeats = torch.cat(qfeats, 0)
        return_dict.update({'qfeats': qfeats, 'qids': qids})

        if model.args.enable_reranking:
            for pis, caption in self.txt_loader:
                caption = caption.to(device)
                with torch.no_grad():
                    # TODO: Optimize logic
                    text_feat, _ = model.base_model.encode_text(caption)
                    text_feat_reexpress_a, _ = model.reexpress_text(text_feat)
                    text_feat_reexpress_a = text_feat_reexpress_a[torch.arange(text_feat_reexpress_a.shape[0]), caption.argmax(dim=-1)].float().cpu()
                qfeats_reexpress_a.append(text_feat_reexpress_a)
            qfeats_reexpress_a = torch.cat(qfeats_reexpress_a, 0)
            return_dict.update({'qfeats_reexpress_a': qfeats_reexpress_a})

        # image
        for pid, img in self.img_loader:
            img = img.to(device)
            with torch.no_grad():
                img_feat = model.encode_image(img).cpu()
            gids.append(pid.view(-1)) # flatten 
            gfeats.append(img_feat)
        gids = torch.cat(gids, 0)
        gfeats = torch.cat(gfeats, 0)
        return_dict.update({'qfeats':qfeats, 'gfeats':gfeats, 'qids':qids, 'gids':gids})

        if self.txt_loader_aerial is not None:
            gfeats_aerial = []
            for pid, caption in self.txt_loader_aerial:
                caption = caption.to(device)
                with torch.no_grad():
                    text_feat = model.encode_text(caption).cpu()
                gfeats_aerial.append(text_feat)
            gfeats_aerial = torch.cat(gfeats_aerial, 0)
            return_dict.update({'gfeats_aerial': gfeats_aerial})

        for k, v in return_dict.items():
            v = v.cpu()
            return_dict.update({k:v})
        return return_dict
    
    def _compute_embedding_tse(self, model):
        model = model.eval() 
        device = next(model.parameters()).device

        qids, gids, qfeats, gfeats = [], [], [], []
        # text
        for pid, caption in self.txt_loader:
            caption = caption.to(device)
            with torch.no_grad():
                text_feat = model.encode_text_tse(caption).cpu()
            qids.append(pid.view(-1)) # flatten 
            qfeats.append(text_feat)
        qids = torch.cat(qids, 0)
        qfeats = torch.cat(qfeats, 0)

        # image
        for pid, img in self.img_loader:
            img = img.to(device)
            with torch.no_grad():
                img_feat = model.encode_image_tse(img).cpu()
            gids.append(pid.view(-1)) # flatten 
            gfeats.append(img_feat)
        gids = torch.cat(gids, 0)
        gfeats = torch.cat(gfeats, 0) 
        return qfeats.cpu(), gfeats.cpu(), qids.cpu(), gids.cpu()
    
    def eval(self, model, i2t_metric=False):
        return_dict_bge = self._compute_embedding(model)
        qfeats, gfeats = return_dict_bge['qfeats'], return_dict_bge['gfeats']
        qids, gids = return_dict_bge['qids'], return_dict_bge['gids']
        if model.args.enable_reranking:
            qfeats_reexpress_a = return_dict_bge['qfeats_reexpress_a']
            qfeats_reexpress_a = F.normalize(qfeats_reexpress_a, p=2, dim=1)
        sims_dict = {}
        qfeats = F.normalize(qfeats, p=2, dim=1) # text features
        gfeats = F.normalize(gfeats, p=2, dim=1) # image features
        # print("DEBUG: shape", qfeats.shape, gfeats.shape)
        sims_bse = qfeats @ gfeats.t()

        sims_dict.update({'BGE': sims_bse})
        if model.args.enable_text_aerial:
            gfeats_aerial = return_dict_bge['gfeats_aerial']
            gfeats_aerial = F.normalize(gfeats_aerial, p=2, dim=1)
            print("DEBUG: shape", qfeats.shape, gfeats_aerial.shape)
            sims_text_aerial = qfeats @ gfeats_aerial.t()
            sims_dict.update({'Text-Aerial': sims_text_aerial})
            sims_dict.update({'BGE+Text-Aerial': (sims_bse + sims_text_aerial)/2})
        if model.args.enable_tse:
            # TODO: optimize computation logic
            vq_feats, vg_feats, _, _ = self._compute_embedding_tse(model)
            vq_feats = F.normalize(vq_feats, p=2, dim=1) # text features
            vg_feats = F.normalize(vg_feats, p=2, dim=1) # image features
            sims_tse = vq_feats@vg_feats.t()

            sims_dict.update({'TSE': sims_tse})
            sims_dict.update({'BGE+TSE': (sims_bse + sims_tse)/2})
        
        if model.args.enable_reranking:
            sims_reexpress_a = qfeats_reexpress_a @ gfeats.t()
            sims_dict.update({'Re-express': sims_reexpress_a})
            sims_dict.update({'BGE+Re-express': (sims_bse + sims_reexpress_a)/2})
            if model.args.enable_tse:
                sims_dict.update({'BGE+Re-express-A+TSE': (sims_bse+sims_reexpress_a + sims_tse)/3})

        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP","rSum"])
        
        for key in sims_dict.keys():
            sims = sims_dict[key]
            rs = get_metrics(sims, qids, gids, f'{key}-t2i',False)
            table.add_row(rs)
            if i2t_metric:
                i2t_cmc, i2t_mAP, i2t_mINP, _ = rank(similarity=sims.t(), q_pids=gids, g_pids=qids, max_rank=10, get_mAP=True)
                i2t_cmc, i2t_mAP, i2t_mINP = i2t_cmc.numpy(), i2t_mAP.numpy(), i2t_mINP.numpy()
                table.add_row(['i2t', i2t_cmc[0], i2t_cmc[4], i2t_cmc[9], i2t_mAP, i2t_mINP])

        table.custom_format["R1"] = lambda f, v: f"{v:.2f}"
        table.custom_format["R5"] = lambda f, v: f"{v:.2f}"
        table.custom_format["R10"] = lambda f, v: f"{v:.2f}"
        table.custom_format["mAP"] = lambda f, v: f"{v:.2f}"
        table.custom_format["mINP"] = lambda f, v: f"{v:.2f}"
        table.custom_format["RSum"] = lambda f, v: f"{v:.2f}"
        self.logger.info('\n' + str(table))
        
        return rs[1]
