from model import objectives
from model import stn

from .CrossEmbeddingLayer_tse import TexualEmbeddingLayer, VisualEmbeddingLayer
from .clip_model import build_CLIP_from_openai_pretrained, convert_weights
import torch
import torch.nn as nn 
import torch.nn.functional as F
from copy import deepcopy

def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

class RDE(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size, enable_vdt=args.enable_vdt)
        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = torch.ones([]) * (1 / args.temperature)

        if args.enable_tse:
            self.visul_emb_layer = VisualEmbeddingLayer(ratio=args.select_ratio)
            self.texual_emb_layer = TexualEmbeddingLayer(ratio=args.select_ratio)

        if args.enable_vdt:
            # TODO: discrete classifier
            self.view_classifier = deepcopy(self.base_model.reexpress_classification)

        if 'TAL' in self.current_task:
            loss_type = 'TAL'
        elif 'TRL' in self.current_task:
            loss_type = 'TRL'
        elif 'InfoNCE' in self.current_task:
            loss_type = 'InfoNCE'
        elif 'SDM' in self.current_task:
            loss_type = 'SDM'
        else:
            exit()
        self.loss_type = loss_type
 
    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')
    
    def encode_image(self, image):
        x, _ = self.base_model.encode_image(image)
        return x[:, 0, :].float()
      
    def encode_text(self, text):
        x, _ = self.base_model.encode_text(text.long())
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def encode_image_tse(self, image):
        x,atten_i = self.base_model.encode_image(image)
        i_tse_f = self.visul_emb_layer(x, atten_i)   
        return i_tse_f.float()
 
    def encode_text_tse(self, text):
        x,atten_t = self.base_model.encode_text(text.long())
        t_tse_f = self.texual_emb_layer(x, text, atten_t)
        return t_tse_f.float()

    def reexpress_text(self, text_feats, view):
        # t_re_a= self.base_model.reexpress_text([text_feats], self.base_model.prompt_a.to(text_feats.device).type(self.base_model.dtype))
        # t_re_g= self.base_model.reexpress_text([text_feats], self.base_model.prompt_g.to(text_feats.device).type(self.base_model.dtype))
        t_re = self.base_model.reexpress_text([text_feats], view)
        return t_re

    def compute_per_loss(self, batch):
        feat_dict_loss = {}
        images = batch['images']
        caption_ids = batch['caption_ids']
        input_dict = {'images':images, 'caption_ids':caption_ids}
        if self.args.dualstream:
            images_a = batch['images_a']
            caption_ids_a = batch['caption_ids_a']
            input_dict.update({'images_a':images_a, 'caption_ids_a':caption_ids_a})

        # TODO: Reexpress loss for CCD?
        # views = batch['views'] if 'views' in batch else None
        features_dict = self.base_model(input_dict)
        image_feats = features_dict['image_features'] # Projected to embed_dim
        text_feats = features_dict['text_features']
        atten_i = features_dict['atten_i']
        atten_t = features_dict['atten_t']
        i_feats = image_feats[:, 0, :].float()
        # i_feats = image_feats.float() # for CLIP ResNet visual model
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
        feat_dict_loss.update({'i_feats':i_feats, 't_feats':t_feats, 'atten_i':atten_i, 'atten_t':atten_t})
        if self.args.dualstream:
            image_feats_a = features_dict['image_features_a']
            text_feats_a = features_dict['text_features_a']
            atten_ia = features_dict['atten_ia']
            atten_ta = features_dict['atten_ta']
            t_feats_a = []
            i_feats_a = image_feats_a[:, 0, :].float()
            for item in text_feats_a:
                t_feats_a.append(item[torch.arange(item.shape[0]), caption_ids.argmax(dim=-1)].float())
            feat_dict_loss.update({'i_feats_a_dualstream':i_feats_a, 't_feats_a_dualstream':t_feats_a, 'atten_ia':atten_ia, 'atten_ta':atten_ta})

        list_la = []
        lossA, _= objectives.compute_per_loss(i_feats, t_feats, batch['pids'], \
                                                    tau=self.args.tau, \
                                                    margin=self.args.margin, \
                                                    loss_type=self.loss_type, \
                                                    logit_scale=self.logit_scale)
        list_la.append(lossA)
        if self.args.dualstream:
            loss_A_gtai, _ = objectives.compute_per_loss(i_feats_a, t_feats, batch['pids'], \
                                                        tau=self.args.tau, \
                                                        margin=self.args.margin, \
                                                        loss_type=self.loss_type, \
                                                        logit_scale=self.logit_scale)
            list_la.append(loss_A_gtai)

            for item in t_feats_a:
                loss_A_gtat, _ = objectives.compute_per_loss(t_feats, item, batch['pids'], \
                                                        tau=self.args.tau, \
                                                        margin=self.args.margin, \
                                                        loss_type=self.loss_type, \
                                                        logit_scale=self.logit_scale)   
                list_la.append(loss_A_gtat)
        lossA_temp = torch.zeros_like(lossA)
        for item in list_la:
            lossA_temp += item
        lossA = lossA_temp / len(list_la)
        
        ret_dict = {}
        ret_dict.update({'lossA': lossA.detach().cpu()})
        if self.args.enable_tse:
            list_lb = []
            i_tse_f = self.visul_emb_layer(image_feats, atten_i)
            t_tse_f = self.texual_emb_layer(text_feats, caption_ids, atten_t)
            lossB, _ = objectives.compute_per_loss(i_tse_f, t_tse_f, batch['pids'],\
                                                    tau=self.args.tau, \
                                                    margin=self.args.margin, \
                                                    loss_type=self.loss_type, \
                                                    logit_scale=self.logit_scale)
            list_lb.append(lossB)
            if self.args.dualstream:
                i_tse_f_a = self.visul_emb_layer(image_feats_a, atten_ia)
                t_tse_f_a = self.texual_emb_layer(text_feats_a, caption_ids, atten_ta)
                loss_B_gtai, _ = objectives.compute_per_loss(i_tse_f_a, t_tse_f, batch['pids'],\
                                                        tau=self.args.tau, \
                                                        margin=self.args.margin, \
                                                        loss_type=self.loss_type, \
                                                        logit_scale=self.logit_scale)
                list_lb.append(loss_B_gtai)
                for item in t_tse_f_a:
                    loss_B_gtat, _ = objectives.compute_per_loss(t_tse_f, item, batch['pids'],\
                                                        tau=self.args.tau, \
                                                        margin=self.args.margin, \
                                                        loss_type=self.loss_type, \
                                                        logit_scale=self.logit_scale)
                    list_lb.append(loss_B_gtat)
            lossB_temp = torch.zeros_like(lossB)
            for item in list_lb:
                lossB_temp += item
            lossB = lossB_temp / len(list_lb)
            ret_dict.update({'lossB': lossB.detach().cpu()})

        return ret_dict

    def forward(self, batch):
        ret = dict()
        input_dict = {}
        ret.update({'temperature': 1 / self.logit_scale})

        images = batch['images']
        caption_ids = batch['caption_ids'] # [B, 77]
        cam_ids = batch['cam_ids']
        input_dict = {'images':images, 'caption_ids':caption_ids}
        views = batch['views'] if 'views' in batch else None
        # print("DEBUG: views", views)
        dict_feat_loss = {}
        
        dict_feat_loss.update({'views':views}) if views is not None else None
        features_dict = self.base_model(input_dict) # Projected to embed_dim
        image_feats = features_dict['image_features']
        text_feats = features_dict['text_features']
        atten_i = features_dict['atten_i']
        atten_t = features_dict['atten_t']
        # TODO: Integrate with TSE and CCD
        if self.args.enable_vdt:
            view_feats = image_feats[:, 1, :]
            logits_view = self.view_classifier(view_feats.type(self.base_model.dtype))
            dict_feat_loss.update({'i_feats_view':view_feats.float()})
            dict_feat_loss.update({'logits_view':logits_view})

        i_feats = image_feats[:, 0, :].float()
        # i_feats = image_feats.float() # for CLIP ResNet visual model
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
        dict_feat_loss.update({'i_feats':i_feats})
        dict_feat_loss.update({'t_feats':t_feats})

        if self.args.dualstream:
            image_feats_a = features_dict['image_features_a']
            text_feats_a = features_dict['text_features_a']
            atten_ia = features_dict['atten_ia']
            atten_ta = features_dict['atten_ta']
            i_feats_a = image_feats_a[:, 0, :].float()
            t_feats_a = []
            for item in text_feats_a:
                t_feats_a.append(item[torch.arange(item.shape[0]), caption_ids.argmax(dim=-1)].float())
            dict_feat_loss.update({'i_feats_a_dualstream':i_feats_a})
            dict_feat_loss.update({'t_feats_a_dualstream':t_feats_a})

        if self.args.enable_reexpress:
            # TODO: Integrate with CCD?
            # TODO: Integrate with TSE?
            text_feats_re = self.reexpress_text(text_feats, cam_ids) # TODO
            t_feats_re = text_feats_re[torch.arange(text_feats_re.shape[0]), caption_ids.argmax(dim=-1)].float()
            # t_feats_a = t_feats_a[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
            # t_feats_g = t_feats_g[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
            # dict_feat_loss.update({'t_feats_a':t_feats_a})
            # dict_feat_loss.update({'t_feats_g':t_feats_g})
            dict_feat_loss.update({'t_feats_re': t_feats_re})

            if self.args.enable_calssifier:
                logits_reexpress_a = self.base_model.reexpress_classify(t_feats_a.type(self.base_model.dtype))
                logits_reexpress_g = self.base_model.reexpress_classify(t_feats_g.type(self.base_model.dtype))
                dict_feat_loss.update({'logits_reexpress_a':logits_reexpress_a})
                dict_feat_loss.update({'logits_reexpress_g':logits_reexpress_g})

        if self.args.enable_tse:
            i_tse_f = self.visul_emb_layer(image_feats, atten_i)
            t_tse_f = self.texual_emb_layer(text_feats, caption_ids, atten_t)
            dict_feat_loss.update({'i_feats_tse':i_tse_f})
            dict_feat_loss.update({'t_feats_tse':t_tse_f})
            if self.args.dualstream:
                i_tse_f_a = self.visul_emb_layer(image_feats_a, atten_ia)
                t_tse_f_a = []
                for i in range(len(text_feats_a)):
                    # TODO: correct logic
                    t_tse_f_a.append(self.texual_emb_layer(text_feats_a[i], caption_ids, atten_ta))
                t_tse_f_a = self.texual_emb_layer(text_feats_a, caption_ids, atten_ta)
                dict_feat_loss.update({'i_feats_tse_a_dualstream':i_tse_f_a})
                dict_feat_loss.update({'t_feats_tse_a_dualstream':t_tse_f_a})
            
        label_hat = batch['label_hat'].to(i_feats.device) 
     
        dict_losses = objectives.compute_rbs(dict_feat_loss, batch['pids'], \
                                              label_hat=label_hat, margin=self.args.margin,tau=self.args.tau,\
                                                loss_type=self.loss_type,logit_scale=self.logit_scale, enable_loss_view=self.args.enable_loss_view,)
        for k,v in dict_losses.items():
            if self.args.enable_reexpress and self.args.scale_reexpress and 'loss_reexpress' in k:
                v = v * self.args.scale_reexpress
            if self.args.enable_calssifier and self.args.scale_calssifier and 'loss_classifier' in k:
                v = v * self.args.scale_calssifier
            if self.args.enable_vdt and self.args.scale_vdt and ('loss_orthogonal' in k or 'loss_ce' in k):
                v = v * self.args.scale_vdt
            if self.args.enable_loss_view and self.args.scale_loss_view and 'loss_view' in k:
                v = v * self.args.scale_loss_view
            ret.update({k:v})

        return ret


def build_model(args, num_classes=11003):
    model = RDE(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    stn = stn.build_stn()
    return model, stn
