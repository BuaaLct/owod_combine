from optparse import Values
from turtle import forward
from detectron2.utils.store import Store
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from .sinkhorn_knopp import SinkhornKnopp
from detectron2.utils.events import get_event_storage

class Prototypes(nn.Module):
    def __init__(self, output_dim, num_prototypes):
        """
        num_prototypes unlabel 类别的数量
        """
        super().__init__()

        self.prototypes = nn.Linear(output_dim, num_prototypes, bias=False)

    @torch.no_grad()
    def normalize_prototypes(self):
        w = self.prototypes.weight.data.clone()
        w = F.normalize(w, dim=1, p=2)
        self.prototypes.weight.copy_(w)

    def forward(self, x):
        return self.prototypes(x)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers=1):
        """
        """
        super().__init__()

        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_hidden_layers - 1):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
            ]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class CosMLP(nn.Module):
    def __init__(self, input_dim_list, hidden_dim, output_dim, num_classes, clip_process, seen_classes):

        super().__init__()
        num_hidden_layers = [2,2]

        self.feature_mlps = nn.ModuleList([
            MLP(input_dim, hidden_dim, output_dim, num_hidden_layer) for 
            input_dim , num_hidden_layer in zip(input_dim_list,num_hidden_layers)
        ])

        # self.cos_mlp = MLP(num_classes,num_classes,num_classes,1)

        self.close_set_cls_mlp = MLP(output_dim,output_dim,1,1)
        self.uno_cls_mlp = MLP(output_dim,output_dim,1,1)
        self.num_classes = num_classes
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.seen_classes = seen_classes
        self.clip_process = clip_process
        self.cos_mlp = MLP(num_classes,num_classes,num_classes,1)

        self.init_sa_matrix()

    def init_sa_matrix(self):
        device = 'cpu'
        sa_features = self.clip_process.get_text_features(device)
        known_sa_features = sa_features[:self.seen_classes]
        bg_sa_features = sa_features[-1:,:]
        unk_sa_features = self.clip_process.get_super_features(device)
        sa_concat_features = torch.cat([known_sa_features,unk_sa_features,bg_sa_features]).to(device)
        num_unknown = unk_sa_features.shape[0]
        self.sa_concat_features = sa_concat_features

        self.sa_matrix = self.cos(sa_concat_features[:,None,:].expand(-1,num_unknown,-1),unk_sa_features[None,...].expand(self.num_classes,-1,-1))

    def forward(self,input_feeatures,sa_features,fg_gt_classes):
        # 点乘
        proposal_num = input_feeatures.shape[0]

        input_adjust_features = self.feature_mlps[0](input_feeatures)
        sa_features = self.feature_mlps[1](self.sa_concat_features.to(input_feeatures.device))
        
        sa_features = sa_features[None,:].expand(proposal_num,-1,-1)
        input_adjust_features = input_adjust_features[:,None,:].expand_as(sa_features)


        # shape is num_proposal*self.num_classes*feat_dim
        dot_multiply_visual_semantic = torch.mul(input_adjust_features,sa_features) 

        # split to known_bg and unk sa tensor
        close_set_dot_multiply_visual_semantic = torch.cat([dot_multiply_visual_semantic[:,:self.seen_classes],
        dot_multiply_visual_semantic[:,-1:]],dim=1)
        unk_dot_multiply_visual_semantic = dot_multiply_visual_semantic[:,self.seen_classes:-1]
            
        close_set_logits = self.close_set_cls_mlp(close_set_dot_multiply_visual_semantic.flatten(0,1))
        uno_logits = self.uno_cls_mlp(unk_dot_multiply_visual_semantic.flatten(0,1))

        # squeeze the last dim and view the logits tensor
        close_set_logits=close_set_logits.squeeze(-1).view(proposal_num,-1) # proposal_num*(self.seen_classes+1)
        uno_logits = uno_logits.squeeze(-1).view(proposal_num,-1) # proposal_num*10

        # adj_cos_sim_to_unk = adj_cos_sim[:,self.seen_classes:-1]
        sa_matrix = self.sa_matrix.to(input_adjust_features.device)
        targets = sa_matrix[fg_gt_classes,:]
       
        targets[(fg_gt_classes >= self.seen_classes)] = (uno_logits[(fg_gt_classes >= self.seen_classes)]>=0.8).to(torch.float32)

        loss_uno = F.binary_cross_entropy_with_logits(
            uno_logits.reshape(-1),
            targets.reshape(-1),
            reduction="mean",
        )

        known_mask = fg_gt_classes < self.seen_classes

        known_adj_cos_sim = close_set_logits[known_mask,:]

        loss_known = F.cross_entropy(known_adj_cos_sim, fg_gt_classes[known_mask], reduction="mean")
        
        return loss_uno + 0.2*loss_known
    
    def inference(self,input_feeatures,sa_features,scores):
        proposal_num = input_feeatures.shape[0]

        input_adjust_features = self.feature_mlps[0](input_feeatures)
        sa_features = self.feature_mlps[1](self.sa_concat_features.to(input_feeatures.device))
        
        sa_features = sa_features[None,:].expand(proposal_num,-1,-1)
        input_adjust_features = input_adjust_features[:,None,:].expand_as(sa_features)

        # shape is num_proposal*self.num_classes*feat_dim
        dot_multiply_visual_semantic = torch.mul(input_adjust_features,sa_features) 

        # split to known_bg and unk sa tensor
        # close_set_dot_multiply_visual_semantic = torch.cat([dot_multiply_visual_semantic[:,:self.seen_classes],
        # dot_multiply_visual_semantic[:,-1:]],dim=1)
        unk_dot_multiply_visual_semantic = dot_multiply_visual_semantic[:,self.seen_classes:-1]
            
        all_logits = self.close_set_cls_mlp(dot_multiply_visual_semantic.flatten(0,1))
        uno_logits = self.uno_cls_mlp(unk_dot_multiply_visual_semantic.flatten(0,1))

        # squeeze the last dim and view the logits tensor
        all_logits=all_logits.squeeze(-1).view(proposal_num,-1) # proposal_num*(self.num_classes)
        uno_logits = uno_logits.squeeze(-1).view(proposal_num,-1) # proposal_num*10

        sa_matrix = self.sa_matrix.to(input_feeatures.device)
        # print(all_logits.shape,sa_matrix.t().shape)
        final_score = torch.mul(torch.matmul(uno_logits,sa_matrix.t()),all_logits)

        # print(scores.shape, final_score.shape)
        scores[:,:self.seen_classes] = final_score[:,:self.seen_classes]
        scores[:,80] = final_score[:,self.seen_classes:-1].max(dim=-1)[0]
        scores[:,-1] = final_score[:,-1]

        return scores  
        # self.sa_matrix[self.seen_classes:-1,:] = torch.eye(num_unknown)
    
    def forward_cos(self,input_feeatures,sa_features,fg_gt_classes):
        # 余弦相似度
        input_adjust_features = self.feature_mlps[0](input_feeatures)
        sa_adjust_features = self.feature_mlps[1](sa_features)
        proposal_num = input_adjust_features.shape[0]
        input_adjust_features = input_adjust_features[:,None,:].expand(-1,self.num_classes,-1)
        sa_adjust_features = sa_adjust_features[None,...].expand(proposal_num,-1,-1)
        
        cos_sim = self.cos(input_adjust_features,sa_adjust_features)
        adj_cos_sim = self.cos_mlp(cos_sim)
        
        adj_cos_sim_to_unk = adj_cos_sim[:,self.seen_classes:-1]
        sa_matrix = self.sa_matrix.to(input_adjust_features.device)
        # print(fg_gt_classes.shape,set(fg_gt_classes.tolist()))
        targets = sa_matrix[fg_gt_classes,:]
       
        targets[(fg_gt_classes >= self.seen_classes)] = (adj_cos_sim_to_unk[(fg_gt_classes >= self.seen_classes)]>=0.8).to(torch.float32)

        loss_total = F.binary_cross_entropy_with_logits(
            adj_cos_sim_to_unk.reshape(-1),
            targets.reshape(-1),
            reduction="mean",
        )
        
        known_mask = fg_gt_classes < self.seen_classes

        known_adj_cos_sim = adj_cos_sim[known_mask,:]
        # print(known_adj_cos_sim.shape, fg_gt_classes[known_mask].shape)
        loss_known = F.cross_entropy(known_adj_cos_sim, fg_gt_classes[known_mask], reduction="mean")

        return loss_total + 0.2*loss_known
    # def forward_1(self,input_feeatures,sa_features,fg_gt_classes):
        
    #     proposal_num = input_feeatures.shape[0]

    #     input_adjust_features = self.feature_mlps[0](input_feeatures)
    #     center_sa_adjust_features = self.feature_mlps[1](sa_features)

    #     sa_features = self.feature_mlps[1](self.sa_concat_features.to(input_feeatures.device))
    #     sa_features = sa_features[None,:].expand(proposal_num,-1,-1)
    #     input_adjust_features

    #     sa_adjust_features = center_sa_adjust_features[fg_gt_classes,:]

    #     dot_multiply_visual_semantic = torch.mul(input_adjust_features,sa_adjust_features) 

    #     all_cls_logits = self.all_cls_mlp(dot_multiply_visual_semantic)
        
    #     uno_cls_logits = self.uno_cls_mlp(dot_multiply_visual_semantic)

    #     # adj_cos_sim_to_unk = adj_cos_sim[:,self.seen_classes:-1]
    #     sa_matrix = self.sa_matrix.to(input_adjust_features.device)
    #     # print(fg_gt_classes.shape,set(fg_gt_classes.tolist()))
    #     targets = sa_matrix[fg_gt_classes,:]
       
    #     targets[(fg_gt_classes >= self.seen_classes)] = (uno_cls_logits[(fg_gt_classes >= self.seen_classes)]>=0.8).to(torch.float32)

    #     loss_uno = F.binary_cross_entropy_with_logits(
    #         uno_cls_logits.reshape(-1),
    #         targets.reshape(-1),
    #         reduction="mean",
    #     )

    #     known_mask = fg_gt_classes < self.seen_classes

    #     known_adj_cos_sim = all_cls_logits[known_mask,:]
    #     # print(known_adj_cos_sim.shape, fg_gt_classes[known_mask].shape)
    #     loss_known = F.cross_entropy(known_adj_cos_sim, fg_gt_classes[known_mask], reduction="mean")
        
    #     return loss_uno + 0.2*loss_known


class MultiHead(nn.Module):
    def __init__(
            self, input_dim, hidden_dim, output_dim, num_prototypes, num_heads, num_hidden_layers=1
    ):
        super().__init__()
        self.num_heads = num_heads

        # projectors
        self.projectors = torch.nn.ModuleList(
            [MLP(input_dim, hidden_dim, output_dim, num_hidden_layers) for _ in range(num_heads)]
        )

        # prototypes
        self.prototypes = torch.nn.ModuleList(
            [Prototypes(output_dim, num_prototypes) for _ in range(num_heads)]
        )
        self.normalize_prototypes()

    @torch.no_grad()
    def normalize_prototypes(self):
        for p in self.prototypes:
            p.normalize_prototypes()

    def forward_head(self, head_idx, feats):
        """
        head_idx 指定是第几个head
        """
        z = self.projectors[head_idx](feats)
        z = F.normalize(z, dim=1)
        return self.prototypes[head_idx](z), z

    def forward(self, feats):
        out = [self.forward_head(h, feats) for h in range(self.num_heads)]
        return [torch.stack(o) for o in map(list, zip(*out))]


class MultiHeadResNet(nn.Module):
    def __init__(
            self,
            num_labeled,
            num_unlabeled,
            hidden_dim=2048,
            proj_dim=256,
            overcluster_factor=3,
            num_heads=1,
            num_hidden_layers=1,
            feat_dim=2048
    ):
        super().__init__()

        # self.unknown_store = Store(num_unlabeled,20)
        # todo 2048
        self.feat_dim = feat_dim

        # head for label
        self.head_lab = Prototypes(self.feat_dim, num_labeled)

        if num_heads is not None:
            # head for unlabel
            self.head_unlab = MultiHead(
                input_dim=self.feat_dim,
                hidden_dim=hidden_dim,
                output_dim=proj_dim,
                num_prototypes=num_unlabeled,
                num_heads=num_heads,
                num_hidden_layers=num_hidden_layers,
            )

            # head for unlabeled overcluster
            self.head_unlab_over = MultiHead(
                input_dim=self.feat_dim,
                hidden_dim=hidden_dim,
                output_dim=proj_dim,

                num_prototypes=num_unlabeled * overcluster_factor,
                num_heads=num_heads,
                num_hidden_layers=num_hidden_layers,
            )

        self.sk = SinkhornKnopp(
            num_iters=3, epsilon=0.05
        )
        self.seen_class = num_labeled
        self.clustering_start_iter = 1000
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.temperature = 1.5
        self.num_classes = 81
        self.seen_classes = 20

    @torch.no_grad()
    def _reinit_all_layers(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @torch.no_grad()
    def normalize_prototypes(self):
        self.head_lab.normalize_prototypes()
        if getattr(self, "head_unlab", False):
            self.head_unlab.normalize_prototypes()
            self.head_unlab_over.normalize_prototypes()

    # @torch.no_grad()
    def forward_scores(self, feats):
        logits_lab=self.head_lab(F.normalize(feats))
        logits_unlab, _ = self.head_unlab(feats)
        max_value,_=logits_unlab[0].max(dim=-1)
        logits_unlab_max=max_value[:,None]
        scores=torch.cat((logits_lab[:,:-1],logits_unlab_max,logits_lab[:,-1:]),dim=-1)

        return scores


    def forward_heads(self, feats):
        out = {"logits_lab": self.head_lab(F.normalize(feats))}
        if hasattr(self, "head_unlab"):
            logits_unlab, proj_feats_unlab = self.head_unlab(feats)
            logits_unlab_over, proj_feats_unlab_over = self.head_unlab_over(feats)
            out.update(
                {
                    "logits_unlab": logits_unlab,
                    "proj_feats_unlab": proj_feats_unlab,
                    "logits_unlab_over": logits_unlab_over,
                    "proj_feats_unlab_over": proj_feats_unlab_over,
                }
            )
        return out

    def forward(self, feats):
        if isinstance(feats, list):
            out = [self.forward_heads(f) for f in feats]
            out_dict = {"feats": torch.stack(feats)}
            for key in out[0].keys():
                out_dict[key] = torch.stack([o[key] for o in out])
            return out_dict
        else:
            out = self.forward_heads(feats)
            out["feats"] = feats
            return out

    def cross_entropy_loss(self, preds, targets):
        preds = F.log_softmax(preds / 0.1, dim=-1)
        return torch.mean(-torch.sum(targets * preds, dim=-1), dim=-1)

    def swapped_prediction(self, logits, targets):
        loss = 0
        for view in range(2):
            for other_view in np.delete(range(2), view):
                loss += self.cross_entropy_loss(logits[other_view], targets[view])
        return loss / 2

    def get_pair_loss_by_this_iter(self, logits, mask_lab,feats ,labels,nlc):
        
        storage = get_event_storage()
        loss_pair = 0 
        if storage.iter > self.clustering_start_iter:
            
            save_features = feats[0]
            unk_features = save_features[labels == self.num_classes-1]

            logits_unlab = logits[0,0,~mask_lab,nlc:]
            unknwon_num=unk_features.shape[0]
            feature_dim=unk_features.shape[1]
            unknwon_max_logits=torch.zeros((unknwon_num,unknwon_num),device=logits_unlab.device)

            for i in range(unknwon_num):
                for j in range(unknwon_num):
                    unknwon_max_logits[i,j]=torch.max(logits_unlab[i]*logits_unlab[j])
            unknwon_max_logits=unknwon_max_logits.view(-1)
            data1=unk_features[:,None,:].expand(-1,unknwon_num,-1).reshape(-1,feature_dim) 
            data2=unk_features[None,...].expand(unknwon_num,-1,-1).reshape(-1,feature_dim)
            cos_sim=self.cos(data1,data2)
            unknwon_targets=(cos_sim>=0.8).to(torch.float32)
            loss_pair = F.binary_cross_entropy_with_logits(
                unknwon_max_logits,
                unknwon_targets,
                reduction="mean",
            )

        return loss_pair
    
    def get_pair_loss_by_ten_iter(self, logits, mask_lab,feats ,labels,nlc):
        
        loss_pair = 0

        storage = get_event_storage()

        logits_unlab = logits[0,0,~mask_lab,nlc:]
        
        if storage.iter == self.clustering_start_iter:
            self.last_ten_iter_unk_features = torch.zeros((0,feats[0].shape[1]),device=feats[0].device)
             
        elif storage.iter > self.clustering_start_iter:
            
            save_features = feats[0]
            unk_features = save_features[labels == self.num_classes-1]
            
            self.last_ten_iter_unk_features = torch.cat((self.last_ten_iter_unk_features,unk_features),dim=0)
            
            if (storage.iter - self.clustering_start_iter) % 10==0:
                unknwon_num=self.last_ten_iter_unk_features.shape[0]
                feature_dim=self.last_ten_iter_unk_features.shape[1]
                unknwon_max_logits=torch.zeros((unknwon_num,unknwon_num),device=logits_unlab.device)
                
                last_ten_iter_scores = self(self.last_ten_iter_unk_features)['logits_unlab'][0]
                
                for i in range(unknwon_num):
                    for j in range(unknwon_num):
                        unknwon_max_logits[i,j]=torch.max(last_ten_iter_scores[i]*last_ten_iter_scores[j])
                unknwon_max_logits=unknwon_max_logits.view(-1)
                data1=self.last_ten_iter_unk_features[:,None,:].expand(-1,unknwon_num,-1).reshape(-1,feature_dim) 
                data2=self.last_ten_iter_unk_features[None,...].expand(unknwon_num,-1,-1).reshape(-1,feature_dim)
                cos_sim=self.cos(data1,data2)
                unknwon_targets=(cos_sim>=0.8).to(torch.float32)
                
                loss_pair=F.binary_cross_entropy_with_logits(
                    unknwon_max_logits,
                    unknwon_targets,
                    reduction="mean",
                )
                self.last_ten_iter_unk_features = torch.zeros((0,feature_dim),device=feats[0].device)
        
        return loss_pair    

    def get_cluster_loss_by_feature_angle(self, feats ,labels ):
        
        loss_uniform=0

        storage = get_event_storage()
        
        if storage.iter == self.clustering_start_iter:
            pass
            
        elif storage.iter > self.clustering_start_iter:
            
            save_features = feats[0]
            adj_features = save_features / torch.norm(save_features,dim=1,keepdim=True)
            
            angle_matrix = torch.matmul(adj_features,adj_features.t())
            flag_matrix = (labels[:,None] == labels[None,:]).to(torch.float32)
            
            # set the diag of matrix -1
            feat_num = save_features.shape[0]
            flag_matrix[range(feat_num),range(feat_num)] = -1
            
            for i,line_flag in enumerate(flag_matrix):
                line_angle = angle_matrix[i]
                
                if len((line_flag == 1).nonzero()) == 0 and len((line_flag == 0).nonzero())==0:
                    continue
                elif len((line_flag == 1).nonzero()) == 0:
                    loss_uniform += min(line_angle[line_flag == 0])
                elif len((line_flag == 0).nonzero())==0:
                    loss_uniform += -max(line_angle[line_flag == 1]) + 1
                else:        
                    loss_uniform += min(line_angle[line_flag == 0])-max(line_angle[line_flag == 1]) + 1
            loss_uniform = 0.3*loss_uniform/feat_num if feat_num!=0 else loss_uniform
        
        return loss_uniform
    def get_cluster_loss_by_last_iter_cos(self,feats ,labels):
        
        loss_uniform = 0
        
        storage = get_event_storage()
        save_features = feats[0]
        
        if storage.iter == self.clustering_start_iter:
            self.last_iter_features = save_features
            self.last_iter_features_gt = labels
        elif storage.iter > self.clustering_start_iter:
            count = 0
            unk_features = save_features[labels == self.num_classes-1]
            
            for unk_feature in unk_features:
                last_iter_unk_features = self.last_iter_features[self.last_iter_features_gt == self.num_classes-1]
                last_iter_known_features = self.last_iter_features[self.last_iter_features_gt < self.seen_classes]
                unk_expand = unk_feature[None,:].expand_as(last_iter_unk_features)
                sim=self.cos(unk_expand,last_iter_unk_features)
                positive_mask = sim>=0.8
                sim_unk_features=last_iter_unk_features[positive_mask]
                neg_features=torch.cat([last_iter_unk_features[~positive_mask],last_iter_known_features])
                # print(sim_unk_features.shape,neg_features.shape)
                for sim_feeature in sim_unk_features:
                    sim=self.cos(unk_feature,sim_feeature)
                    unk_feature_expand = unk_feature.expand_as(neg_features)# gt 
                    diss_sim=self.cos(unk_feature_expand,neg_features)
                    sim=(sim/self.temperature).exp()
                    diss_sim=(diss_sim/self.temperature).exp().sum()
                    loss_uniform=loss_uniform+(-torch.log(sim/(sim+diss_sim)))
                    count=count+1
            self.last_iter_features = save_features
            self.last_iter_features_gt = labels

            loss_uniform = 0.1*loss_uniform/count if count!=0 else loss_uniform  
        
        return loss_uniform
    def get_uno_loss(self, feats, labels, mask_lab,storage_unk_feats,unknown_feats):
        labels = labels.long()
        
        self.normalize_prototypes()
        outputs = self(feats)

        outputs["logits_lab"] = (
            outputs["logits_lab"].unsqueeze(1).expand(-1, 1, -1, -1)
        )
        logits = torch.cat([outputs["logits_lab"], outputs["logits_unlab"]], dim=-1)
        logits_over = torch.cat([outputs["logits_lab"], outputs["logits_unlab_over"]], dim=-1)
        targets_lab = (
            F.one_hot(labels[mask_lab], num_classes=self.seen_classes).float().to(labels.device)
        )

        # 10*20 unknwon feature 1 targets-->5000 si 10 sj 10 sij 10 。
        targets = torch.zeros_like(logits)
        targets_over = torch.zeros_like(logits_over)
        nlc = self.seen_class  
        # print(logits.shape) # [2, 1, 7, 30]
        # loss_pair = self.get_pair_loss_by_ten_iter(logits,mask_lab,feats,labels,nlc)
        loss_pair = self.get_pair_loss_by_this_iter(logits,mask_lab,feats,labels,nlc) 
        
        # loss_uniform = self.get_cluster_loss_by_feature_angle(feats,labels)
        loss_uniform = self.get_cluster_loss_by_last_iter_cos(feats,labels)
        
        for v in range(2):
            for h in range(1):
                targets[v, h, mask_lab, :nlc] = targets_lab.type_as(targets)
                targets_over[v, h, mask_lab, :nlc] = targets_lab.type_as(targets)
                targets[v, h, ~mask_lab, nlc:] = self.sk(
                    outputs["logits_unlab"][v, h, ~mask_lab]
                ).type_as(targets)
                targets_over[v, h, ~mask_lab, nlc:] = self.sk(
                    outputs["logits_unlab_over"][v, h, ~mask_lab]
                ).type_as(targets)
        # loss_cluster = self.swapped_prediction(logits, targets)
        # loss_overcluster = self.swapped_prediction(logits_over, targets_over)
        # loss_cluster = loss_cluster.mean()
        # loss_overcluster = loss_overcluster.mean()
        # loss = (loss_cluster + loss_overcluster) / 2
        # loss+=loss_pair
        # loss+=loss_cluster+loss_uniform
        
        loss = loss_pair+loss_uniform
        # loss+=loss_cluster
        return {"uno_loss": loss}



    def res(self):
        # # known_features = save_features[labels < self.num_classes -1]
        # # known_label_set = set(labels[labels < self.num_classes -1].tolist())

        # known_mask = labels < self.num_classes -1
        # known_features = save_features[known_mask]
        # known_labels = labels[known_mask]
        # for idx,(known_feature, knwon_label) in enumerate(zip(known_features,known_labels)) :
        #     positive_mask = (known_labels == knwon_label)
        #     neg_features = known_features[~positive_mask]
            
        #     # remove this round known feature from positive features
        #     positive_mask[idx] = False
        #     positive_features=known_features[positive_mask]
        #     for sim_feeature in positive_features:
        #         sim=self.cos(known_feature,sim_feeature)
        #         knwon_feature_expand = known_feature.expand_as(neg_features)# gt 
        #         diss_sim=self.cos(knwon_feature_expand,neg_features)
        #         sim=(sim/self.temperature).exp()
        #         diss_sim=(diss_sim/self.temperature).exp().sum()
        #         loss_uniform=loss_uniform+(-torch.log(sim/(sim+diss_sim)))
        #         count=count+1
        pass