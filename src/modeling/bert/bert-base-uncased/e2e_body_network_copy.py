"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

"""

import torch
import src.modeling.data.config as cfg
from torch import nn


class Graphormer_Body_Network_2(torch.nn.Module):
    '''
    End-to-end Graphormer network for human pose and mesh reconstruction from a single image.
    '''
    def __init__(self, args, config, backbone, trans_encoder, mesh_sampler):
        super(Graphormer_Body_Network_2, self).__init__()
        self.config = config
        self.config.device = args.device
        self.backbone = backbone
        self.trans_encoder = trans_encoder
        self.upsampling = torch.nn.Linear(431, 1723)
        self.upsampling2 = torch.nn.Linear(1723, 6890)
        self.cam_param_fc = torch.nn.Linear(3, 1)
        self.cam_param_fc2 = torch.nn.Linear(431, 250)
        self.cam_param_fc3 = torch.nn.Linear(250, 3)
        self.grid_feat_dim = torch.nn.Linear(1024, 2051)

        #######################################
        ## 注意这边最后得到的image feature 维度是 batch * 2048
        ## 这里的目的是将img feature中的接触信息以线性层的方式传给6890个vertices
        self.img_feature_mlp = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.Linear(2048, 2048*3),
            nn.Linear(2048*3, 6890),
            nn.LayerNorm(6890)
        )

        ## 因为特征只有一维，把三维的vertex特征放回到一维
        self.vertex_squeeze = nn.Linear(3, 1)

        ## 把图片信息融合和三维特征信息融合后再做一层mlp
        self.fusion_proj_mlp = nn.Sequential(
            nn.Linear(6890*2, 6890),
            nn.Linear(6890, 6890*2),
            nn.Linear(6890*2, 6890),
            nn.LayerNorm(6890),
            nn.Sigmoid()
        )

    def forward(self, images, smpl, mesh_sampler, meta_masks=None, is_train=False):
        batch_size = images.size(0)
        # Generate T-pose template mesh
        template_pose = torch.zeros((1,72))
        template_pose[:,0] = 3.1416 # Rectify "upside down" reference mesh in global coord
        template_pose = template_pose.cuda(self.config.device)
        template_betas = torch.zeros((1,10)).cuda(self.config.device)
        template_vertices = smpl(template_pose, template_betas)

        # template mesh simplification
        template_vertices_sub = mesh_sampler.downsample(template_vertices)
        template_vertices_sub2 = mesh_sampler.downsample(template_vertices_sub, n1=1, n2=2)

        # template mesh-to-joint regression 
        template_3d_joints = smpl.get_h36m_joints(template_vertices)
        template_pelvis = template_3d_joints[:,cfg.H36M_J17_NAME.index('Pelvis'),:]
        template_3d_joints = template_3d_joints[:,cfg.H36M_J17_TO_J14,:]
        num_joints = template_3d_joints.shape[1]

        # normalize
        template_3d_joints = template_3d_joints - template_pelvis[:, None, :]
        template_vertices_sub2 = template_vertices_sub2 - template_pelvis[:, None, :]

        # concatinate template joints and template vertices, and then duplicate to batch size
        ref_vertices = torch.cat([template_3d_joints, template_vertices_sub2],dim=1)
        ref_vertices = ref_vertices.expand(batch_size, -1, -1)

        # extract grid features and global image features using a CNN backbone
        image_feat, grid_feat = self.backbone(images)
        

        ### image_feature和需要训练的vertex特征做融合，做出一维向量，首先给他接一个mlp层，做一下维度转换
        fuse_img_f = self.img_feature_mlp(image_feat)

        # batch * verti_shape * 2048
        # concatinate image feat and 3d mesh template
        image_feat = image_feat.view(batch_size, 1, 2048).expand(-1, ref_vertices.shape[-2], -1)
        # process grid features
        grid_feat = torch.flatten(grid_feat, start_dim=2)
        grid_feat = grid_feat.transpose(1,2)
        grid_feat = self.grid_feat_dim(grid_feat)
        # concatinate image feat and template mesh to form the joint/vertex queries
        features = torch.cat([ref_vertices, image_feat], dim=2)
        # prepare input tokens including joint/vertex queries and grid features
        features = torch.cat([features, grid_feat],dim=1)

        if is_train==True:
            # apply mask vertex/joint modeling
            # meta_masks is a tensor of all the masks, randomly generated in dataloader
            # we pre-define a [MASK] token, which is a floating-value vector with 0.01s
            special_token = torch.ones_like(features[:,:-49,:]).cuda()*0.01
            features[:,:-49,:] = features[:,:-49,:]*meta_masks + special_token*(1-meta_masks)          

        # forward pass
        if self.config.output_attentions==True:
            features, hidden_states, att = self.trans_encoder(features)
        else:
            features = self.trans_encoder(features)

        pred_3d_joints = features[:,:num_joints,:]
        ## batch * 431 * 3
        pred_vertices_sub2 = features[:,num_joints:-49,:]

        # learn camera parameters
        x = self.cam_param_fc(pred_vertices_sub2)
        x = x.transpose(1,2)
        x = self.cam_param_fc2(x)
        x = self.cam_param_fc3(x)
        cam_param = x.transpose(1,2)
        cam_param = cam_param.squeeze()

        temp_transpose = pred_vertices_sub2.transpose(1,2)
        pred_vertices_sub = self.upsampling(temp_transpose)
        pred_vertices_full = self.upsampling2(pred_vertices_sub)

        ## full vertex的3D特征投影到2维
        ## batch * 6890
        ver_1d_feat = self.vertex_squeeze(pred_vertices_full.transpose(1,2)).squeeze(dim=-1)
        print("vertex_1d_feat:", ver_1d_feat.shape())
        print("pre_vertices_full:", pred_vertices_full.shape())
        ## 和img的信息融合
        ## 这里图片维度和vertex信息维度都是一样的，所以相加和拼接都行
        #print(ver_1d_feat.size(), fuse_img_f.size())
        fusion_full = torch.cat([ver_1d_feat, fuse_img_f], dim=1)
        ## 得到的fusion信息是图片和3D vertex信息的融合，在投射一次作为3D 模型contact的输出
        contact_ver_info = self.fusion_proj_mlp(fusion_full) # batch*6890
        #print(contact_ver_info.size())

        pred_vertices_sub = pred_vertices_sub.transpose(1,2)
        pred_vertices_full = pred_vertices_full.transpose(1,2)

        ### 最后输出拼接到后面
        if self.config.output_attentions==True:
            return cam_param, pred_3d_joints, pred_vertices_sub2, pred_vertices_sub, pred_vertices_full, hidden_states, att, contact_ver_info
        else:
            return cam_param, pred_3d_joints, pred_vertices_sub2, pred_vertices_sub, pred_vertices_full, contact_ver_info
        
    # 在这里打包所有的新增加的参数，如果只训练接触信息，原模型参数不参与训练
    def contact_parameters(self):
        params = []
        new_param_list = [self.img_feature_mlp, self.vertex_squeeze, self.fusion_proj_mlp]
        for func in new_param_list:
            for n, param in func.named_parameters():
                params.append(param)

        return iter(params)

    