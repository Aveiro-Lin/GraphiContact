"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

"""

import torch
import src.modeling.data.config as cfg
from torch import nn

# from src.modeling.bert.diff_renderer import Pytorch3D

from src.modeling.bert.hrnet import hrnet_w32

class Decoder(nn.Module):
    def __init__(self, in_dim, out_dim, encoder='hrnet'):
        super(Decoder, self).__init__()
        self.out_dim = out_dim
        if encoder == 'swin':
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(),
                nn.ConvTranspose2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(),
                nn.ConvTranspose2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_dim),
                nn.Softmax(1)
            )
        elif encoder == 'hrnet':
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(),
                nn.ConvTranspose2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_dim),
                # nn.ReLU(),
                # nn.ConvTranspose2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                # nn.BatchNorm2d(out_dim),
                nn.Softmax(1)
            )
        else:
            raise NotImplementedError('Decoder not implemented')

    def forward(self, x):
        out = self.upsample(x)
        return out

class Encoder(nn.Module):
    def __init__(self, encoder='hrnet', pretrained=True):
        super(Encoder, self).__init__()

        if encoder == 'swin':
            '''Swin Transformer encoder'''
            self.encoder = torchvision.models.swin_b(weights='DEFAULT')
            self.encoder.head = nn.GELU()
        elif encoder == 'hrnet':
            '''HRNet encoder'''
            self.encoder = hrnet_w32(pretrained=pretrained)
        else:
            raise NotImplementedError('Encoder not implemented')

    def forward(self, x):
        out = self.encoder(x)
        return out  
class Graphormer_Body_Network(torch.nn.Module):
    '''
    End-to-end Graphormer network for human pose and mesh reconstruction from a single image.
    '''
    def __init__(self, args, config, backbone, trans_encoder, trans_encoder2, mesh_sampler):
        super(Graphormer_Body_Network, self).__init__()
        self.config = config
        self.config.device = args.device
        self.backbone = backbone
        self.encoder_sem = Encoder(encoder='hrnet').to(args.device)
        self.encoder_part = Encoder(encoder='hrnet').to(args.device)

        self.decoder_sem = Decoder(480, 133, encoder='hrnet').to(args.device)
        self.decoder_part = Decoder(480, 26, encoder='hrnet').to(args.device)
        if trans_encoder2 == None:
            self.trans_encoder2 = trans_encoder
        else:
            self.trans_encoder2 = trans_encoder2
        self.trans_encoder = trans_encoder
        self.upsampling = torch.nn.Linear(431, 1723)
        self.upsampling2 = torch.nn.Linear(1723, 6890)
        self.cam_param_fc = torch.nn.Linear(3, 1)
        self.cam_param_fc2 = torch.nn.Linear(431, 250)
        self.cam_param_fc3 = torch.nn.Linear(250, 3)
        self.grid_feat_dim = torch.nn.Linear(1024, 2051)


        ## The resulting image feature dimension is batch * 2048.
        ## The aim here is to pass contact information from the img feature to the 6890 vertices in a linear layer.
        self.img_feature_mlp = nn.Sequential(
            nn.Linear(2048, 6890),
            nn.LayerNorm(6890)
        )

        self.img_feature_mlp2 = nn.Sequential(
            nn.Linear(2048, 6890),
            nn.LayerNorm(6890)
        )

        ## Since features only have one dimension, put the three-dimensional vertex feature back in one dimension
        self.vertex_squeeze = nn.Linear(3, 1)

        ##The image information is fused with 3D feature information, and then a layer of mlp is made
        self.fusion_proj_mlp = nn.Sequential(
            nn.Linear(6890*2, 6890),
            nn.LayerNorm(6890),
            nn.Sigmoid()
        )

        self.fusion_proj_mlp2 = nn.Sequential(
            nn.Linear(6890*2, 6890),
            nn.LayerNorm(6890),
            nn.Sigmoid()
        )
    def paint_contact(self, pred_contact):
        """
        Paints the contact vertices on the SMPL mesh

        Args:
            pred_contact: prbabilities of contact vertices

        Returns:
            pred_rgb: RGB colors for the contact vertices
        """
        bs = pred_contact.shape[0]

        # initialize black and while colors
        colors = torch.tensor([[0, 0, 0], [1, 1, 1]]).float().to('cuda')
        colors = torch.unsqueeze(colors, 0).expand(bs, -1, -1)

        # add another dimension to the contact probabilities for inverse probabilities
        pred_contact = torch.unsqueeze(pred_contact, 2)
        pred_contact = torch.cat((1 - pred_contact, pred_contact), 2)

        # get pred_rgb colors
        pred_vert_rgb = torch.bmm(pred_contact, colors)
        pred_face_rgb = pred_vert_rgb[:, self.body_faces, :][:, :, 0, :] # take the first vertex color
        pred_face_texture = torch.zeros((bs, self.body_faces.shape[0], 1, 1, 3), dtype=torch.float32).to('cuda')
        pred_face_texture[:, :, 0, 0, :] = pred_face_rgb
        return pred_vert_rgb, pred_face_texture


    def forward(self, images, smpl, mesh_sampler, meta_masks=None, is_train=False):
        batch_size = images.size(0)
        # Generate T-pose template mesh
        template_pose = torch.zeros((1,72))
        template_pose[:,0] = 3.1416 # Rectify "upside down" reference mesh in global coord
        template_pose = template_pose.to(self.config.device)
        template_betas = torch.zeros((1,10)).to(self.config.device)
        template_vertices = smpl(template_pose, template_betas)
        self.body_faces = torch.LongTensor(smpl.faces.to('cpu')).to('cuda')
        
        # self.body_faces = torch.LongTensor(smpl.vertices.to('cpu')).to('cuda')

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
        template_vertices_sub2 = template_vertices_sub2.expand(batch_size, -1, -1)

        # extract grid features and global image features using a CNN backbone
        image_feat, grid_feat = self.backbone(images)
        ### task 2: scene and part ###### 
        sem_enc_out = self.encoder_sem(images)
        part_enc_out = self.encoder_part(images)

        sem_mask_pred = self.decoder_sem(sem_enc_out)
        part_mask_pred = self.decoder_part(part_enc_out)



        ### image_feature is fused with vertex features that need to be trained to make a one-dimensional vector. 
        # First, attach an mlp layer to it and do a dimensional transformation.
        fuse_img_f = self.img_feature_mlp(image_feat)

        fuse_img_f2 = self.img_feature_mlp2(image_feat)

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
            features1, hidden_states, att = self.trans_encoder(features)
        else:
            features1 = self.trans_encoder(features)

        pred_3d_joints = features1[:,:num_joints,:]
        ## batch * 431 * 3
        pred_vertices_sub2 = features1[:,num_joints:-49,:]

        # learn camera parameters
        x = self.cam_param_fc(pred_vertices_sub2)
        x = x.transpose(1,2)
        x = self.cam_param_fc2(x)
        x = self.cam_param_fc3(x)
        cam_param = x.transpose(1,2)
        # cam_param = cam_param.squeeze()

        temp_transpose = pred_vertices_sub2.transpose(1,2)
        pred_vertices_sub = self.upsampling(temp_transpose)
        pred_vertices_full = self.upsampling2(pred_vertices_sub)

        ## full vertex's 3D features are projected to 2 dimensions
        ## batch * 6890
        ver_1d_feat = self.vertex_squeeze(pred_vertices_full.transpose(1,2)).squeeze(dim=-1)
        ## fusion with img features
        #print(ver_1d_feat.size(), fuse_img_f.size())
        fusion_full = torch.cat([ver_1d_feat, fuse_img_f], dim=1)
        ## The resulting fusion information is the fusion of the image and the 3D vertex information, which is projected once as the output of the 3D model contact
        contact_ver_info = self.fusion_proj_mlp(fusion_full) # batch*6890
        #print(contact_ver_info.size())

        pred_vertices_sub = pred_vertices_sub.transpose(1,2)
        pred_vertices_full = pred_vertices_full.transpose(1,2)

        if self.config.output_attentions==True:
            features2, hidden_states, att = self.trans_encoder2(features)
        else:
            features2 = self.trans_encoder2(features)

        pred_3d_joints = features2[:,:num_joints,:]
        ## batch * 431 * 3
        pred_vertices_sub2 = features2[:,num_joints:-49,:]


        temp_transpose = pred_vertices_sub2.transpose(1,2)
        pred_vertices_sub = self.upsampling(temp_transpose)
        pred_vertices_full = self.upsampling2(pred_vertices_sub)

        ver_1d_feat = self.vertex_squeeze(pred_vertices_full.transpose(1,2)).squeeze(dim=-1)
        fusion_full = torch.cat([ver_1d_feat, fuse_img_f2], dim=1)
        ## The resulting fusion information is the fusion of the image and the 3D vertex information, which is projected once as the output of the 3D model contact.
        contact_ver_info2 = self.fusion_proj_mlp2(fusion_full) # batch*6890

        pred_vertices_sub = pred_vertices_sub.transpose(1,2)
        pred_vertices_full = pred_vertices_full.transpose(1,2)

        #### pal loss #####

        vertex_colors, face_textures = self.paint_contact(contact_ver_info2)
        # print(contact_ver_info2.shape)
        # print(face_textures.shape)
        #front_view = self.render_batch(template_vertices_sub2, cam_param, 1.0, vertex_colors, face_textures)
        # front_view_rgb = front_view[:, :3, :, :].permute(0, 2, 3, 1)
        # front_view_mask = front_view[:, 3, :100, :100].reshape(batch_size,-1)[:,:6890]
        # print(front_view_mask.shape)
        #### pal loss #####

        # front_view_rgb = front_view_rgb[valid_mask == 1]
        # gt_contact_polygon = gt_contact_polygon[valid_mask == 1]
        # loss = self.ce_loss(front_view_rgb, gt_contact_polygon)

        ### Finally the output is spliced back
        if self.config.output_attentions==True:
            return cam_param, pred_3d_joints, sem_mask_pred, part_mask_pred, pred_vertices_full, hidden_states, att, contact_ver_info, contact_ver_info2
        else:
            return cam_param, pred_3d_joints, sem_mask_pred, part_mask_pred, pred_vertices_full, contact_ver_info, contact_ver_info2

        
    # All the newly added parameters are packed here. If only contact information is trained, the original model parameters are not involved in the training
    def contact_parameters(self):
        params = []
        new_param_list = [self.img_feature_mlp, self.img_feature_mlp2, self.fusion_proj_mlp, self.fusion_proj_mlp2]
        for func in new_param_list:
            for n, param in func.named_parameters():
                params.append(param)

        return iter(params)

    