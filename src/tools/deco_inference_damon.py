"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

End-to-end inference codes for 
3D human body mesh reconstruction from an image
"""

from __future__ import absolute_import, division, print_function
import argparse
import open3d as o3d
import skimage.io as io
from chumpy.utils import row, col
import os
os.environ['CRYPTOGRAPHY_OPENSSL_NO_LEGACY']="1"
import os.path as op
import code
import json
import time
import datetime
import torch
import torchvision.models as models
from torchvision.utils import make_grid
import gc
import numpy as np
import cv2
import sys
sys.path.append('Path/to/GraphiContact')
from src.modeling.bert.diff_renderer import Pytorch3D
from src.modeling.bert import BertConfig, Graphormer
from src.modeling.bert import Graphormer_Body_Network as Graphormer_Network
from src.modeling._smpl import SMPL, Mesh
from src.tools.smpl import SMPL as SMPL2
from src.modeling.hrnet.hrnet_cls_net_gridfeat import get_cls_net_gridfeat
from src.modeling.hrnet.config import config as hrnet_config
from src.modeling.hrnet.config import update_config as hrnet_update_config
import src.modeling.data.config as cfg
from src.datasets.build import make_data_loader
from src.tools.visualize import *
import src.tools.new_renderer as vis_util

from src.utils.logger import setup_logger
from src.utils.comm import synchronize, is_main_process, get_rank, get_world_size, all_gather
from src.utils.miscellaneous import mkdir, set_seed
from src.utils.metric_logger import AverageMeter, EvalMetricsLogger
from src.utils.renderer import Renderer as Renderer2
from src.utils.renderer import visualize_reconstruction_and_att_local, visualize_reconstruction_no_text, visualize_reconstruction_no_text_new
from src.utils.metric_pampjpe import reconstruction_error
from src.utils.geometric_layers import orthographic_projection
import copy
from PIL import Image
from torchvision import transforms

# data augmentation
transform = transforms.Compose([           
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])])

# normal resize
transform_visualize = transforms.Compose([           
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor()])

class Minimal(object):
    def __init__(self, **kwargs):
        self.__dict__ = kwargs

def read_obj(filename):

    obj_directory = split(filename)[0]
    lines = open(filename).read().split('\n')

    d = {'v': [], 'vn': [], 'f': [], 'vt': [], 'ft': []}

    mtls = {}
    for line in lines:
        line = line.split()
        if len(line) < 2:
            continue

        key = line[0]
        values = line[1:]

        if key == 'v':
            d['v'].append([np.array([float(v) for v in values[:3]])])
        elif key == 'f':
            spl = [l.split('/') for l in values]
            d['f'].append([np.array([int(l[0])-1 for l in spl[:3]], dtype=np.uint32)])
            if len(spl[0]) > 1 and spl[1] and 'ft' in d:
                d['ft'].append([np.array([int(l[1])-1 for l in spl[:3]])])

            # TOO: redirect to actual vert normals?
            #if len(line[0]) > 2 and line[0][2]:
            #    d['fn'].append([np.concatenate([l[2] for l in spl[:3]])])
        elif key == 'vn':
            d['vn'].append([np.array([float(v) for v in values])])
        elif key == 'vt':
            d['vt'].append([np.array([float(v) for v in values])])
        elif key == 'mtllib':
            fname = join(obj_directory, values[0])
            if not exists(fname):
                fname = values[0]
            if not exists(fname):
                raise Exception("Can't find path %s" % (values[0]))
            _update_mtl(mtls, fname)
        elif key == 'usemtl':
            cur_mtl = mtls[values[0]]

            if 'map_Kd' in cur_mtl:
                src_fname = cur_mtl['map_Kd'][0]
                dst_fname = join(split(cur_mtl['filename'])[0], src_fname)
                if not exists(dst_fname):
                    dst_fname = join(obj_directory, src_fname)
                if not exists(dst_fname):
                    dst_fname = src_fname
                if not exists(dst_fname):
                    raise Exception("Unable to find referenced texture map %s" % (src_fname,))
                else:
                    d['texture_filepath'] = normpath(dst_fname)
                    im = cv2.imread(dst_fname)
                    sz = np.sqrt(np.prod(im.shape[:2]))
                    sz = int(np.round(2 ** np.ceil(np.log(sz) / np.log(2))))
                    d['texture_image'] = cv2.resize(im, (sz, sz)).astype(np.float64)/255.

    for k, v in list(d.items()):
        if k in ['v','vn','f','vt','ft']:
            if v:
                d[k] = np.vstack(v)
            else:
                del d[k]
        else:
            d[k] = v

    result = Minimal(**d)

    return result

def mask_split(img, num_parts):
    if not len(img.shape) == 2:
        img = img[:, :, 0]
    mask = np.zeros((img.shape[0], img.shape[1], num_parts))
    for i in np.unique(img):
        mask[:, :, i] = np.where(img == i, 1., 0.)
    return np.transpose(mask, (2, 0, 1))
def get_posed_mesh(betas, pose, transl, debug=False):

    # extra smplx params
    print('!!',pose.shape)
    extra_args = {'jaw_pose': torch.zeros((1, 3)).float(),
                  'leye_pose': torch.zeros((1, 3)).float(),
                  'reye_pose': torch.zeros((1, 3)).float(),
                  'expression': torch.zeros((1, 10)).float(),
                  'left_hand_pose': torch.zeros((1, 45)).float(),
                  'right_hand_pose': torch.zeros((1, 45)).float()}
    body_model = SMPL2('Path/to/GraphiContact/src/smpl/')
    smpl_output = body_model(betas=betas.reshape(1,-1),
                                  body_pose=pose[ 3:].reshape(1,-1),
                                  global_orient=pose[ :3].reshape(1,-1),
                                  pose2rot=True,
                                  transl=transl.reshape(1,-1),
                                  **extra_args)
    smpl_verts = smpl_output.vertices
    smpl_joints = smpl_output.joints

    if debug:
        for mesh_i in range(smpl_verts.shape[0]):
            out_dir = 'temp_meshes'
            os.makedirs(out_dir, exist_ok=True)
            out_file = os.path.join(out_dir, f'temp_mesh_{mesh_i:04d}.obj')
            save_results_mesh(smpl_verts[mesh_i], self.body_model.faces, out_file)
    return smpl_verts, smpl_joints

def paint_contact(pred_contact):
    """
    Paints the contact vertices on the SMPL mesh

    Args:
        pred_contact: prbabilities of contact vertices

    Returns:
        pred_rgb: RGB colors for the contact vertices
    """
    bs = pred_contact.shape[0]

    # initialize black and while colors
    colors = torch.tensor([[0, 0, 0], [1, 1, 1]]).float()
    colors = torch.unsqueeze(colors, 0).expand(bs, -1, -1)

    # add another dimension to the contact probabilities for inverse probabilities
    pred_contact = torch.unsqueeze(pred_contact, 2)
    pred_contact = torch.cat((1 - pred_contact, pred_contact), 2)

    # get pred_rgb colors
    pred_vert_rgb = torch.bmm(pred_contact, colors)
    body_model = SMPL2('Path/to/GraphiContact/src/smpl/')
    body_faces = torch.LongTensor(body_model.faces.astype(np.int32))
    pred_face_rgb = pred_vert_rgb[:, body_faces, :][:, :, 0, :] # take the first vertex color
    pred_face_texture = torch.zeros((bs, body_faces.shape[0], 1, 1, 3), dtype=torch.float32)
    pred_face_texture[:, :, 0, 0, :] = pred_face_rgb
    return pred_vert_rgb, pred_face_texture

def render_batch(smpl_verts, cam_k, img_scale_factor, vertex_colors=None, face_textures=None, debug=False):

    bs = 1

    # Incorporate resizing factor into the camera
    img_w = 256 # TODO: Remove hardcoding
    img_h = 256 # TODO: Remove hardcoding
    focal_length_x = cam_k[ 0, 0] * img_scale_factor[ 0]
    focal_length_y = cam_k[ 1, 1] * img_scale_factor[ 1]
    # convert to float for pytorch3d
    focal_length_x, focal_length_y = focal_length_x.float().unsqueeze(0), focal_length_y.float().unsqueeze(0)

    # concatenate focal length
    focal_length = torch.stack([focal_length_x, focal_length_y], dim=1)

    # Setup renderer
    body_model = SMPL2('Path/to/GraphiContact/src/smpl/')
    body_faces = torch.LongTensor(body_model.faces.astype(np.int32))
    renderer = Pytorch3D(img_h=img_h,
                              img_w=img_w,
                              focal_length=focal_length.cuda(),
                              smpl_faces=body_faces.cuda(),
                              texture_mode='deco',
                              vertex_colors=vertex_colors.cuda(),
                              face_textures=face_textures.cuda(),
                              is_train=True,
                              is_cam_batch=True)
    front_view = renderer(smpl_verts.cuda())
    if debug:
        # visualize the front view as images in a temp_image folder
        for i in range(bs):
            front_view_rgb = front_view[i, :3, :, :].permute(1, 2, 0).detach().cpu()
            front_view_mask = front_view[i, 3, :, :].detach().cpu()
            out_dir = 'temp_images'
            os.makedirs(out_dir, exist_ok=True)
            out_file_rgb = os.path.join(out_dir, f'{i:04d}_rgb.png')
            out_file_mask = os.path.join(out_dir, f'{i:04d}_mask.png')
            cv2.imwrite(out_file_rgb, front_view_rgb.numpy()*255)
            cv2.imwrite(out_file_mask, front_view_mask.numpy()*255)

    return front_view

def _update_mtl(mtl, filename):

    lines = [l.strip() for l in open(filename).read().split('\n')]

    curkey = ''
    for line in lines:
        spl = line.split()

        if len(spl) < 2:
            continue
        key = spl[0]
        values = spl[1:]

        if key == 'newmtl':
            curkey = values[0]
            mtl[curkey] = {'filename': filename}
        elif curkey:
            mtl[curkey][key] = values
def wget(url, dest_fname=None):
    try: #python3
        from urllib.request import urlopen
    except: #python2
        from urllib2 import urlopen

    from os.path import split, join

    curdir = split(__file__)[0]
    print(url)
    

    if dest_fname is None:
        dest_fname = join(curdir, split(url)[1])

    try:
        contents = urlopen(url).read()
    except:
        raise Exception('Unable to get url: %s' % (url,))
    open(dest_fname, 'w').write(contents)
from os.path import split, splitext, join, exists, normpath
def load_mesh(filename):

    extension = splitext(filename)[1]
    if  extension == '.ply':
        return read_ply(filename)
    elif extension == '.obj':
        return read_obj(filename)
    else:
        raise Exception('Unsupported file extension for %s' % (filename,))
    
def get_earthmesh(trans, rotation):

    from copy import deepcopy
    if not hasattr(get_earthmesh, 'm'):

        def wg(url):
            dest = join('/tmp', split(url)[1])
            if not exists(dest):
                wget(url, dest)

        fname = join('Path/to/GraphiContact/src/tools/', 'nasa_earth.obj')
        mesh = load_mesh(fname)

        mesh.v = np.asarray(mesh.v, order='C')
        mesh.vc = mesh.v*0 + 1
        mesh.v -= row(np.mean(mesh.v, axis=0))
        mesh.v /= np.max(mesh.v)
        mesh.v *= 2.0
        get_earthmesh.mesh = mesh

    mesh = deepcopy(get_earthmesh.mesh)
    mesh.v = mesh.v.dot(cv2.Rodrigues(np.asarray(np.array(rotation), np.float64))[0])
    mesh.v = mesh.v + row(np.asarray(trans))
    return mesh


def resize_img(img, scale_factor):
    new_size = (np.floor(np.array(img.shape[0:2]) * scale_factor)).astype(int)
    new_img = cv2.resize(img, (new_size[1], new_size[0]))
    # This is scale factor of [height, width] i.e. [y, x]
    actual_factor = [
        new_size[0] / float(img.shape[0]), new_size[1] / float(img.shape[1])
    ]
    return new_img, actual_factor


def scale_and_crop(image, scale, center, img_size):
    image_scaled, scale_factors = resize_img(image, scale)
    # Swap so it's [x, y]
    scale_factors = [scale_factors[1], scale_factors[0]]
    center_scaled = np.round(center * scale_factors).astype(np.int)

    margin = int(img_size / 2)
    image_pad = np.pad(
        image_scaled, ((margin, ), (margin, ), (0, )), mode='edge')
    center_pad = center_scaled + margin
    # figure out starting point
    start_pt = center_pad - margin
    end_pt = center_pad + margin
    # crop:
    crop = image_pad[start_pt[1]:end_pt[1], start_pt[0]:end_pt[0], :]
    proc_param = {
        'scale': scale,
        'start_pt': start_pt,
        'end_pt': end_pt,
        'img_size': img_size
    }

    return crop, proc_param

def preprocess_image(img_path, json_path=None):
    img = io.imread(img_path)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    if json_path is None:
        if np.max(img.shape[:2]) != 224:
            print('Resizing so the max image size is %d..' % 224)
            scale = (float(224) / np.max(img.shape[:2]))
        else:
            scale = 1.
        center = np.round(np.array(img.shape[:2]) / 2).astype(int)
        # image center in (x,y)
        center = center[::-1]
        


    crop, proc_param = scale_and_crop(img, scale, center,
                                               224)
    
    crop = 2 * ((crop / 255.) - 0.5)

    return crop, proc_param, img

def run_inference(args, image_list, Graphormer_model, smpl, mesh_sampler):
    # switch to evaluate mode
    Graphormer_model.eval()
    smpl.eval()
    with torch.no_grad():
        for image_file in image_list:
            if 'pred' not in image_file:
                att_all = []
                img = Image.open(image_file)
                img_tensor = transform(img)
                img_visual = transform_visualize(img)

                batch_imgs = torch.unsqueeze(img_tensor, 0).cuda()
                batch_visual_imgs = torch.unsqueeze(img_visual, 0).cuda()
                # forward-pass
                pred_camera, pred_3d_joints, sem_mask_pred, part_mask_pred, pred_vertices, hidden_states, att, pred_contact_ver, _ = Graphormer_model(batch_imgs, smpl, mesh_sampler)
                vertex_colors, face_textures = paint_contact(pred_contact_ver.cpu())
                print(face_textures.shape)
                
                pred_contact_ver_np = pred_contact_ver.cpu().numpy()
                temp_fname = image_file[:-4]
                import numpy as np
                # store 3d points
                np.save(os.path.join('Path/to/GraphiContact/src/tools',f"2_pred_con_ver"),pred_contact_ver_np)
                print(pred_camera.size())
                # copy the camera parameters again
                pred_camera1 = copy.deepcopy(pred_camera)
                # obtain 3d joints from full mesh
                pred_3d_joints_from_smpl = smpl.get_h36m_joints(pred_vertices)

                pred_3d_pelvis = pred_3d_joints_from_smpl[:,cfg.H36M_J17_NAME.index('Pelvis'),:]
                pred_3d_joints_from_smpl = pred_3d_joints_from_smpl[:,cfg.H36M_J17_TO_J14,:]
                pred_3d_joints_from_smpl = pred_3d_joints_from_smpl - pred_3d_pelvis[:, None, :]

                label = torch.where(pred_contact_ver>0, 1, 0)
                #print(label.size())
                # The original vertex 3 is used to subtract the pred_3d_pelvis coordinates
                # breakpoint()
                pred_vertices = pred_vertices - pred_3d_pelvis[:, None, :]
                pred_vertices_np = pred_vertices.cpu().numpy()
                np.save(os.path.join('Path/to/GraphiContact/src/tools',f"2_pred_ver"),pred_vertices_np)
                pred_labels = pred_vertices * torch.tensor(label).unsqueeze(dim=2).cuda()

                ver = pred_vertices.cpu().numpy()[0]
                con = pred_contact_ver.cpu().numpy()[0]

                con_label = np.where(con>0.5, 1, 0)
                un_con_label = np.where(con>0.5, 0, 1)
                con_ver = ver * np.expand_dims(con_label,axis=1)
                un_con_ver = ver * np.expand_dims(un_con_label,axis=1)
                

                # The points of contact are made red, and the other is white
                colors = np.zeros_like(con_ver)
                white = np.asarray([1, 1, 1])
                red = np.asarray([1,0,0])
                for i in range(len(con_label)):
                    if con_label[i] == 0:
                        colors[i] = white
                    else:
                        colors[i] = red


                         
                smpl_path = os.path.join('Path/to/GraphiContact/src/tools', 'smpl_neutral_tpose.ply')
                cont = pred_contact_ver.detach().cpu().numpy().squeeze()
                cont_smpl = []
                for indx, i in enumerate(cont):
                    if i >= 0.5:
                        cont_smpl.append(indx)

                img = cv2.imread(image_file)
                img = cv2.resize(img, (256, 256), cv2.INTER_CUBIC)
                img = img.transpose(2,0,1)/255.0
                img = img[np.newaxis,:,:,:]
                img = torch.tensor(img, dtype = torch.float32).to('cuda')
                
                img = img.detach().cpu().numpy()		
                img = np.transpose(img[0], (1, 2, 0))		
                img = img * 255		
                img = img.astype(np.uint8)
                
                contact_smpl = np.zeros((1, 1, 6890))
                contact_smpl[0][0][cont_smpl] = 1

                body_model_smpl = trimesh.load(smpl_path, process=False)
                for vert in range(body_model_smpl.visual.vertex_colors.shape[0]):
                    body_model_smpl.visual.vertex_colors[vert] = [130, 130, 130, 255]
                body_model_smpl.visual.vertex_colors[cont_smpl] = [0, 255, 0, 255]

                # rend = create_scene(body_model_smpl, img)
                sem_mask_pred = sem_mask_pred.detach().cpu().numpy()
                part_mask_pred = part_mask_pred.detach().cpu().numpy()

                                        
                out_dir = os.path.join('Path/to/GraphiContact/src/tools', 'Preds', os.path.basename(image_file).split('.')[0])
                os.makedirs(out_dir, exist_ok=True)          

                # logger.info(f'Saving mesh to {out_dir}')
                # shutil.copyfile(img_name, os.path.join(out_dir, os.path.basename(img_name)))
                # body_model_smpl.export(os.path.join(out_dir, 'pred.obj'))

                data_path='Path/to/GraphiContact/HOT-Annotated/images'
                label_path='Path/to/GraphiContact/HOT-Annotated/Release_Datasets/damon/hot_dca_trainval.npz'
                org_img_paths = os.listdir(data_path)
                # org_img_paths = [os.path.join(data_path, path) for path in org_img_paths]
                train_label = np.load(label_path)

                
                tr_idx = 0
                foridx = 'datasets/HOT-Annotated/images/' + args.inputs
                for x in range(len(train_label['imgname'])):
                    # if int(train_label['imgname'][x].split('/')[-1].split('_')[-1].split('.')[0]) >= 301402:
                    #     print(train_label['imgname'][x])
                    if train_label['imgname'][x]  == foridx:
                        tr_idx = x
                print(foridx)
                train_imgs_names = train_label['imgname'][tr_idx]
                print('input:', tr_idx)

                img = cv2.imread(os.path.join(data_path, train_imgs_names.split('/')[-1] ))
                img = cv2.resize(img, (256, 256), cv2.INTER_CUBIC)
                img = img.transpose(2,0,1)/255.0
                img = img[np.newaxis,:,:,:]
                img = torch.tensor(img, dtype = torch.float32).to('cuda')
                
                img = img.detach().cpu().numpy()		
                img = np.transpose(img[0], (1, 2, 0))		
                img = img * 255		
                img = img.astype(np.uint8)

                rend1 = create_scene(body_model_smpl, img)
                
                os.makedirs(os.path.join('Path/to/GraphiContact/src/tools', 'Renders'), exist_ok=True) 
                rend1.save(os.path.join('Path/to/GraphiContact/src/tools', 'Renders', '2' + '.png'))


                


                img_h, img_w, _ = img.shape


                img_scale_factor = np.array([256 / img_w, 256 / img_h])

                seg_path='Path/to/GraphiContact/HOT-Annotated/segments'
                seg_path='Path/to/GraphiContact/datasets/Damon_Scene/damon_segmentations/segmentation_masks/training'
                part_path='Path/to/GraphiContact/datasets/Damon_Scene/damon_segmentations/parts/training'
                seg_imgs_names = train_label['scene_seg'][tr_idx]
                part_imgs_names = train_label['part_seg'][tr_idx]


                img2 = cv2.imread(os.path.join(seg_path, seg_imgs_names.split('/')[-1] ))
                img2 = cv2.resize(img2, (256, 256), cv2.INTER_CUBIC)
                img2 = mask_split(img2, 133)


                img3 = cv2.imread(os.path.join(part_path, part_imgs_names.split('/')[-1] ))
                img3 = cv2.resize(img3, (256, 256), cv2.INTER_CUBIC)
                img3 = mask_split(img3, 26)	


                
                print(train_label.files)
                pose = train_label['pose'][tr_idx]
                betas = train_label['shape'][tr_idx]
                print(betas)
                transl = train_label['transl'][tr_idx]
                transl = torch.tensor(transl, dtype=torch.float32)
                betas = torch.tensor(betas, dtype=torch.float32)
                pose = torch.tensor(pose, dtype=torch.float32)
                # has_smpl = train_label['has_smpl']
                # is_smplx = train_label['is_smplx']

                cam_k = train_label['cam_k'][tr_idx]
                cam_k = torch.tensor(cam_k, dtype=torch.float32)
                img_scale_factor = torch.tensor(img_scale_factor, dtype=torch.float32)
                smpl_body_params = {'pose': pose, 'betas': betas,
                                'transl': transl}
        
                smpl_verts, smpl_joints = get_posed_mesh(betas, pose, transl )

                pred_vertices = smpl_verts.cpu().numpy() - pred_3d_pelvis[:, None, :].cpu().numpy()
                pred_vertices_np = pred_vertices
                np.save(os.path.join('Path/to/GraphiContact/src/tools',"2_pred_ver"),pred_vertices_np)

                
                front_view = render_batch(smpl_verts, cam_k, img_scale_factor, vertex_colors, face_textures)
                front_view_rgb = front_view[:, :3, :, :].permute(0, 2, 3, 1)
                
                contact_2d_pred_rgb = front_view_rgb.detach().cpu().numpy()
                rend2 = gen_render(img, cont, contact_2d_pred_rgb, img2, img3)
                rend2.save(os.path.join('Path/to/GraphiContact/src/tools', 'Renders', '2' + 'sence.png'))

                # visual_imgs_output = visualize_mesh_and_attention( renderer, batch_visual_imgs[0],
                #                                             pred_vertices[0].detach(), 
                #                                             pred_vertices_sub2[0].detach(), 
                #                                             pred_2d_431_vertices_from_smpl[0].detach(),
                #                                             pred_2d_joints_from_smpl[0].detach(),
                #                                             pred_camera.detach(),
                #                                             att[-1][0].detach())

                # visual_imgs_output = torch.tensor(visual_imgs_output.transpose(1,2,0))
                body_model = SMPL2('Path/to/GraphiContact/src/smpl/')
                
                body_faces = torch.LongTensor(body_model.faces.astype(np.int32))
                renderer = Renderer2(faces=body_model.faces.reshape([ 13776, 3]))
                # renderer = vis_util.SMPLRenderer(face_path=body_model.faces.reshape([ 13776, 3]))
                print('smpl',smpl.faces.cpu().numpy().shape)
                # input_img, proc_param, img = preprocess_image(os.path.join(data_path, train_imgs_names.split('/')[-1] ))
                
                new_background_img = Image.open(os.path.join(data_path, train_imgs_names.split('/')[-1] ))
                img_visual = transform_visualize(new_background_img)
                img_visual = torch.unsqueeze(img_visual, 0)
                print('ii', img_visual.shape)
                label = torch.where(pred_contact_ver>0.5, 1, 0)
                ver = pred_vertices_np[0]
                con = pred_contact_ver_np[0]

                con_label = np.where(con>0.5, 1, 0)
                un_con_label = np.where(con>0.5, 0, 1)
                con_ver = ver * np.expand_dims(con_label,axis=1)
                un_con_ver = ver * np.expand_dims(un_con_label,axis=1)

                # Those who have touched it are given red, and those who have not touched it are white
                colors = np.zeros_like(con_ver)
                white = np.asarray([1, 1, 1])
                red = np.asarray([1,0,0])
                for i in range(len(con_label)):
                    if con_label[i] == 0:
                        colors[i] = white
                    else:
                        colors[i] = red
                print('111',colors.shape)
                pred_labels = torch.tensor(pred_vertices).cuda() * torch.tensor(label).unsqueeze(dim=2).cuda()
                print('cam',pred_camera1.shape)
                print(cam_k.shape)
                visual_imgs_output = visualize_mesh( renderer, img_visual.detach().numpy()[0],
                                                                smpl_verts[0].numpy(), 
                                                                cam_k[0],color=colors)
                np.save(os.path.join('Path/to/GraphiContact/src/tools',"colors"),colors)
                np.save(os.path.join('Path/to/GraphiContact/src/tools',"pred_vertices"),smpl_verts)
                np.save(os.path.join('Path/to/GraphiContact/src/tools',"pred_camera1"),cam_k.cpu().numpy())
                # cam_for_render, vert_shifted, joints_orig = vis_util.get_original(
                #     proc_param, pred_vertices, cam_k[0].cpu(), smpl_joints, img_size=img.shape[:2])
                # rend_img_overlay = renderer(
                #     vert_shifted, cam=cam_for_render, img=img, do_alpha=False)
                
                visual_imgs = visual_imgs_output
                visual_imgs = np.asarray(visual_imgs).transpose(1,2,0)
                # print(visual_imgs)
                print(image_file[:-4])
                temp_fname = 'Path/to/GraphiContact/src/tools/' + image_file[:-4].split('/')[-1] + '_deco_pred.jpg'
                print('save to ', temp_fname)
                
                cv2.imwrite(temp_fname, np.asarray(visual_imgs[:,:,::-1]*255))
                import open3d as o3d




                # Create a window object
                vis = o3d.visualization.Visualizer()
                # Set the window title
                

                
                # Create a point cloud object
                pcd= o3d.open3d.geometry.PointCloud()
                pcd.points = o3d.open3d.utility.Vector3dVector(ver)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                # Estimate normals
                radius1 = 0.1   # Search radius
                max_nn = 100    # Maximum number of points in the neighborhood used for normal estimation
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius1, max_nn))  # Perform normal estimation                #估计滚球半径
                # Estimate ball radius
                distances = pcd.compute_nearest_neighbor_distance()
                avg_dist = np.mean(distances)
                radius = 1.5 * avg_dist   
                # Convert the point cloud to a mesh
                # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                        # pcd,
                        # o3d.utility.DoubleVector([radius, radius * 2]))
                # o3d.io.write_triangle_mesh("Path/to/GraphiContact/src/tools/output_mesh.ply", mesh)
                

                from opendr.renderer import ColoredRenderer
                rn = ColoredRenderer()
                import chumpy as ch

                # import matplotlib.pyplot as plt
                # cv2.imwrite(temp_fname, np.asarray(rn.r[:,:,::-1]*255))
                break
                

    return 

def visualize_mesh( renderer, images,
                    pred_vertices_full,
                    pred_camera,
                    color='light_blue'):
    img = images.transpose(1,2,0)
    # Get predict vertices for the particular example
    vertices_full = pred_vertices_full
    #vertices_full = np.array([x for x in vertices_full if x[0] != 0])
    print(vertices_full.shape)
    cam = pred_camera.cpu().numpy().squeeze()
    # print(cam)
    # breakpoint()
    # Visualize only mesh reconstruction 
    rend_img = visualize_reconstruction_no_text_new(img, 256, vertices_full, cam, renderer, color=color)

    rend_img = rend_img.transpose(2,0,1)
    return rend_img

def visualize_mesh_and_attention( renderer, images,
                    pred_vertices_full,
                    pred_vertices, 
                    pred_2d_vertices,
                    pred_2d_joints,
                    pred_camera,
                    attention):
    img = images.cpu().numpy().transpose(1,2,0)
    # Get predict vertices for the particular example
    vertices_full = pred_vertices_full.cpu().numpy() 
    vertices = pred_vertices.cpu().numpy()
    vertices_2d = pred_2d_vertices.cpu().numpy()
    joints_2d = pred_2d_joints.cpu().numpy()
    cam = pred_camera.cpu().numpy()
    att = attention.cpu().numpy()
    # Visualize reconstruction and attention
    rend_img = visualize_reconstruction_and_att_local(img, 224, vertices_full, vertices, vertices_2d, cam, renderer, joints_2d, att, color='light_blue')
    rend_img = rend_img.transpose(0,1)
    return rend_img


def parse_args():
    parser = argparse.ArgumentParser()

    # The test.npz file was not used, so there is no gt_ana, making it impossible to calculate precision, recall, F1 score, FP error, and FN error.


    #########################################################
    # Data related arguments
    #########################################################
    parser.add_argument("--num_workers", default=4, type=int, 
                        help="Workers in dataloader.")
    parser.add_argument("--img_scale_factor", default=1, type=int, 
                        help="adjust image resolution.")
    parser.add_argument("--image_file_or_path", default='Path/to/GraphiContact/samples/human-body_deco', type=str, 
                        help="test data") 
    #########################################################
    # Loading/saving checkpoints
    #########################################################
    parser.add_argument("--model_name_or_path", default='Path/to/GraphiContact/src/modeling/bert/bert-base-uncased', type=str, required=False,
                        help="Path to pre-trained transformer model or model type.")
    # metamodel./models/graphormer_release/graphormer_3dpw_state_dict.bin
    parser.add_argument("--resume_checkpoint", default='Path/to/GraphiContact/models/graphormer_release/graphormer_3dpw_state_dict.bin', type=str, required=False,
                        help="Path to specific checkpoint for resume training.")
    parser.add_argument("--deco_resume_checkpoint", default='Path/to/GraphiContact/ckpt/deco_grph_damon.pt', type=str, required=False,
                        help="Path to specific checkpoint for resume training.")
    parser.add_argument("--output_dir", default='output/', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    parser.add_argument("--config_name", default="", type=str, 
                        help="Pretrained config name or path if not the same as model_name.")
    #########################################################
    # Model architectures
    #########################################################
    parser.add_argument('-a', '--arch', default='hrnet-w64',
                    help='CNN backbone architecture: hrnet-w64, hrnet, resnet50')
    parser.add_argument("--num_hidden_layers", default=4, type=int, required=False, 
                        help="Update model config if given")
    parser.add_argument("--hidden_size", default=-1, type=int, required=False, 
                        help="Update model config if given")
    parser.add_argument("--num_attention_heads", default=4, type=int, required=False, 
                        help="Update model config if given. Note that the division of "
                        "hidden_size / num_attention_heads should be in integer.")
    parser.add_argument("--intermediate_size", default=-1, type=int, required=False, 
                        help="Update model config if given.")
    parser.add_argument("--input_feat_dim", default='2051,512,128', type=str, 
                        help="The Image Feature Dimension.")          
    parser.add_argument("--hidden_feat_dim", default='1024,256,64', type=str, 
                        help="The Image Feature Dimension.")   
    parser.add_argument("--which_gcn", default='0,0,1', type=str, 
                        help="which encoder block to have graph conv. Encoder1, Encoder2, Encoder3. Default: only Encoder3 has graph conv") 
    parser.add_argument("--mesh_type", default='body', type=str, help="body or hand") 
    parser.add_argument("--interm_size_scale", default=2, type=int)
    #########################################################
    # Others
    #########################################################
    parser.add_argument("--run_eval_only", default=True, action='store_true',) 
    parser.add_argument("--device", type=str, default='cuda', 
                        help="cuda or cpu")
    parser.add_argument('--seed', type=int, default=88, 
                        help="random seed for initialization.")

    args = parser.parse_args()
    return args


def main(args):
    global logger
    # Setup CUDA, GPU & distributed training
    args.num_gpus = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    os.environ['OMP_NUM_THREADS'] = str(args.num_workers)
    print('set os.environ[OMP_NUM_THREADS] to {}'.format(os.environ['OMP_NUM_THREADS']))

    args.distributed = args.num_gpus > 1
    args.device = torch.device(args.device)

    args.inputs = 'vcoco_000000298689.jpg'
    # vcoco_000000298689.jpg: cover image of skiing; vcoco_000000008383.jpg: segmented image of a bed and computer; vcoco_000000098596.jpg: bicycle image; vcoco_000000132430.jpg: lying down image; vcoco_000000258061.jpg: surfing image;
    mkdir(args.output_dir)
    logger = setup_logger("Graphormer", args.output_dir, get_rank())
    set_seed(args.seed, args.num_gpus)
    logger.info("Using {} GPUs".format(args.num_gpus))

    # Mesh and SMPL utils
    smpl = SMPL().to(args.device)
    mesh_sampler = Mesh()

    # Renderer for visualization
    #renderer = Renderer(faces=smpl.faces.cpu().numpy())

    # breakpoint()

    # Load model
    trans_encoder = []

    input_feat_dim = [int(item) for item in args.input_feat_dim.split(',')]
    hidden_feat_dim = [int(item) for item in args.hidden_feat_dim.split(',')]
    output_feat_dim = input_feat_dim[1:] + [3]

    # which encoder block to have graph convs
    which_blk_graph = [int(item) for item in args.which_gcn.split(',')]
    
    if args.run_eval_only==True and args.resume_checkpoint!=None and args.resume_checkpoint!='None' and 'state_dict' not in args.resume_checkpoint:
        # if only run eval, load checkpoint
        logger.info("Evaluation: Loading from checkpoint {}".format(args.resume_checkpoint))
        _model = torch.load(args.resume_checkpoint)
    else:
        # init three transformer-encoder blocks in a loop
        for i in range(len(output_feat_dim)):
            config_class, model_class = BertConfig, Graphormer
            config = config_class.from_pretrained(args.config_name if args.config_name \
                    else args.model_name_or_path)

            config.output_attentions = False
            config.img_feature_dim = input_feat_dim[i] 
            config.output_feature_dim = output_feat_dim[i]
            args.hidden_size = hidden_feat_dim[i]
            args.intermediate_size = int(args.hidden_size*args.interm_size_scale)

            if which_blk_graph[i]==1:
                config.graph_conv = True
                logger.info("Add Graph Conv")
            else:
                config.graph_conv = False

            config.mesh_type = args.mesh_type

            # update model structure if specified in arguments
            update_params = ['num_hidden_layers', 'hidden_size', 'num_attention_heads', 'intermediate_size']

            for idx, param in enumerate(update_params):
                arg_param = getattr(args, param)
                config_param = getattr(config, param)
                if arg_param > 0 and arg_param != config_param:
                    logger.info("Update config parameter {}: {} -> {}".format(param, config_param, arg_param))
                    setattr(config, param, arg_param)

            # init a transformer encoder and append it to a list
            assert config.hidden_size % config.num_attention_heads == 0
            model = model_class(config=config) 
            logger.info("Init model from scratch.")
            trans_encoder.append(model)

        # init ImageNet pre-trained backbone model
        if args.arch=='hrnet':
            hrnet_yaml = 'Path/to/GraphiContact/models/hrnet/cls_hrnet_w40_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
            hrnet_checkpoint = 'Path/to/GraphiContact/models/hrnet/hrnetv2_w40_imagenet_pretrained.pth'
            hrnet_update_config(hrnet_config, hrnet_yaml)
            backbone = get_cls_net_gridfeat(hrnet_config, pretrained=hrnet_checkpoint)
            logger.info('=> loading hrnet-v2-w40 model')
        elif args.arch=='hrnet-w64':
            hrnet_yaml = 'Path/to/GraphiContact/models/hrnet/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
            hrnet_checkpoint = 'Path/to/GraphiContact/models/hrnet/hrnetv2_w64_imagenet_pretrained.pth'
            hrnet_update_config(hrnet_config, hrnet_yaml)
            backbone = get_cls_net_gridfeat(hrnet_config, pretrained=hrnet_checkpoint)
            logger.info('=> loading hrnet-v2-w64 model')
        else:
            print("=> using pre-trained model '{}'".format(args.arch))
            backbone = models.__dict__[args.arch](pretrained=True)
            # remove the last fc layer
            backbone = torch.nn.Sequential(*list(backbone.children())[:-2])


        trans_encoder = torch.nn.Sequential(*trans_encoder)
        total_params = sum(p.numel() for p in trans_encoder.parameters())
        logger.info('Graphormer encoders total parameters: {}'.format(total_params))
        backbone_total_params = sum(p.numel() for p in backbone.parameters())
        logger.info('Backbone total parameters: {}'.format(backbone_total_params))

        # build end-to-end Graphormer network (CNN backbone + multi-layer graphormer encoder)
        _model = Graphormer_Network(args, config, backbone, trans_encoder, trans_encoder, mesh_sampler)

        if args.resume_checkpoint!=None and args.resume_checkpoint!='None':
            # for fine-tuning or resume training or inference, load weights from checkpoint
            logger.info("Loading state dict from checkpoint {}".format(args.resume_checkpoint))
            # workaround approach to load sparse tensor in graph conv.
            states = torch.load(args.resume_checkpoint)
            # states = checkpoint_loaded.state_dict()
            for k, v in states.items():
                states[k] = v.cpu()
            # del checkpoint_loaded
            _model.load_state_dict(states, strict=False)
            del states
            gc.collect()
            torch.cuda.empty_cache()

    # update configs to enable attention outputs
    setattr(_model.trans_encoder[-1].config,'output_attentions', True)
    setattr(_model.trans_encoder[-1].config,'output_hidden_states', True)
    _model.trans_encoder[-1].bert.encoder.output_attentions = True
    _model.trans_encoder[-1].bert.encoder.output_hidden_states =  True
    for iter_layer in range(4):
        _model.trans_encoder[-1].bert.encoder.layer[iter_layer].attention.self.output_attentions = True
    for inter_block in range(3):
        setattr(_model.trans_encoder[-1].config,'device', args.device)

    _model.to(args.device)
    """"""
    if args.deco_resume_checkpoint is not None:
        ckpt = torch.load(args.deco_resume_checkpoint)
        _model.load_state_dict(ckpt, strict=False)
    logger.info("Run inference")

    image_list = []
    if not args.image_file_or_path:
        raise ValueError("image_file_or_path not specified")
    if op.isfile(args.image_file_or_path):
        image_list = [args.image_file_or_path]
    elif op.isdir(args.image_file_or_path):
        # should be a path with images only
        for filename in os.listdir(args.image_file_or_path):
            if filename.endswith(".png") or filename.endswith(".jpg") and 'pred' not in filename:
                image_list.append(args.image_file_or_path+'/'+filename) 
    else:
        raise ValueError("Cannot find images at {}".format(args.image_file_or_path))

    run_inference(args, image_list, _model, smpl, mesh_sampler)   

if __name__ == "__main__":
    args = parse_args()
    main(args)
