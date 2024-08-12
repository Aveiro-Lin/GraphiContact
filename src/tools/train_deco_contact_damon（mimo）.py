#from __future__ import absolute_import, division, print_functionhom
import argparse
import os
import os.path as op
import code
import json
import time
import datetime
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  
import sys
sys.path.append('Path/to/GraphiContact')
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.utils import make_grid
if not torch.cuda.is_available():
    raise SystemError('CUDA is not available.')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
import gc
import numpy as np

import cv2
from src2.modeling.bert import BertConfig, Graphormer
from src2.modeling.bert import Graphormer_Body_Network as Graphormer_Network
# from src2.modeling.bert import Graphormer_Body_Network_1 as Graphormer_Network_1
# from src2.modeling.bert import Graphormer_Body_Network_2 as Graphormer_Network_2
from src2.modeling._smpl import SMPL, Mesh
from torch.optim.lr_scheduler import MultiStepLR
from src2.modeling.hrnet.hrnet_cls_net_gridfeat import get_cls_net_gridfeat
from src2.modeling.hrnet.config import config as hrnet_config
from src2.modeling.hrnet.config import update_config as hrnet_update_config
import src2.modeling.data.config as cfg
from src2.datasets.build import make_data_loader

from src2.utils.logger import setup_logger
from src2.utils.comm import synchronize, is_main_process, get_rank, get_world_size, all_gather
from src2.utils.miscellaneous import mkdir, set_seed
# from src2.utils.metric_logger import AverageMeter, EvalMetricslogger
#from src2.utils.renderer import Renderer, visualize_reconstruction_and_att_local, visualize_reconstruction_no_text
from src2.utils.metric_pampjpe import reconstruction_error
from src2.utils.geometric_layers import orthographic_projection
from PIL import Image
from torchvision import transforms
from src2.utils.evaluator import evaluator

from torchinfo import summary
from tqdm import tqdm

transform = transforms.Compose([           
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])])

transform_1d = transforms.Compose([           
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485],
                        std=[0.229])])

def parse_args():
    parser = argparse.ArgumentParser()
    #########################################################
    # Data related arguments
    #########################################################
    parser.add_argument("--num_workers", default=4, type=int, 
                        help="Workers in dataloader.")
    parser.add_argument("--img_scale_factor", default=1, type=int, 
                        help="adjust image resolution.")
    parser.add_argument("--image_file_or_path", default='./samples/human-body_deco', type=str, 
                        help="test data") 
    #########################################################
    # Loading/saving checkpoints
    #########################################################
    parser.add_argument("--model_name_or_path", default='Path/to/GraphiContact/src/modeling/bert/bert-base-uncased', type=str, required=False,
                        help="Path to pre-trained transformer model or model type.")
    # checkpoint here, you can fill in the original ckpt or the training ckpt, the original is./models/graphormer_release/graphormer_3dpw_state_dict.bin  新的/home/linxj/project/MeshGraphormer/ckpt/deco_grph.pt
    parser.add_argument("--resume_checkpoint", default='Path/to/GraphiContact/models/graphormer_release/graphormer_3dpw_state_dict.bin', type=str, required=False,
                        help="Path to specific checkpoint for resume training.")
    parser.add_argument("--resume_checkpoint2", default='Path/to/GraphiContact/models/graphormer_release/graphormer_h36m_state_dict.bin', type=str, required=False,
                        help="Path to specific checkpoint for resume training.")
    parser.add_argument("--config_name", default="", type=str, 
                        help="Pretrained config name or path if not the same jas model_name.")
    parser.add_argument("--output_dir", default='Path/to/GraphiContact/ckpt', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    #########################################################
    # Model architectures
    #########################################################
    parser.add_argument('-a', '--arch', default='hrnet-w64',
                    help='CNN backbone architecture: hrnet-w64, hrnet, resnet50')
    parser.add_argument("--num_hidden_layers", default=2, type=int, required=False, 
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
    parser.add_argument("--run_eval_only", default=False, action='store_true',) 
    parser.add_argument("--device", type=str, default='cuda:0', 
                        help="cuda or cpu")
    parser.add_argument('--seed', type=int, default=88, 
                        help="random seed for initialization.")
    #########################################################
    # Here are some hyperparameters to set for the training
    #########################################################
    # parser.add_argument("--lr", default=5e-5) 
    parser.add_argument("--lr", default=8e-3, type=float) 
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=2000)
    
    parser.add_argument("--n_infers", type=int, default=3, help="number of inferences for each batch")
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
    
    mkdir(args.output_dir)
    # logger = setup_logger("Graphormer", args.output_dir, get_rank())
    set_seed(args.seed, args.num_gpus)
    # logger.info("Using {} GPUs".format(args.num_gpus))

    _model, smpl, mesh_sampler = load_models(args=args)
    # update configs to enable attention outputs
    # assert False, (_model, smpl, mesh_sampler)
    

    dev = args.device
    _model.to(dev)
    smpl.to(dev)
    # summary(_model, input_data=(torch.randn(1, 3, 224, 224).to(dev), smpl, mesh_sampler))
    # print(smpl)
    # print(mesh_sampler)
    # assert False

    batch_size = args.batch_size
    # logger.info("Run training")
    lr = args.lr
    num_epochs = args.num_epochs
    
    
    loss_fn = nn.BCELoss()
        
    # Only the newly added parameters are trained, leaving the original parameters unchanged
    optimizer = torch.optim.Adam(_model.contact_parameters(),lr=lr)
    # print(_model.contact_parameters)
    scheduler = MultiStepLR(optimizer, milestones=[40, 80], gamma=0.1)
    
    if args.run_eval_only:
        all_pred = []
        all_gt = []
        test_data = prepare_dataset_test(batch_size)
        for idx, batch in enumerate(test_data):
            pred_ana, pred_ana2, ana_tensor = test_batch(model=_model, smpl=smpl, mesh_sampler=mesh_sampler,
                            batch=batch, optimizer=optimizer, loss_fn=loss_fn, dev=dev)
            all_pred.append(pred_ana)
            all_gt.append(ana_tensor)
        pre, rec, f1, fp_geo_err, fn_geo_err = evaluator(all_pred, all_gt)
        print('pre:', pre.cpu().numpy(), 'rec:', rec.cpu().numpy(), "f1:", f1.cpu().numpy(), 'fp_geo_err:', fp_geo_err.cpu().numpy(), 'fn_geo_err:', fn_geo_err.cpu().numpy())
        
    for i in range(9):
        train_data, len_train_data = prepare_dataset(batch_size, n_infers=args.n_infers)
        # for idx, batch in enumerate(tqdm(train_data, total=len_train_data)):
        for idx, batch in enumerate(train_data):
            try:
            
                loss = train_batch(model=_model, smpl=smpl, mesh_sampler=mesh_sampler,
                                batch=batch, optimizer=optimizer, scheduler=scheduler, loss_fn=loss_fn, dev=dev,
                                n_infers=args.n_infers)
                if (idx % 100) == 0:
                    print(loss.data.cpu().numpy())
                
            # org_img_batch, ana_contact_batch = batch
            # if len(org_img_batch) >= 1:
            #     org_img_arry = [transform(Image.open(image_file)) for image_file in org_img_batch]

            #     org_tensor = torch.tensor([item.cpu().detach().numpy() for item in org_img_arry]).float()
            #     ana_tensor = torch.tensor(ana_contact_batch).float()
            
            #     _, _, _, _, _, pred_ana = _model(org_tensor.to(dev), smpl, mesh_sampler)
            
            #     loss = loss_fn(pred_ana, ana_tensor.to(dev))
            #     optimizer.zero_grad()
            #     loss.backward()
            #     torch.nn.utils.clip_grad_norm_(_model.parameters(),1.)
            #     optimizer.step()
            #     print(loss.data.cpu().numpy())
            #     scheduler.step()
                # return loss.data
            except Exception as e:
                print('??')
                print(e)
                # raise e
                continue

        if (i) % 1 == 0:
            print(i)
            all_pred = []
            all_gt = []
            all_pred2 = []
            all_pred3 = []
            all_gt2 = []
            test_data = prepare_dataset_test(batch_size, n_infers=args.n_infers)

            for idx, batch in enumerate(test_data):
                pred_ana, pred_ana2, ana_tensor  = test_batch(model=_model, smpl=smpl, mesh_sampler=mesh_sampler,
                                batch=batch, optimizer=optimizer, loss_fn=loss_fn, dev=dev, n_infers=args.n_infers)
                
                                
                # org_img_batch, ana_contact_batch = batch
                # #if len(org_img_batch) >= 1:
                # org_img_arry = [transform(Image.open(image_file)) for image_file in org_img_batch]
    
                # org_tensor = torch.tensor([item.cpu().detach().numpy() for item in org_img_arry]).float()
                # ana_tensor = torch.tensor(ana_contact_batch).float()

                # _, _, _, _, _, pred_ana = _model(org_tensor.to(dev), smpl, mesh_sampler)
                
                all_pred2.append(pred_ana.cpu().numpy())
                all_pred3.append(pred_ana2.cpu().numpy())
                #pred_ana = pred_ana*0.58 + pred_ana2*0.57
                pred_ana[pred_ana > 0.45] = 1
                # pred_ana[pred_ana < 0.59] = 0
                all_pred.append(pred_ana)
                
                all_gt.append(ana_tensor)
                
                all_gt2.append(ana_tensor.cpu().numpy())
            

            pre, rec, f1, fp_geo_err, fn_geo_err = evaluator(all_pred, all_gt)
            print('pre:', pre.cpu().numpy(), 'rec:', rec.cpu().numpy(), "f1:", f1.cpu().numpy(), 'fp_geo_err:', fp_geo_err.cpu().numpy(), 'fn_geo_err:', fn_geo_err.cpu().numpy())

    # Model saving path after training
            torch.save(_model.state_dict(), 'Path/to/GraphiContact/ckpt/deco_grph_damon.pt')




def mask_split(img, num_parts):
    if not len(img.shape) == 2:
        img = img[:, :, 0]
    mask = np.zeros((img.shape[0], img.shape[1], num_parts))
    for i in np.unique(img):
        mask[:, :, i] = np.where(img == i, 1., 0.)
    return np.transpose(mask, (2, 0, 1))
        
# Train Function
def train_batch(model, smpl, mesh_sampler, batch, optimizer, scheduler, loss_fn, dev='cuda', n_infers=1):
    
    org_img_batchs, seg_batchs, part_batchs, ana_contact_batchs = batch
    loss = 0
    # org_img_batch, seg_batch, part_batch, ana_contact_batch = org_img_batchs[i_infers], seg_batchs[i_infers], part_batchs[i_infers], ana_contact_batchs[i_infers]
    if len(org_img_batchs[0]) >= 1:
        
        org_img_arry = [[transform(Image.open(image_file)) for image_file in org_img_batch] for org_img_batch in org_img_batchs]
        org_seg_arry = [[mask_split(cv2.resize(cv2.imread(image_file), (224, 224), cv2.INTER_CUBIC), 133) for image_file in seg_batch] for seg_batch in seg_batchs]
        org_part_arry = [[mask_split(cv2.resize(cv2.imread(image_file), (224, 224), cv2.INTER_CUBIC), 26) for image_file in part_batch] for part_batch in part_batchs]

        seg_tensor = torch.tensor(org_seg_arry, dtype=torch.float32)
        part_tensor = torch.tensor(org_part_arry, dtype=torch.float32)

        org_tensor = torch.tensor([[item.cpu().detach().numpy() for item in arr] for arr in org_img_arry]).float()
        # assert org_tensor.shape == (3, 4, 3, 224, 224), org_tensor.shape
        ana_tensor = torch.tensor(ana_contact_batchs).float()

        _, _, seg_pred, part_pred, _, pred_ana, pred_ana2 = model(org_tensor.to(dev), smpl, mesh_sampler)
        # print(sum(pred_ana))
        # print(sum(ana_tensor))
        
        # print(pred_ana)
        # print(dev)
    else:
        return 0
    loss = 0
    for i_infers in range(n_infers):
        loss += loss_fn(pred_ana[i_infers], ana_tensor[i_infers].to(dev)) / n_infers
        loss += loss_fn(pred_ana2[i_infers], ana_tensor[i_infers].to(dev)) * 0.1 / n_infers
        loss += loss_fn(seg_pred[i_infers], seg_tensor[i_infers].to(dev)) / n_infers
        loss += loss_fn(part_pred[i_infers], part_tensor[i_infers].to(dev)) / n_infers
        
        
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    optimizer.step()
    # scheduler.step()
    return loss.data
    
# Test Function
@torch.no_grad()
def test_batch(model, smpl, mesh_sampler, batch, optimizer, loss_fn, dev='cuda', n_infers=3):
    org_img_batchs, seg_batchs, part_batchs, ana_contact_batchs = batch
    #if len(org_img_batch) >= 1:

    org_img_arry = [[transform(Image.open(image_file)) for image_file in org_img_batch] for org_img_batch in org_img_batchs]
    org_seg_arry = [[mask_split(cv2.resize(cv2.imread(image_file), (224, 224), cv2.INTER_CUBIC), 133) for image_file in seg_batch] for seg_batch in seg_batchs]
    org_part_arry = [[mask_split(cv2.resize(cv2.imread(image_file), (224, 224), cv2.INTER_CUBIC), 26) for image_file in part_batch] for part_batch in part_batchs]

    seg_tensor = torch.tensor(org_seg_arry, dtype=torch.float32)
    part_tensor = torch.tensor(org_part_arry, dtype=torch.float32)

    org_tensor = torch.tensor([[item.cpu().detach().numpy() for item in arr] for arr in org_img_arry]).float()
    ana_tensor = torch.tensor(ana_contact_batchs).float()
    
    _, _, seg_pred, part_pred, _, pred_ana, pred_ana2 = model(org_tensor.to(dev), smpl, mesh_sampler)


    # _, _, _, _, _, pred_ana, pred_ana2 = model(org_tensor.to(dev), smpl, mesh_sampler)

        # loss = loss_fn(pred_ana, ana_tensor.to(dev))
        # optimizer.zero_grad()
        # loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    pred_ana = pred_ana.mean(0)
    pred_ana2 = pred_ana2.mean(0)
    ana_tensor = ana_tensor.mean(0)

    return pred_ana, pred_ana2, ana_tensor

from PIL import Image


# def prepare_dataset(batch_size, data_path='Path/to/GraphiContact/datasets/rich/cam_00/',
#                     label_path='Path/to/GraphiContact/datasets/rich/lb_LectureHall_003_wipingchairs1/'):
#     org_img_paths = []
#     org_label_paths = []
#     org_img_paths_raw = os.listdir(data_path)
    

#     for i in org_img_paths_raw:
        
#         if int(i.split('_')[0]) <= 500:
#             temp_l = label_path + i.split('_')[0] + '/003_contact.npz'
#             label_t = np.load(temp_l, allow_pickle=True)['arr_0']
#             org_img_paths.append(os.path.join(data_path, i))
#             org_label_paths.append(label_t)
            
        
#     batch_size = 4
#     train_data = org_img_paths
#     contact_label = org_label_paths
#     train_data = [data for data in train_data if len(train_data) >= 5]
#     contact_label = [data for data in contact_label if len(data) == 6890]
#     print('total training length: ' + str(len(train_data)))
    
#     batch_input = []
#     for i in range(0, int(len(train_data)/batch_size)):
#         data_batch = train_data[i*batch_size:(i+1)*batch_size]
#         label_batch = contact_label[i*batch_size:(i+1)*batch_size]
#         batch_input.append((data_batch,label_batch))
#     return iter(batch_input)


# def prepare_dataset_test(batch_size, data_path='Path/to/GraphiContact/datasets/rich/cam_00/',
#                     label_path='Path/to/GraphiContact/datasets/rich/lb_LectureHall_003_wipingchairs1/'):
#     org_img_paths_raw = os.listdir(data_path)
#     org_img_paths = []
#     org_label_paths = []
#     for i in org_img_paths_raw:
        
#         if int(i.split('_')[0]) > 500 and int(i.split('_')[0])  < 700:
#             temp_l = label_path + i.split('_')[0] + '/003_contact.npz'
#             label_t = np.load(temp_l, allow_pickle=True)['arr_0']
#             org_img_paths.append(os.path.join(data_path, i))
#             org_label_paths.append(label_t)
            
#     train_data = org_img_paths
#     contact_label = org_label_paths
#     train_data = [data for data in train_data if len(train_data) >= 5]
#     contact_label = [data for data in contact_label if len(data) == 6890]
#     print('total test length: ' + str(len(train_data)))
    
#     batch_input = []
#     for i in range(0, int(len(train_data)/batch_size)):
#         data_batch = train_data[i*batch_size:(i+1)*batch_size]
#         label_batch = contact_label[i*batch_size:(i+1)*batch_size]
#         batch_input.append((data_batch,label_batch))
#     return iter(batch_input)
 
# 准备3d接触信息的数据集
# def prepare_dataset(batch_size, data_path='Path/to/GraphiContact/datasets/behave/Date01_Sub01_basketball/',
#                     label_path='Path/to/GraphiContact/HOT-Annotated/Release_Datasets/damon/hot_dca_trainval.npz'):
#     org_img_paths = []
#     org_label_paths = []
#     org_img_paths_raw = os.listdir(data_path)
    
#     for i in org_img_paths_raw:
#         if i != 'info.json' and int(i.split('.')[0].split('t')[-1]) <= 40:
#             temp_l = os.path.join(data_path, i) + '/' + 'sports ball/fit01/sports ball_fit_contact.npz'
#             label_t = np.load(temp_l, allow_pickle=True)['arr_0']
#             org_label_paths.append(label_t[:6890])
#             org_img_paths.append(os.path.join(data_path, i) + '/k0.color.jpg')
#             org_label_paths.append(label_t[:6890])
#             org_img_paths.append(os.path.join(data_path, i) + '/k1.color.jpg')
#             org_label_paths.append(label_t[:6890])
#             org_img_paths.append(os.path.join(data_path, i) + '/k2.color.jpg')
#             org_label_paths.append(label_t[:6890])
#             org_img_paths.append(os.path.join(data_path, i) + '/k3.color.jpg')
            
            

#     batch_size = 4
#     train_data = org_img_paths
#     contact_label = org_label_paths
#     train_data = [data for data in train_data if len(train_data) >= 5]
#     contact_label = [data for data in contact_label if len(data) == 6890]
#     print('total training length: ' + str(len(train_data)))
    
#     batch_input = []
#     for i in range(0, int(len(train_data)/batch_size)):
#         data_batch = train_data[i*batch_size:(i+1)*batch_size]
#         label_batch = contact_label[i*batch_size:(i+1)*batch_size]
#         batch_input.append((data_batch,label_batch))
#     return iter(batch_input)


# def prepare_dataset_test(batch_size, data_path='Path/to/GraphiContact/datasets/behave/Date01_Sub01_basketball/',
#                     label_path='Path/to/GraphiContact/HOT-Annotated/Release_Datasets/damon/hot_dca_trainval.npz'):
#     org_img_paths_raw = os.listdir(data_path)
#     org_img_paths = []
#     org_label_paths = []
#     for i in org_img_paths_raw:
#         if i != 'info.json' and int(i.split('.')[0].split('t')[-1]) > 40:
#             temp_l = os.path.join(data_path, i) + '/' + 'sports ball/fit01/sports ball_fit_contact.npz'
#             # temp_l = os.path.join(data_path, i) + '/' + 'boxlarge/fit01/boxlarge_fit_contact.npz'
#             label_t = np.load(temp_l, allow_pickle=True)['arr_0']
#             org_label_paths.append(label_t[:6890])
#             org_img_paths.append(os.path.join(data_path, i) + '/k0.color.jpg')
#             org_label_paths.append(label_t[:6890])
#             org_img_paths.append(os.path.join(data_path, i) + '/k1.color.jpg')
#             org_label_paths.append(label_t[:6890])
#             org_img_paths.append(os.path.join(data_path, i) + '/k2.color.jpg')
#             org_label_paths.append(label_t[:6890])
#             org_img_paths.append(os.path.join(data_path, i) + '/k3.color.jpg')
            
#     train_data = org_img_paths
#     contact_label = org_label_paths
#     train_data = [data for data in train_data if len(train_data) >= 5]
#     contact_label = [data for data in contact_label if len(data) == 6890]
#     print('total test length: ' + str(len(train_data)))
    
#     batch_input = []
#     for i in range(0, int(len(train_data)/batch_size)):
#         data_batch = train_data[i*batch_size:(i+1)*batch_size]
#         label_batch = contact_label[i*batch_size:(i+1)*batch_size]
#         batch_input.append((data_batch,label_batch))
#     return iter(batch_input)





def prepare_dataset(batch_size, total_num=4384, data_path='Path/to/GraphiContact/HOT-Annotated/images',
                    label_path='Path/to/GraphiContact/HOT-Annotated/Release_Datasets/damon/hot_dca_trainval.npz',
                    n_infers=3):
    total_num = total_num
    org_img_paths = os.listdir(data_path)
    org_img_paths = [os.path.join(data_path, path) for path in org_img_paths]

    seg_path='Path/to/GraphiContact/HOT-Annotated/segments'
    org_seg_paths = os.listdir(seg_path)
    seg_img_paths = [os.path.join(seg_path, path) for path in org_seg_paths]
    seg_path='Path/to/GraphiContact/HOT-Annotated/segments'
    part_path='Path/to/GraphiContact/datasets/Damon_Scene/damon_segmentations/parts/training'
    org_part_paths = os.listdir(part_path)
    part_img_paths = [os.path.join(part_path, path) for path in org_part_paths]

    # batch_size = 4
    train_label = np.load(label_path)
    train_imgs_names = train_label['imgname']
    seg_imgs_names = train_label['scene_seg']
    part_imgs_names = train_label['part_seg']
    print(part_imgs_names)
    train_imgs_names = [img.split('/')[-1] for img in train_imgs_names]
    seg_imgs_names = [img.split('/')[-1] for img in seg_imgs_names]
    part_imgs_names = [img.split('/')[-1] for img in part_imgs_names]
    # The key corresponding to the smpl annotation here is contact_label
    train_3d_con_label = train_label['contact_label']

    train_data = []
    seg_data = []
    part_data = []
    contact_label = []
    print('org_img_paths', len(org_img_paths))
    print('seg_img_paths', len(seg_img_paths))
    print('part_img_paths', len(part_img_paths))
    # for img_path in org_img_paths:
    #     for i in range(len(train_imgs_names)):
    #         if train_imgs_names[i] in img_path:
    #             train_data.append(img_path)
    #             contact_label.append(train_3d_con_label[i])
    #     if len(train_data) >= total_num:
    #         break
    # for img_path in seg_img_paths:    
    #     for i in range(len(seg_imgs_names)):
    #         if seg_imgs_names[i] in img_path:
    #             seg_data.append(img_path)
    # for img_path in part_img_paths:  
    #     for i in range(len(part_imgs_names)):
    #         if part_imgs_names[i] in img_path:
    #             part_data.append(img_path)
    train_data = [op.join(data_path, img_name) for img_name in train_imgs_names]
    contact_label = [data for data in train_3d_con_label]
    assert len(train_data) == total_num, len(train_data)
    part_data = [op.join(part_path, img_name) for img_name in part_imgs_names]
    seg_data = [op.join(seg_path, img_name) for img_name in seg_imgs_names]
    print(len(train_data))
    print('seg', len(seg_data))
    print('part', len(part_data))
    # train_data = train_data[:total_num]
    # contact_label = contact_label[:total_num]
    train_data = [data for data in train_data if len(train_data) >= 5]
    contact_label = [data for data in contact_label if len(data) == 6890]
    print('total training length: ' + str(len(train_data)))
    
    # generate batch data that every batch contains 4*n_infers images
    # modify dataset for multi input multi output (MIMO) here
    # batch_input = [batch_size, n_infers, 224, 224, 3]
    # Each batch contains batch * n_infers samples. In order to traverse all samples, the first sample in n_infers is followed
    # Data sets are acquired sequentially, and the remaining n_infers-1 samples are randomly selected to allow duplicate samples
    batch_input = []
    for i in range(0, int(len(train_data)/batch_size)):
        data_batch = []
        seg_batch = []
        part_batch = []
        label_batch = []
        data_batch.append(train_data[i*batch_size:(i+1)*batch_size])
        seg_batch.append(seg_data[i*batch_size:(i+1)*batch_size])
        part_batch.append(part_data[i*batch_size:(i+1)*batch_size])
        label_batch.append(contact_label[i*batch_size:(i+1)*batch_size])

        for j in range(n_infers - 1):
            rand_idx = np.random.choice(len(train_data), batch_size, replace=True)
            data_batch.append([train_data[idx] for idx in rand_idx])
            seg_batch.append([seg_data[idx] for idx in rand_idx])
            part_batch.append([part_data[idx] for idx in rand_idx])  
            label_batch.append([contact_label[idx] for idx in rand_idx])
        batch_input.append((data_batch, seg_batch, part_batch, label_batch))
    return iter(batch_input), len(batch_input)

def prepare_dataset_test(batch_size, data_path='Path/to/GraphiContact/HOT-Annotated/images',
                    label_path='Path/to/GraphiContact/HOT-Annotated/Release_Datasets/damon/hot_dca_test.npz',
                    n_infers=3):
    
    org_img_paths = os.listdir(data_path)
    org_img_paths = [os.path.join(data_path, path) for path in org_img_paths]

    seg_path='Path/to/GraphiContact/HOT-Annotated/segments'
    org_seg_paths = os.listdir(seg_path)
    seg_img_paths = [os.path.join(seg_path, path) for path in org_seg_paths]

    part_path='Path/to/GraphiContact/datasets/Damon_Scene/damon_segmentations/parts/training'
    org_part_paths = os.listdir(part_path)
    part_img_paths = [os.path.join(part_path, path) for path in org_part_paths]

    
    
    test_label = np.load(label_path)
    test_imgs_names = test_label['imgname']
    test_imgs_names = [img.split('/')[-1] for img in test_imgs_names]
    # The key corresponding to the smpl annotation here is contact_label
    test_3d_con_label = test_label['contact_label']

    seg_imgs_names = test_label['scene_seg']
    part_imgs_names = test_label['part_seg']

    test_data = []
    seg_data = []
    part_data = []
    contact_label = []
    for img_path in seg_img_paths:    
        for i in range(len(seg_imgs_names)):
            if seg_imgs_names[i] in img_path:
                seg_data.append(img_path)
    for img_path in part_img_paths:  
        for i in range(len(part_imgs_names)):
            if part_imgs_names[i] in img_path:
                part_data.append(img_path)
    print('seg', len(seg_data))
    print('part', len(part_data))
    for img_path in org_img_paths:
        for i in range(len(test_imgs_names)):
            if test_imgs_names[i] in img_path:
                test_data.append(img_path)
                contact_label.append(test_3d_con_label[i])
    test_data = [data for data in test_data if len(test_data) >= 5]
    contact_label = [data for data in contact_label if len(data) == 6890]
    print('total test length: ' + str(len(test_data)))
    
    batch_input = []
    for i in range(0, int(len(test_data)/batch_size)):
        # data_batch = test_data[i*batch_size:(i+1)*batch_size]
        # seg_batch = seg_data[i*batch_size:(i+1)*batch_size]
        # part_batch = part_data[i*batch_size:(i+1)*batch_size]
        # label_batch = contact_label[i*batch_size:(i+1)*batch_size]
        # batch_input.append((data_batch, seg_batch, part_batch, label_batch))
        data_batch = []
        seg_batch = []
        part_batch = []
        label_batch = []
        data_batch.append(test_data[i*batch_size:(i+1)*batch_size])
        seg_batch.append(seg_data[i*batch_size:(i+1)*batch_size])
        part_batch.append(part_data[i*batch_size:(i+1)*batch_size])
        label_batch.append(contact_label[i*batch_size:(i+1)*batch_size])

        for j in range(n_infers - 1):
            data_batch.append(test_data[i*batch_size:(i+1)*batch_size])
            seg_batch.append(seg_data[i*batch_size:(i+1)*batch_size])
            part_batch.append(part_data[i*batch_size:(i+1)*batch_size])
            label_batch.append(contact_label[i*batch_size:(i+1)*batch_size])
        batch_input.append((data_batch, seg_batch, part_batch, label_batch))
    return iter(batch_input)


def load_models(args):
    smpl = SMPL().to(args.device)
    mesh_sampler = Mesh()

    # Load model
    trans_encoder = []
    trans_encoder2 = []
    input_feat_dim = [int(item) for item in args.input_feat_dim.split(',')]
    hidden_feat_dim = [int(item) for item in args.hidden_feat_dim.split(',')]
    output_feat_dim = input_feat_dim[1:] + [3]

    # which encoder block to have graph convs
    which_blk_graph = [int(item) for item in args.which_gcn.split(',')]

    # init three transformer-encoder blocks in a loop
    for i in range(3):
        config_class, model_class = BertConfig, Graphormer
        config = config_class.from_pretrained(args.config_name if args.config_name \
                else args.model_name_or_path)
        pretrained_dict = torch.load(args.resume_checkpoint)
        pretrained_dictt = torch.load(args.resume_checkpoint2)
        
        # print(model.state_dict())
        config.output_attentions = False
        config.img_feature_dim = input_feat_dim[i] 
        config.output_feature_dim = output_feat_dim[i]
        args.hidden_size = hidden_feat_dim[i]
        args.intermediate_size = int(args.hidden_size*args.interm_size_scale)

        if which_blk_graph[i]==1:
            config.graph_conv = True
            # logger.info("Add Graph Conv")
        else:
            config.graph_conv = False

        config.mesh_type = args.mesh_type

        # update model structure if specified in arguments
        update_params = ['num_hidden_layers', 'hidden_size', 'num_attention_heads', 'intermediate_size']

        for idx, param in enumerate(update_params):
            arg_param = getattr(args, param)
            config_param = getattr(config, param)
            if arg_param > 0 and arg_param != config_param:
                # logger.info("Update config parameter {}: {} -> {}".format(param, config_param, arg_param))
                setattr(config, param, arg_param)

        # init a transformer encoder and append it to a list
        assert config.hidden_size % config.num_attention_heads == 0
        model = model_class(config=config)
        #graphormer's three-tier encoder also uses a pre-trained model
        #Two sets of Encoders are initialized with two pre-trained weights, 3dpw and h36m, and the output is averaged.
        model_dict = model.state_dict()
        # for k, v in pretrained_dict.items():
        #     if k.split('trans_encoder.2.')[-1] in model_dict.keys():
        #         print(k)
        # print(model)
        out = ['cls_head.weight', 'cls_head.bias', 'residual.weight', 'residual.bias']
        out = []
        if i < 3:
            pretrained_dict2 = {k.split('trans_encoder.'+str(i)+'.')[-1]: v for k, v in pretrained_dict.items() if k.split('trans_encoder.'+str(i)+'.')[-1] in model_dict.keys()\
            and k.split('trans_encoder.'+str(i)+'.')[-1] not in out }
            model_dict.update(pretrained_dict2)
            model.load_state_dict(model_dict)
        # logger.info("Init model from scratch.")
        trans_encoder.append(model)
        
        model2 = model_class(config=config)
        model_dict = model2.state_dict()
        # for k, v in pretrained_dict.items():
        #     if k.split('trans_encoder.2.')[-1] in model_dict.keys():
        #         print(k)
        # print(model)
        out = ['cls_head.weight', 'cls_head.bias', 'residual.weight', 'residual.bias']
        out = []
        if i < 3:
            pretrained_dict2 = {k.split('trans_encoder.'+str(i)+'.')[-1]: v for k, v in pretrained_dictt.items() if k.split('trans_encoder.'+str(i)+'.')[-1] in model_dict.keys()\
            and k.split('trans_encoder.'+str(i)+'.')[-1] not in out }
            model_dict.update(pretrained_dict2)
            model2.load_state_dict(model_dict)
        # logger.info("Init model from scratch.")
        trans_encoder2.append(model2)

    # init ImageNet pre-trained backbone model
    dir = 'Path/to/GraphiContact/'
    if args.arch=='hrnet':
        # hrnet_yaml = 'models/hrnet/cls_hrnet_w40_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
        # hrnet_checkpoint = 'models/hrnet/hrnetv2_w40_imagenet_pretrained.pth'
        hrnet_yaml = op.join(dir, 'models/hrnet/cls_hrnet_w40_sgd_lr5e-2_wd1e-4_bs32_x100.yaml')
        hrnet_checkpoint = op.join(dir, 'models/hrnet/hrnetv2_w40_imagenet_pretrained.pth')
        hrnet_update_config(hrnet_config, hrnet_yaml)
        backbone = get_cls_net_gridfeat(hrnet_config, pretrained=hrnet_checkpoint)
        # logger.info('=> loading hrnet-v2-w40 model')
    elif args.arch=='hrnet-w64':
        # hrnet_yaml = 'models/hrnet/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
        # hrnet_checkpoint = 'models/hrnet/hrnetv2_w64_imagenet_pretrained.pth'
        hrnet_yaml = op.join(dir, 'models/hrnet/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml')
        hrnet_checkpoint = op.join(dir, 'models/hrnet/hrnetv2_w64_imagenet_pretrained.pth')
        hrnet_update_config(hrnet_config, hrnet_yaml)
        backbone = get_cls_net_gridfeat(hrnet_config, pretrained=hrnet_checkpoint)
        # logger.info('=> loading hrnet-v2-w64 model')
    else:
        print("=> using pre-trained model '{}'".format(args.arch))
        backbone = models.__dict__[args.arch](pretrained=True)
        # remove the last fc layer
        backbone = torch.nn.Sequential(*list(backbone.children())[:-1])


    trans_encoder = torch.nn.Sequential(*trans_encoder)
    total_params = sum(p.numel() for p in trans_encoder.parameters())
    # logger.info('Graphormer encoders total parameters: {}'.format(total_params))
    
    trans_encoder2 = torch.nn.Sequential(*trans_encoder2)
    total_params = sum(p.numel() for p in trans_encoder2.parameters())
    # logger.info('Graphormer encoders total parameters: {}'.format(total_params))
    
    backbone_total_params = sum(p.numel() for p in backbone.parameters())
    # logger.info('Backbone total parameters: {}'.format(backbone_total_params))

    # build end-to-end Graphormer network (CNN backbone + multi-layer graphormer encoder)
    _model = Graphormer_Network(args, config, backbone, trans_encoder, trans_encoder2, mesh_sampler)

    # if args.resume_checkpoint!=None and args.resume_checkpoint!='None':
    #     # for fine-tuning or resume training or inference, load weights from checkpoint
    #     logger.info("Loading state dict from checkpoint {}".format(args.resume_checkpoint))
    #     # workaround approach to load sparse tensor in graph conv.
    #     states = torch.load(args.resume_checkpoint)
    #     # states = checkpoint_loaded.state_dict()
    #     for k, v in states.items():
    #         states[k] = v.cpu()
    #     # del checkpoint_loaded
    #     _model.load_state_dict(states, strict=False)
    #     del states
    #     gc.collect()
    #     torch.cuda.empty_cache()
    return _model, smpl, mesh_sampler

if __name__ == "__main__":
    args = parse_args()
    main(args)
