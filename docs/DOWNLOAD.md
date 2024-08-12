# Download

## Getting Started

1. Create folders that store pretrained models, datasets, and predictions.
    ```bash
    export REPO_DIR=$PWD
    mkdir -p $REPO_DIR/models  # pre-trained models
    mkdir -p $REPO_DIR/datasets  # datasets
    mkdir -p $REPO_DIR/predictions  # prediction outputs
    ```

2. Download pretrained models.

    Our pre-trained models can be downloaded with the following command.
    ```bash
    cd $REPO_DIR
    bash scripts/download_models.sh
    ```
    The scripts will download three models that are trained for mesh reconstruction on Human3.6M, 3DPW, and FreiHAND, respectively. For your convenience, this script will also download HRNet pre-trained weights, which will be used in training. 

    The resulting data structure should follow the hierarchy as below. 
    ```
    ${REPO_DIR}  
    |-- models  
    |   |-- graphormer_release
    |   |   |-- graphormer_h36m_state_dict.bin
    |   |   |-- graphormer_3dpw_state_dict.bin
    |   |   |-- graphormer_hand_state_dict.bin
    |   |-- hrnet
    |   |   |-- hrnetv2_w40_imagenet_pretrained.pth
    |   |   |-- hrnetv2_w64_imagenet_pretrained.pth
    |   |   |-- cls_hrnet_w40_sgd_lr5e-2_wd1e-4_bs32_x100.yaml
    |   |   |-- cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml
    |-- src 
    |-- datasets 
    |-- predictions 
    |-- README.md 
    |-- ... 
    |-- ... 
    ```

3. Download SMPL and MANO models from their official websites

    To run our code smoothly, please visit the following websites to download SMPL and MANO models. 

    - Download `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` from [SMPLify](http://smplify.is.tue.mpg.de/), and place it at `${REPO_DIR}/src/modeling/data`.
    - Download `MANO_RIGHT.pkl` from [MANO](https://mano.is.tue.mpg.de/), and place it at `${REPO_DIR}/src/modeling/data`.

    Please put the downloaded files under the `${REPO_DIR}/src/modeling/data` directory. The data structure should follow the hierarchy below. 
    ```
    ${REPO_DIR}  
    |-- src  
    |   |-- modeling
    |   |   |-- data
    |   |   |   |-- basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
    |   |   |   |-- MANO_RIGHT.pkl
    |-- models
    |-- datasets
    |-- predictions
    |-- README.md 
    |-- ... 
    |-- ... 
    ```
    Please check [/src/modeling/data/README.md](../src/modeling/data/README.md) for further details.

4. Download prediction files that were evaluated on FreiHAND Leaderboard.

    The prediction files can be downloaded with the following command.
    ```bash
    cd $REPO_DIR
    bash scripts/download_preds.sh
    ```
    You could submit the prediction files to FreiHAND Leaderboard and reproduce our results.

5. Download datasets for training.

    We use three datasets for experiments in this project, with specific datasets and corresponding links provided below.
    1) Download the DAMON dataset
    ⚠️ Register account on the [DECO website](https://deco.is.tue.mpg.de/register.php), and then use your username and password to login to the _Downloads_ page.

    Follow the instructions on the _Downloads_ page to download the DAMON dataset. The provided metadata in the `npz` files is described as follows: 
    - `imgname`: relative path to the image file
    - `pose` : SMPL pose parameters inferred from [CLIFF](https://github.com/huawei-noah/noah-research/tree/master/CLIFF)
    - `transl` : SMPL root translation inferred from [CLIFF](https://github.com/huawei-noah/noah-research/tree/master/CLIFF)
    - `shape` : SMPL shape parameters inferred from [CLIFF](https://github.com/huawei-noah/noah-research/tree/master/CLIFF)
    - `cam_k` : camera intrinsic matrix inferred from [CLIFF](https://github.com/huawei-noah/noah-research/tree/master/CLIFF)
    - `polygon_2d_contact`: 2D contact annotation from [HOT](https://hot.is.tue.mpg.de/)
    - `contact_label`: 3D contact annotations on the SMPL mesh
    - `contact_label_smplx`: 3D contact annotation on the SMPL-X mesh
    - `contact_label_objectwise`: 3D contact annotations split into separate object labels on the SMPL mesh
    - `contact_label_smplx_objectwise`: 3D contact annotations split into separate object labels on the SMPL-X mesh
    - `scene_seg`: path to the scene segmentation map from [Mask2Former](https://github.com/facebookresearch/Mask2Former)
    - `part_seg`: path to the body part segmentation map

    The order of values is the same for all the keys. 

    <a name="convert-damon"></a>
    #### Converting DAMON contact labels to SMPL-X format (and back)

    To convert contact labels from SMPL to SMPL-X format and vice-versa, run the following command
    ```bash
    python reformat_contacts.py \
        --contact_npz datasets/Release_Datasets/damon/hot_dca_trainval.npz \
        --input_type 'smpl'
    ```

    ## Run demo on images
    The following command will run DECO on all images in the specified `--img_src`, and save rendering and colored mesh in `--out_dir`. The `--model_path` flag is used to specify the specific checkpoint being used. Additionally, the base mesh color and the color of predicted contact annotation can be specified using the `--mesh_colour` and `--annot_colour` flags respectively. 
    ```bash
    python inference.py \
        --img_src example_images \
        --out_dir demo_out
    ```

    2) Download the RICH dataset
    You can download the RICH dataset via this link: [https://rich.is.tue.mpg.de/login.php].

    3) Download the BEHAVE dataset
    You can download the BEHAVE dataset via this link: [https://virtualhumans.mpi-inf.mpg.de/behave/license.html].

    