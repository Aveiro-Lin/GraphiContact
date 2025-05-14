# Quick Demo 
We provide demo codes for end-to-end inference here. 

Our inference codes will iterate all images in a given folder, and generate the results.

## Important notes

 - **This demo doesn't perform human detection**. Our model requires a centered target in the image. 
 - As **GraphiContact is a data-driven approach**, it may not perform well if the test samples are very different from the training data. We observe that our model does not work well if the target is out-of-the-view. Some examples can be found in our supplementary material (Sec. I Limitations).

## Human Body Reconstruction 

This demo runs 3D human mesh reconstruction and contact prediction from a single image. 

Our codes require the input images that are already **cropped with the person centered** in the image. The input images should have the size of `224x224`. To run the demo, please place your test images under path to your test path , and then run the following script.

The **Scene Decoder** and **Part Decoder** visualization images are stored in the subdirectory: `GraphiContact/src/tools/Renders`.  

The **colored** and **non-contact point-colored** mesh `.ply` files can be found in the folder: `GraphiContact/src/tools/`.  

For example, if we utilize the DAMON dataset, the following code you can refer:
```bash
python GraphiContact/src/tools/graphi_inference_damon.py --image_file_or_path [IMAGE_FILE_OR_PATH] --save_root [SAVE_ROOT]
```

The `--image_file_or_path` parameter specifies the path to the input image(s), denoted as `[IMAGE_FILE_OR_PATH]`, while the `--save_root` parameter defines the root directory for saving the prediction results, denoted as `[SAVE_ROOT]`. The prediction output includes scene segmentation results and colorized meshes, which are saved under the directory `[SAVE_ROOT]/[IMAGE_NAME]`. If `[IMAGE_FILE_OR_PATH]` refers to a single image file, `[IMAGE_NAME]` corresponds to the name of that file. If `[IMAGE_FILE_OR_PATH]` is a directory, `[IMAGE_NAME]` corresponds to the names of the individual images contained within that directory.

If you want to view more detailed colorful mesh results, you can run the command locally. Below is the sample code.
```bash
import open3d as o3d

mesh = o3d.t.io.read_triangle_mesh("./colored_mesh2.ply")  #If you need a contactless predictive pose-aware mesh, you can use this file "colored_mesh.ply".
mesh.compute_vertex_normals()

from  open3d.visualization import *
draw([mesh],bg_color=(255.0, 255.0, 0.5, 0.5),)
```








