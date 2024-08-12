# Quick Demo 
We provide demo codes for end-to-end inference here. 

Our inference codes will iterate all images in a given folder, and generate the results.

## Important notes

 - **This demo doesn't perform human detection**. Our model requires a centered target in the image. 
 - As **GraphiContact is a data-driven approach**, it may not perform well if the test samples are very different from the training data. We observe that our model does not work well if the target is out-of-the-view. Some examples can be found in our supplementary material (Sec. I Limitations).

## Human Body Reconstruction 

This demo runs 3D human mesh reconstruction and contact prediction from a single image. 

Our codes require the input images that are already **cropped with the person centered** in the image. The input images should have the size of `224x224`. To run the demo, please place your test images under path to your test path , and then run the following script.

For example, if we utilize the DAMON dataset, the following code you can refer:
```bash
python GraphiContact/src/tools/deco_inference_damon.py
```

If you would like to check the contact vertices in the human cloud points, kindly run the code:
```bash
python GraphiContact/src/tools/3d_visualization.ipynb
```

After running, it will generate the results in the folder
`--scene-part_segmentation GraphiContact/src/tools/Renders/Xscene.png`
`--colored_mesh_file_or_path GraphiContact/src/tools/colored_mesh.ply`

If you want to view more detailed colorful mesh results, you can run the command locally. Below is the sample code.
```bash
import open3d as o3d

mesh = o3d.t.io.read_triangle_mesh("./colored_mesh2.ply")
mesh.compute_vertex_normals()

from  open3d.visualization import *
draw([mesh],bg_color=(255.0, 255.0, 0.5, 0.5),)
```








