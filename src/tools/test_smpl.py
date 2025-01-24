from nosmpl.smpl_onnx import SMPLOnnxRuntime
import numpy as np
from nosmpl.vis.vis_o3d import vis_mesh_o3d, Open3DVisualizer

smpl = SMPLOnnxRuntime()

body = np.random.randn(1, 23, 3).astype(np.float32)
global_orient = np.random.randn(1, 1, 3).astype(np.float32)
outputs = smpl.forward(body, global_orient)

vertices, joints, faces = outputs
vertices = vertices[0].squeeze()
joints = joints[0].squeeze()

faces = faces.astype(np.int32)
vis_mesh_o3d(vertices, faces)