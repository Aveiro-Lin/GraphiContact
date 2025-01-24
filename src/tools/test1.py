import torch

# Loading the Checkpoint
checkpoint = torch.load('Path/to/GraphiContact/ckpt/deco_grph_damon_scene.pt')

# Inspecting the Checkpoint
print("Keys in the checkpoint:", checkpoint.keys())

# Printing Specific Training Metrics
if 'pre' in checkpoint:
    print(f"Precision: {checkpoint['pre']}")

if 'rec' in checkpoint:
    print(f"Recall: {checkpoint['rec']}")

if 'f1' in checkpoint:
    print(f"F1 Score: {checkpoint['f1']}")

if 'fp_geo_err' in checkpoint:
    print(f"False Positive Geometric Error: {checkpoint['fp_geo_err']}")

if 'fn_geo_err' in checkpoint:
    print(f"False Negative Geometric Error: {checkpoint['fn_geo_err']}")

# Extending for Additional Metrics
