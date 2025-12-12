# ✨GraphiContact✨


GraphiContact is a novel method designed to improve 3D human mesh reconstruction and contact point prediction from monocular RGB images. The core functionality of the model integrates pose-aware features with human-scene interaction understanding, enhancing the accuracy of both contact point detection and 3D human reconstruction. The key innovation lies in its use of **Single-Image Multi-Infer Uncertainty (SIMU) Modeling**, which simulates perceptual variations (like occlusions and lighting changes) to boost model robustness in challenging real-world scenarios. Additionally, GraphiContact incorporates a **transformer-based approach**, utilizing pre-trained models through transfer learning and a novel adaptive aggregation mechanism to integrate pose-aware features for more precise human-environment interaction modeling. This unique combination enables the system to achieve superior performance in contact prediction and human reconstruction tasks across multiple benchmark datasets.

 <img src="docs/Overview.png" width="850"> 
 <img src="docs/deco_graph.png" width="900"> 

## Installation
Check [INSTALL.md](docs/INSTALL.md) for installation instructions.
For more detailed installation information, please refer to [requirements.txt](GraphiContact/requirements.txt)


## Model Zoo and Download
Please download our pre-trained models and other relevant files that are important to run our code. 

Check [DOWNLOAD.md](docs/DOWNLOAD.md) for details. 

## Quick demo
We provide demo codes to run end-to-end inference on the test images.

Check [DEMO.md](docs/DEMO.md) for details.

## Experiments
We provide python codes for training and evaluation.

Check [EXP.md](docs/EXP.md) for details.


## License

Our research code is released under the MIT license. See [LICENSE](LICENSE) for details. 



