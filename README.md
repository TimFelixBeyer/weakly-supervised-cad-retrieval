# Weakly-Supervised End-to-End CAD Retrieval to Scan Objects
This is the code along with pretrained models for the paper [Weakly-Supervised End-to-End CAD Retrieval to Scan Objects](https://arxiv.org/abs/2203.12873).


## Installation
1. Install the requirements from `requirements.txt`. Note that some experiments require additional (heavy) dependencies, which are noted within the respective folders.
2. Fill in the paths in `data/datapaths.json`.
3. Download the ShapeNet, ScanNet, Scan2CAD and the Scan-CAD Object Similarity Dataset and place them into the respective folders.
4. Install this package using `setup.py` from the main folder.
5. Run `python3 data/prepare_shapenet.py && python3 data/prepare_scannet.py && python3 data/prepare_sdf.py && python3 data/split_data_by_category.py && python3 data/convert_to_common_format.py` to set up all datasets.
6. Run `python3 similarity_metrics/PerceptualMetric/perceptual_metric.py && python3 similarity_metrics/maskedIoU/masked_iou_scaled.py` to populate the distance caches and evaluate the performance of the perceptual and geometric similarities. WARNING: This step may take a long time (50 hrs+) to complete. If you want to save time, run `python3 similarity_metrics/maskedIoU/masked_iou.py` instead. This will not rescale the voxel grids to fit the scan objects which results in lower performance. You can set w_percep to 1 for all further experiments to ignore the maskedIoU similarity altogether.

## Usage
Three folders are of interest: `network` (for the main experiments), `ablations` (to reproduce Table 3 from the paper), `similarity_metrics` (contains the perceptual and geometric similarity metric computations).
Generally, all scripts are designed to be run from the main folder (`instance_retrieval`).
To train the main configuration, run e.g. `python3 network/train.py --log-dir=/path/to/log`
To evaluate a pretrained model, run e.g. `python3 network/test.py --path=pretrained/unseen.pth`
It should achieve similar performance to the values in the paper.
## Miscellaneous
- In Appendix F we explain a mistake in the differentiable Top-K layer implementation of [Cordonnier et al. (2020)](https://arxiv.org/abs/2104.03059). You can find a demonstration of incorrect vs. corrected Differentiable Top-K layer behaviour at https://colab.research.google.com/drive/1GhGkqaiK1POqu3EwhjlorEDYDn-SDlcy?usp=sharing