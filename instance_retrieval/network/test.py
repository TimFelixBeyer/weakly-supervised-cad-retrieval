import argparse

import torch
from instance_retrieval.baselines import evaluate_method
from instance_retrieval.network.model import Model3d

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--path", required=True)
args = parser.parse_args()

print(f"Eval on all splits for checkpoint path {args.path}")
cad_encoder = Model3d().to(device)
cad_encoder.load_state_dict(torch.load(args.path))
scan_encoder = cad_encoder

cad_encoder.eval()
scan_encoder.eval()

def method(sample):
    vox = sample["voxels"].unsqueeze(0).to("cuda")
    # When using a siamese network this distinction doesn't matter
    if sample["type"] == "cad":
        return cad_encoder(vox).cpu().numpy()
    else:
        return scan_encoder(vox).cpu().numpy()

evaluate_method(method, "cosine")

