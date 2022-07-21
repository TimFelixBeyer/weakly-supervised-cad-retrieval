from instance_retrieval.io import dump_json, load_json
import numpy as np


datapaths = load_json("./data/datapaths.json")
similarities = load_json(datapaths["scan_cad_similarity"] + "scan-cad_similarity_public_v0.json")
splits = load_json(datapaths["scan_cad_similarity"] + "scan2cad_objects_split.json")["scan2cad_objects"]
splits = dict(("_".join(k.split("_")[:-2]), v) for k, v in splits.items())


shapenet_paths = np.load("./assets/shapenet_paths.npy").tolist()
scannet_paths = np.load("./assets/scannet_paths.npy").tolist()

all_matches = {}
train_matches = {}
val_matches = {}
test_matches = {}

for scene in similarities["samples"]:
    scene_id = scene["reference"]["name"].split("/")[2].split("__")[0]
    instance_id = scene["reference"]["name"].split("/")[2].split("__")[1].split("_")[0]
    catid_cad, cad_id = scene["reference"]["name"].split("/")[2].split("_")[-2:]
    match = f"{catid_cad}/{cad_id}"

    all_matches[f"{scene_id}/{instance_id}"] = match
    split = splits["_".join(scene["reference"]["name"].split("/")[2].split("_")[:-2])]
    if split == "train":
        train_matches[f"{scene_id}/{instance_id}"] = match
    elif split == "validation":
        val_matches[f"{scene_id}/{instance_id}"] = match
    elif split == "test":
        test_matches[f"{scene_id}/{instance_id}"] = match
    else:
        print("Something is wrong!")

pred_matches = load_json("./assets/3dmatches_all_perceptual.json")

train_matches_perceptual = {m: pred_matches[m] for m in train_matches}

dump_json("./assets/3dmatches_all_ranked.json", all_matches)
dump_json("./assets/3dmatches_train_ranked.json", train_matches)
dump_json("./assets/3dmatches_train_ranked_perceptual.json", train_matches_perceptual)
dump_json("./assets/3dmatches_val_ranked.json", val_matches)
dump_json("./assets/3dmatches_test_ranked.json", test_matches)
print("All matches", len(all_matches))
print("Train matches", len(train_matches))
print("Val matches", len(val_matches))
print("Test matches", len(test_matches))
