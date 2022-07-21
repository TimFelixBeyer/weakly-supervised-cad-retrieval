"""Splits a pre-computed dataset into train/val/test"""
from collections import Counter

from instance_retrieval.io import dump_json, load_json

N_TRAIN_CATS = 8  # Train on N most common categories (+ least common)
N_TEST_CATS = 5  # Evaluate on the next N categories

matches_path = "./assets/3dmatches_all.json"
matches_perceptual_path = "./assets/3dmatches_all_perceptual.json"

all = load_json(matches_path)
all_perceptual = load_json(matches_perceptual_path)

with open("./data/scannetv2_train.txt", "r") as f:
    train_scenes = f.read().splitlines()

# First find counts per category
instances = []
for match in all.values():
    cat = match.split("/")[0]
    instances.append(cat)
ordered_categories = []
for cat, count in Counter(instances).most_common():
    ordered_categories.append(cat)

train_dict = {}
val_dict = {}
test_dict = {}

for scene, match in all.items():
    s = scene.split("/")[0]
    cat = match.split("/")[0]
    if cat in ordered_categories[:N_TRAIN_CATS]:
        if s in train_scenes:  # From training categories + in Scannet train-set
            train_dict[scene] = match
        else:  # From the training categories but are in ScanNet val-set
            val_dict[scene] = match
    elif cat in ordered_categories[N_TRAIN_CATS : N_TRAIN_CATS + N_TEST_CATS]:
        test_dict[scene] = match

train_dict_perceptual = {}

for scene, match in all_perceptual.items():
    if scene in train_dict:
        train_dict_perceptual[scene] = match

#################################
# Print category statistics
datapaths = load_json("./data/datapaths.json")
taxonomy = load_json(datapaths["shapenet"] + "taxonomy.json")

synset = {}
for item in taxonomy:
    synset[item["synsetId"]] = item["name"]

dump_json("./data/taxonomy.json", {k: v.split(",")[0] for k, v in synset.items()})


def print_counts(matches, synset):
    instances = []
    for match in matches.values():
        cat = match.split("/")[0]
        instances.append(synset[cat])
    for i, (a, b) in enumerate(Counter(instances).most_common()):
        print(f"{i} {a}: {b}")


print("Categories contained in each split:")
print("All ---------------")
print_counts(all, synset)
print("Train -------------")
print_counts(train_dict, synset)
print("Val  --------")
print_counts(val_dict, synset)
print("Test ---------------")
print_counts(test_dict, synset)
print("Train Perceptual --------------")
print_counts(train_dict_perceptual, synset)


print(f"There are {len(train_dict)} train instances")
print(f"There are {len(val_dict)} validation instances")
print(f"There are {len(test_dict)} test instances")
print(f"There are {len(train_dict_perceptual)} train perceptual instances")


dump_json(matches_path.replace("all", "train"), train_dict)
dump_json(matches_path.replace("all", "val"), val_dict)
dump_json(matches_path.replace("all", "test"), test_dict)
dump_json(matches_perceptual_path.replace("all", "train"), train_dict_perceptual)
