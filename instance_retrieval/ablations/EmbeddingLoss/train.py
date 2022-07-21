import argparse
from datetime import datetime

import tensorboardX
import torch
from instance_retrieval.io import load_npy_to_torch
from instance_retrieval.network.datasets import RetrievalDataset
from instance_retrieval.network.evaluate import evaluate_retrieval
from instance_retrieval.network.losses import EmbeddingLoss
from instance_retrieval.network.model import Model3d

device = "cuda" if torch.cuda.is_available() else "cpu"


parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", default=32, type=int)
parser.add_argument("--lr", default=3e-4, type=float)
parser.add_argument("--epochs", default=101, type=int)
parser.add_argument("--store-embeddings", default=False, type=bool)
parser.add_argument("--k", default=1, type=int)
parser.add_argument("--sigma", default=0.1, type=float)
parser.add_argument("--siamese", default=True, type=bool)
parser.add_argument(
    "--log-dir", default="./logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
)
args = parser.parse_args()

train_writer = tensorboardX.SummaryWriter(args.log_dir + "/train")
val_writer = tensorboardX.SummaryWriter(args.log_dir + "/val")
test_writer = tensorboardX.SummaryWriter(args.log_dir + "/test")
train_gt_writer = tensorboardX.SummaryWriter(args.log_dir + "/train-gt")

hparams = {
    "batch_size": args.batch_size,
    "lr": args.lr,
}


train_dataset = RetrievalDataset("./assets/3dmatches_train_perceptual.json")
val_dataset = RetrievalDataset("./assets/3dmatches_val.json", return_images=True)
test_dataset = RetrievalDataset("./assets/3dmatches_test.json", return_images=True)
train_gt_dataset = RetrievalDataset("./assets/3dmatches_train.json")

train_loader = torch.utils.data.DataLoader(
    train_dataset, args.batch_size, shuffle=True, num_workers=4
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, max(args.batch_size, 128), num_workers=4
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, max(args.batch_size, 128), num_workers=4
)
train_gt_loader = torch.utils.data.DataLoader(
    train_gt_dataset, max(args.batch_size, 128), num_workers=4
)

splits = ["Train", "Val", "Test", "Train-gt"]
loaders = [train_loader, val_loader, test_loader, train_gt_loader]
writers = [train_writer, val_writer, test_writer, train_gt_writer]

cad_encoder = Model3d().to(device)
if args.siamese:
    scan_encoder = cad_encoder
    opt = torch.optim.Adam(list(cad_encoder.parameters()), lr=args.lr)
else:
    scan_encoder = Model3d().to(device)
    opt = torch.optim.Adam(
        list(cad_encoder.parameters()) + list(scan_encoder.parameters()), lr=args.lr
    )

loss_fn = EmbeddingLoss(
    similarity_matrix=(
        1 - load_npy_to_torch("assets/perceptual_distance_cache.npy").to(device)
    ),
)


print("Before training")
for split, loader, writer in zip(splits, loaders, writers):
    avg_metrics, metrics = evaluate_retrieval(
        loader,
        writer,
        loss_fn,
        cad_encoder,
        scan_encoder,
        step=0,
        store_embeddings=args.store_embeddings,
    )
    print(f"{split}: {avg_metrics}")


def shift_tensor(x, shift=[0, 0, 0, 0, 0], fill=0):
    assert len(shift) == x.ndim
    out = torch.ones_like(x) * fill
    out[
        max(shift[0], 0) : x.shape[0] + shift[0],
        max(shift[1], 0) : x.shape[1] + shift[1],
        max(shift[2], 0) : x.shape[2] + shift[2],
        max(shift[3], 0) : x.shape[3] + shift[3],
        max(shift[4], 0) : x.shape[4] + shift[4],
    ] = x[
        max(-shift[0], 0) : x.shape[0] - shift[0],
        max(-shift[1], 0) : x.shape[1] - shift[1],
        max(-shift[2], 0) : x.shape[2] - shift[2],
        max(-shift[3], 0) : x.shape[3] - shift[3],
        max(-shift[4], 0) : x.shape[4] - shift[4],
    ]
    return out


n_batches = 0
for epoch in range(args.epochs):
    cad_encoder.train()
    scan_encoder.train()
    loss_ = 0
    for i, batch in enumerate(train_loader):
        opt.zero_grad(set_to_none=True)
        shift = torch.randint(-2, 3, (3,))
        cad_voxel = shift_tensor(batch["shapenet_voxel"].to(device), [0, 0, *shift])
        scan_voxel = shift_tensor(batch["scannet_voxel"].to(device), [0, 0, *shift])

        x = cad_encoder(cad_voxel)
        y = scan_encoder(scan_voxel)

        loss = loss_fn(x, y, batch)
        loss.backward()
        opt.step()

        print(f"Epoch: {epoch+1}, Batch: {i+1}", end="\r")
        n_batches += 1
        loss_ += loss.detach() * cad_voxel.shape[0]

    train_writer.add_scalar(
        "epoch loss", torch.sum(loss_) / len(train_dataset), global_step=n_batches
    )

    if epoch % 5 == 0:
        cad_encoder.eval()
        scan_encoder.eval()
        print("")
        with torch.no_grad():
            for split, loader, writer in zip(splits, loaders, writers):
                avg_metrics, metrics = evaluate_retrieval(
                    loader,
                    writer,
                    loss_fn,
                    cad_encoder,
                    scan_encoder,
                    step=n_batches,
                    store_embeddings=args.store_embeddings,
                )
                print(f"{split}: {avg_metrics}")

