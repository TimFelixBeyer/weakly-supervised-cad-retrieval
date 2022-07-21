import argparse
from datetime import datetime

import tensorboardX
import torch
from instance_retrieval.network.datasets import RetrievalDataset
from instance_retrieval.network.evaluate import evaluate_retrieval
from instance_retrieval.network.losses import MinerLoss
from instance_retrieval.network.model import Model3d
from instance_retrieval.baselines import evaluate_method
from pytorch_metric_learning import losses, miners

device = "cuda" if torch.cuda.is_available() else "cpu"


parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", default=64, type=int)
parser.add_argument("--lr", default=3e-4, type=float)
parser.add_argument("--epochs", default=101, type=int)
parser.add_argument("--store-embeddings", default=False, type=bool)
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
scan_encoder = cad_encoder
opt = torch.optim.Adam(list(cad_encoder.parameters()), lr=args.lr)

loss_fn = MinerLoss(
    loss_fn=losses.TripletMarginLoss(),
    miner=miners.BatchEasyHardMiner(pos_strategy="hard", neg_strategy="hard"),
    pos_pairs="ids",
    neg_pairs="ids",
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


n_batches = 0
for epoch in range(args.epochs):
    cad_encoder.train()
    scan_encoder.train()
    for i, batch in enumerate(train_loader):
        opt.zero_grad()
        cad_voxel = batch["shapenet_voxel"].to(device)
        scan_voxel = batch["scannet_voxel"].to(device)

        x = cad_encoder(cad_voxel)
        y = scan_encoder(scan_voxel)

        loss = loss_fn(x, y, batch)
        loss.backward()
        opt.step()

        train_writer.add_scalar("batch loss", loss.item(), global_step=n_batches)
        print(f"Epoch: {epoch+1}, Batch: {i+1}", end="\r")
        n_batches += 1

    if epoch % 5 == 0:
        cad_encoder.eval()
        scan_encoder.eval()
        print("")  # newline
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

        def method(sample):
            vox = sample["voxels"].unsqueeze(0).to("cuda")
            # When using a siamese network this distinction doesn't matter
            if sample["type"] == "cad":
                return cad_encoder(vox).cpu().numpy()
            else:
                return scan_encoder(vox).cpu().numpy()

        evaluate_method(method, "cosine")
