import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses, miners
from sklearn.preprocessing import LabelEncoder
from instance_retrieval.io import save_img
from instance_retrieval.network.metrics import iou, perceptual_similarity
from instance_retrieval.network.model import PerturbedTopK
from torchvision import models


device = "cuda" if torch.cuda.is_available() else "cpu"


def masked_iou(batch):
    cad_vox = batch["shapenet_voxel"].to(device)
    scan_vox = batch["scannet_voxel"].to(device)
    mask = batch["scannet_sdf"].to(device) > 0.0
    return iou(cad_vox.unsqueeze(1), scan_vox.unsqueeze(0), mask)


def masked_iou_pretrain(batch):
    cad_vox = batch["shapenet_voxel_filled"].to(device)
    return iou(cad_vox.unsqueeze(1), cad_vox.unsqueeze(0))


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from
    the same class and label == 0 otherwise.
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (
            target.float() * distances
            + (1 - target).float()
            * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2)
        )
        return losses.mean() if size_average else losses.sum()


class MinerLoss(nn.Module):
    def __init__(
        self,
        loss_fn=losses.ContrastiveLoss(),
        miner=miners.BatchEasyHardMiner(),
        pos_pairs="ids",
        neg_pairs="cats",
    ):
        super(MinerLoss, self).__init__()
        self.loss_fn = loss_fn
        self.miner = miner
        self.pos_pairs = pos_pairs
        self.neg_pairs = neg_pairs

    def forward(self, cad_embeddings, scan_embeddings, batch):
        emb = torch.cat([cad_embeddings, scan_embeddings], dim=0)
        if self.pos_pairs == "ids":
            pos_labels = to_canonical(batch["cad_id"])
        elif self.pos_pairs == "cats":
            pos_labels = to_canonical(batch["catid_cad"])

        if self.neg_pairs == "ids":
            neg_labels = to_canonical(batch["cad_id"])
        elif self.neg_pairs == "cats":
            neg_labels = to_canonical(batch["catid_cad"])

        pairs = (*self.miner(emb, pos_labels)[:2], *self.miner(emb, neg_labels)[2:])
        # pos_labels are passed to the loss_fn but are ignored bc pairs are passed too
        return self.loss_fn(emb, pos_labels, pairs)


class EmbeddingIOULoss(nn.Module):
    def __init__(self, similarity_matrix=None):
        super(EmbeddingIOULoss, self).__init__()
        self.similarity_matrix = similarity_matrix
        if self.similarity_matrix is not None:
            self.similarity_matrix = np.load(similarity_matrix)

    def forward(self, cad_embeddings, scan_embeddings, batch):
        pred_sim = torch.mm(cad_embeddings, scan_embeddings.T)
        if self.similarity_matrix is None:
            cad_vox = batch["shapenet_voxel"].to(device)
            scan_vox = batch["scannet_voxel"].to(device)
            mask = batch["scannet_sdf"].to(device) > 0.0

            tgt_sim = iou(cad_vox.unsqueeze(1), scan_vox.unsqueeze(0), mask)
        else:
            tgt_sim = self.similarity_matrix[batch["scannet_idx"]][:, batch["shapenet_idx"]]
        tgt_sim = tgt_sim / tgt_sim.max(0)[0]

        pi = torch.acos(torch.zeros(1).to(device)) * 2
        tgt_sim = 0.5 + torch.asin(tgt_sim * 2 - 1) / pi
        return F.mse_loss(pred_sim, tgt_sim)


class EmbeddingLoss(nn.Module):
    def __init__(self, similarity_fn=None, similarity_matrix=None):
        super(EmbeddingLoss, self).__init__()
        assert not (similarity_fn is None and similarity_matrix is None)

        self.similarity_fn = similarity_fn
        self.similarity_matrix = similarity_matrix
        if self.similarity_matrix is not None:
            assert (self.similarity_matrix >= 0).all()
            assert (self.similarity_matrix <= 1).all()

    def forward(self, cad_embeddings, scan_embeddings, batch):
        pred_sim = torch.mm(scan_embeddings, cad_embeddings.T)  # [S_B, C_B]
        if self.similarity_fn is not None:
            tgt_sim = self.similarity_fn(batch)
        else:
            tgt_sim = self.similarity_matrix[batch["scannet_idx"]][:, batch["shapenet_idx"]]
        tgt_sim = tgt_sim / tgt_sim.max(0)[0]
        return F.mse_loss(pred_sim, tgt_sim)


class TopKLoss(nn.Module):
    def __init__(self, k, sigma, similarity_fn=None, similarity_matrix=None):
        """[summary]

        Parameters
        ----------
        k : int > 0
            K for the Top-K layer.
        sigma : float
            Standard deviation for the noise of the Top-K Layer.
        similarity_matrix : torch.tensor, shape=(N_scans, N_shapes), optional
            A 2D matrix where entry [i,j] describes the similarity of scan[i] & shape[j]
            higher means more similarity, by default None
        similarity_fn : function, optional
            Function with signature fn(batch) returning the similarity of the scans in
            the batch with the cad models in the batch, by default None
        """
        super(TopKLoss, self).__init__()
        self.k = k
        self.sigma = sigma
        assert not (similarity_fn is None and similarity_matrix is None)
        self.similarity_fn = similarity_fn
        self.similarity_matrix = similarity_matrix

    def forward(self, cad_embeddings, scan_embeddings, batch):
        # Find the predicted similarities + top-k them
        pred_sim = torch.mm(scan_embeddings, cad_embeddings.T)  # [S_B, C_B]
        k = min(self.k, pred_sim.shape[0])
        topk = PerturbedTopK(k, sigma=self.sigma)
        topk_indicators = topk(pred_sim).transpose(0, 1)  # [K, S_B, C_B]

        # Find the target similarities
        if self.similarity_fn is not None:
            tgt_sim = self.similarity_fn(batch)
        else:
            tgt_sim = self.similarity_matrix[batch["scannet_idx"]][:, batch["shapenet_idx"]]
        # Top-K actually works better without normalization!
        # tgt_sim = tgt_sim / tgt_sim.max(0)[0]

        # Compute the loss given predicted and target similarity
        topk = PerturbedTopK(k, sigma=self.sigma / 200, num_samples=1000)
        topk_tgt_indicators = topk(tgt_sim).transpose(0, 1)

        topk_tgt = (topk_tgt_indicators.detach() * tgt_sim)  # [K, S_B, C_B]
        retr_scores = topk_indicators * topk_tgt
        retr_loss = - retr_scores.sum(-1).mean((0, 1))
        return retr_loss


def to_canonical(batch_ids):
    le = LabelEncoder()
    tgt = torch.as_tensor(le.fit_transform(batch_ids))
    labels = torch.cat([tgt, tgt], dim=0).to(device)
    return labels
