# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


class SynCo(nn.Module):
    """
    Build a SynCo model (based on MoCo) with: a query encoder, a key encoder, and a queue
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=True,
                 n_hard=1024, n1=256, n2=256, n3=256, n4=64, n5=64, n6=64,
                 sigma=0.1, delta=0.01, eta=0.01,
                 warmup_epochs=10,
                 ):
        """
        base_encoder (nn.Module): The base encoder architecture (e.g., ResNet)
        dim (int): Feature dimension (default: 128)
        K (int): Queue size; number of negative keys (default: 65536)
        m (float): MoCo momentum for updating key encoder (default: 0.999)
        T (float): Softmax temperature (default: 0.07)
        mlp (bool): Whether to use MLP head (default: True)
        n_hard (int): Number of hard negatives to consider (default: 1024)
        n1 (int): Number of type 1 hard negatives (interpolation) (default: 256)
        n2 (int): Number of type 2 hard negatives (extrapolation) (default: 256)
        n3 (int): Number of type 3 hard negatives (mixup) (default: 256)
        n4 (int): Number of type 4 hard negatives (noise injection) (default: 64)
        n5 (int): Number of type 5 hard negatives (gradient-based) (default: 64)
        n6 (int): Number of type 6 hard negatives (adversarial) (default: 64)
        sigma (float): Noise level for type 4 hard negatives (default: 0.1)
        delta (float): Perturbation strength for type 5 hard negatives (default: 0.01)
        eta (float): Step size for type 6 hard negatives (default: 0.01)
        warmup_epochs (int): Number of warmup epochs without hard negatives (default: 10)
        """
        super(SynCo, self).__init__()

        # moco 
        self.K = K
        self.m = m
        self.T = T
        
        # synco
        self.hard_alpha = 0.5 # alpha in (0, 0.5) --> MoCHI type 2
        self.hard_beta = 1.5  # beta  in (1, 1.5)
        self.hard_gamma = 1   # gamma in (0, 1.0) --> MoCHI type 1
        self.sigma = sigma    # noise inject
        self.delta = delta    # perturbed
        self.eta = eta        # adversarial
        self.warmup_epochs = warmup_epochs
        
        self.n_hard = n_hard
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.n5 = n5
        self.n4 = n4
        self.n6 = n6
        
        self.use_type1 = n1 > 0
        self.use_type2 = n2 > 0
        self.use_type3 = n3 > 0
        self.use_type5 = n4 > 0
        self.use_type4 = n5 > 0
        self.use_type6 = n6 > 0
        
        assert n_hard >= max(n1, n2, n3, n5, n4, n6)

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)
        
        if mlp == True:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)
            
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False     # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(K, dim))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr : ptr + batch_size] = keys
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]
    
    def find_hard_negatives(self, logits):
        """ 
        Find top-N hard negatives from queue
        """
        _, idxs_hard = torch.topk(logits.clone().detach(), k=self.n_hard, dim=-1, sorted=False)
        return idxs_hard
    
    def hard_negatives_interpolation(self, q, idxs_hard):
        """
        Type 1 hard negatives: interpolated
        """
        batch_size, device = q.shape[0], q.device
        idxs = torch.randint(0, self.n_hard, size=(batch_size, self.n1), device=device)
        alpha = torch.rand(size=(batch_size, self.n1, 1), device=device) * self.hard_alpha 
        hard_negatives = self.queue[torch.gather(idxs_hard, dim=1, index=idxs)].clone().detach()
        hard_negatives = alpha * q.clone().detach()[:, None] + (1 - alpha) * hard_negatives
        return nn.functional.normalize(hard_negatives, dim=-1).detach()

    def hard_negatives_extrapolation(self, q, idxs_hard):
        """
        Type 2 hard negatives: extrapolated
        """
        batch_size, device = q.shape[0], q.device
        idxs = torch.randint(0, self.n_hard, size=(batch_size, self.n2), device=device)
        beta = 1 + torch.rand(size=(batch_size, self.n2, 1), device=device) * (self.hard_beta - 1)
        hard_negatives = self.queue[torch.gather(idxs_hard, dim=1, index=idxs)].clone().detach()
        hard_negatives = q.clone().detach()[:, None] + beta * (hard_negatives - q.clone().detach()[:, None])
        return nn.functional.normalize(hard_negatives, dim=-1).detach()
    
    def hard_negatives_mixup(self, q, idxs_hard):
        """
        Type 3 hard negatives: mixup
        """
        batch_size, device = q.shape[0], q.device
        batch_size, device = q.shape[0], q.device
        idxs1, idxs2 = torch.randint(0, self.n_hard, size=(2, batch_size, self.n3), device=device)
        gamma = torch.rand(size=(batch_size, self.n3, 1), device=device) * self.hard_gamma
        hard_negatives1 = self.queue[torch.gather(idxs_hard, dim=1, index=idxs1)].clone().detach()
        hard_negatives2 = self.queue[torch.gather(idxs_hard, dim=1, index=idxs2)].clone().detach()
        neg_hard = gamma * hard_negatives1 + (1 - gamma) * hard_negatives2
        return nn.functional.normalize(neg_hard, dim=-1).detach()
    
    def hard_negatives_noise_inject(self, q, idxs_hard):
        """
        Type 4 hard negatives: noise injected
        """
        batch_size, device = q.shape[0], q.device
        idxs = torch.randint(0, self.n_hard, size=(batch_size, self.n4), device=device)
        hard_negatives = self.queue[torch.gather(idxs_hard, dim=1, index=idxs)].clone().detach()
        noise = torch.randn_like(hard_negatives) * self.sigma
        return nn.functional.normalize(hard_negatives + noise, dim=-1).detach()
        
    def hard_negatives_grad(self, q, idxs_hard, epsilon=1e-5):
        """
        Type 5 hard negatives: perturbed
        """
        batch_size, device = q.shape[0], q.device
        idxs = torch.randint(0, self.n_hard, size=(batch_size, self.n5), device=device)
        hard_negatives = self.queue[torch.gather(idxs_hard, dim=1, index=idxs)].detach().clone()
        hard_negatives_list = []
        for i in range(hard_negatives.size(1)):
            neighbor = hard_negatives[:, i, :].detach().clone()
            perturbation = torch.randn_like(neighbor) * epsilon
            perturbed_neighbor_plus = neighbor + perturbation
            perturbed_neighbor_minus = neighbor - perturbation
            similarity_plus = torch.einsum('nc,nc->n', [q, perturbed_neighbor_plus])
            similarity_minus = torch.einsum('nc,nc->n', [q, perturbed_neighbor_minus])
            approx_gradient = (similarity_plus - similarity_minus) / (2 * epsilon)
            approx_gradient = approx_gradient.unsqueeze(-1)
            perturbed_neighbor = neighbor + self.delta * approx_gradient * perturbation
            hard_negatives_list.append(perturbed_neighbor.detach())
        hard_negatives_final = torch.stack(hard_negatives_list, dim=1)
        return nn.functional.normalize(hard_negatives_final, dim=-1).detach()

    def hard_negatives_adversarial(self, q, idxs_hard, epsilon=1e-5):
        """
        Type 6 hard negatives: adversarial
        """
        batch_size, device = q.shape[0], q.device
        idxs = torch.randint(0, self.n_hard, size=(batch_size, self.n6), device=device)
        hard_negatives = self.queue[torch.gather(idxs_hard, dim=1, index=idxs)].clone().detach()
        hard_negatives_list = []
        for i in range(hard_negatives.size(1)):
            neighbor = hard_negatives[:, i, :].detach().clone()
            perturbation = torch.randn_like(neighbor) * epsilon
            perturbed_neighbor_plus = neighbor + perturbation
            perturbed_neighbor_minus = neighbor - perturbation
            similarity_plus = torch.einsum('nc,nc->n', [q, perturbed_neighbor_plus])
            similarity_minus = torch.einsum('nc,nc->n', [q, perturbed_neighbor_minus])
            approx_gradient = (similarity_plus - similarity_minus) / (2 * epsilon)
            approx_gradient = approx_gradient.unsqueeze(-1)
            perturbed_neighbor = neighbor + self.eta * approx_gradient.sign() * perturbation
            hard_negatives_list.append(perturbed_neighbor.detach())
        hard_negatives_final = torch.stack(hard_negatives_list, dim=1)
        return nn.functional.normalize(hard_negatives_final, dim=-1).detach()

    def forward(self, im_q, im_k, epoch=None):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
            epoch: current epoch iteration
        Output:
            logits, targets
        """
        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.T.clone().detach()])
        
        
        if epoch is None or epoch >= self.warmup_epochs:
            # N-hardest negatives
            idxs_hard = self.find_hard_negatives(l_neg)
            
            # append negative logits with harder negatives
            if self.use_type1:
                h1 = self.hard_negatives_interpolation(q, idxs_hard)
                l_neg_1 = torch.einsum("nc,nkc->nk", [q, h1])
                l_neg = torch.cat([l_neg, l_neg_1], dim=1)

            if self.use_type2:
                h2 = self.hard_negatives_extrapolation(q, idxs_hard)
                l_neg_2 = torch.einsum("nc,nkc->nk", [q, h2])
                l_neg = torch.cat([l_neg, l_neg_2], dim=1)
                
            if self.use_type3:
                h3 = self.hard_negatives_mixup(q, idxs_hard)
                l_neg_3 = torch.einsum("nc,nkc->nk", [q, h3])
                l_neg = torch.cat([l_neg, l_neg_3], dim=1)

            if self.use_type4:
                h4 = self.hard_negatives_noise_inject(q, idxs_hard)
                l_neg_4 = torch.einsum("nc,nkc->nk", [q, h4])
                l_neg = torch.cat([l_neg, l_neg_4], dim=1)
                
            if self.use_type5:
                h5 = self.hard_negatives_grad(q, idxs_hard)
                l_neg_5 = torch.einsum("nc,nkc->nk", [q, h5])
                l_neg = torch.cat([l_neg, l_neg_5], dim=1)

            if self.use_type6:
                h6 = self.hard_negatives_adversarial(q, idxs_hard)
                l_neg_6 = torch.einsum("nc,nkc->nk", [q, h6])
                l_neg = torch.cat([l_neg, l_neg_6], dim=1)

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
