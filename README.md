## SynCo: Contrastive Learning with Synthetic Hard Negatives

This is a PyTorch implementation of the [SynCo paper](https://arxiv.org/abs/XXXX.XXXXX):
```
@Article{giakoumoglou2024synco,
  author  = {Giakoumoglou Nikolaos and Stathaki Tania},
  title   = {SynCo: Contrastive Learning with Synthetic Hard Negatives},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2024},
}
```

### Preparation

Install PyTorch and ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet).

This repo is based on [MoCo v2 code](https://github.com/facebookresearch/moco), with some adjustments from [BarlowTwins code](https://github.com/facebookresearch/barlowtwins):
```
diff main_synco.py <(curl https://raw.githubusercontent.com/facebookresearch/moco/main_moco.py)
diff main_lincls.py <(curl https://raw.githubusercontent.com/facebookresearch/moco/main_lincls.py)
diff main_lincls.py <(curl https://raw.githubusercontent.com/facebookresearch/barlowtwins/evaluate.py)
```


### Unsupervised Training

This implementation only supports **multi-gpu**, **DistributedDataParallel** training, which is faster and simpler; single-gpu or DataParallel training is not supported.

To do unsupervised pre-training of a ResNet-50 model on ImageNet in an 8-gpu machine, run:
```
python main_synco.py \
  -a resnet50 \
  --lr 0.03 \
  --batch-size 256 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --mlp --moco-t 0.2 --aug-plus --cos \
  --n-hard 1024 --n1 256 --n2 256 --n3 256 --n4 64 --n5 64 --n6 64 \
  [your imagenet-folder with train and val folders]
```

This script uses all the default hyper-parameters as described in the [MoCo v2 paper](https://arxiv.org/abs/1911.05722).

### Linear Classification

With a pre-trained model, to train a supervised linear classifier on frozen features/weights in an 8-gpu machine, run:
```
python main_lincls.py \
  -a resnet50 \
  --lr 30.0 \
  --batch-size 256 \
  --pretrained [your checkpoint path]/checkpoint_0199.pth.tar \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  [your imagenet-folder with train and val folders]
```

### Semi-supervised Learning

To train a supervised linear classifier on features/weights (either frozen or fine-tuned) using a pre-trained model on an 8-GPU machine with a subset of the ImageNet training set, run:
```
python main_lincls.py \
  -a resnet50 \
  --lr 0.005 \
  --batch-size 256 \
  --pretrained [your checkpoint path]/checkpoint_0199.pth.tar \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --train-percent 1 --weights finetune \
  [your imagenet-folder with train and val folders]
```

### Transferring to Object Detection

See [./detection](detection).

### Models

Our pre-trained ResNet-50 models can be downloaded as follows:

<table>
<tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">epochs</th>
<th valign="bottom">top-1 acc.</th>
<th valign="bottom">model</th>
<!-- TABLE BODY -->
<tr>
<td align="left"><a href="https://arxiv.org/abs/2003.04297">MoCo v2</a></td>
<td align="center">200</td>
<td align="center">67.7</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_200ep/moco_v2_200ep_pretrain.pth.tar">download</a></td>
</tr>
<tr>
<tr>
<td align="left"><a href="https://arxiv.org/abs/XXXX.XXXXX">SynCo</a></td>
<td align="center">200</td>
<td align="center">68.1</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v1_200ep/moco_v1_200ep_pretrain.pth.tar">download</a></td>
</tr>
<td align="left"><a href="https://arxiv.org/abs/2003.04297">MoCo v2</a></td>
<td align="center">800</td>
<td align="center">71.1</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar">download</a></td>
</tr>
<tr>
<td align="left"><a href="https://arxiv.org/abs/XXXX.XXXXX">SynCo</a></td>
<td align="center">800</td>
<td align="center">72.1</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_200ep/moco_v2_200ep_pretrain.pth.tar">download</a></td>
</tr>
</tbody>
</table>


### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.
