## SynCo: Synthetic Hard Negatives in Contrastive Learning for Better Unsupervised Visual Representations

This is a PyTorch implementation of the [SynCo paper](https://arxiv.org/abs/2410.02401):
```
@misc{giakoumoglou2024synco,
  author  = {Nikolaos Giakoumoglou and Tania Stathaki},
  title   = {SynCo: Synthetic Hard Negatives in Contrastive Learning for Better Unsupervised Visual Representations},
  journal = {arXiv preprint arXiv:2410.02401},
  year    = {2024},
}
```

### Preparation

Install PyTorch and ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet).

This repo is based on [MoCo v2](https://github.com/facebookresearch/moco) and [Barlow Twins](https://github.com/facebookresearch/barlowtwins) code:
```
diff main_synco.py <(curl https://raw.githubusercontent.com/facebookresearch/moco/main_moco.py)
diff main_lincls.py <(curl https://raw.githubusercontent.com/facebookresearch/moco/main_lincls.py)
diff main_semisup.py <(curl https://raw.githubusercontent.com/facebookresearch/barlowtwins/evaluate.py)
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

This script uses all the default hyper-parameters as described in the [MoCo v2 paper](https://arxiv.org/abs/1911.05722).

### Semi-supervised Learning

To fine-tune the model end-to-end, including training a linear classifier on features/weights using a pre-trained model on an 8-GPU machine with a subset of the ImageNet training set, run:
```
python main_semisup.py \
  -a resnet50 \
  --lr-backbone [YOUR_LR] --lr-classifier [YOUR_LR] \
  --train-percent 1 --weights finetune \
  --batch-size 256 \
  --pretrained [your checkpoint path]/checkpoint_0199.pth.tar \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
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
<td align="left"><a href="https://arxiv.org/abs/2410.02401">SynCo</a></td>
<td align="center">200</td>
<td align="center">68.1</td>
<td align="center"><a href="https://drive.google.com/file/d/1sdc9Q5zIOdyEEL47pq9aJrCkrN6RVPOe/view?usp=drive_link">download</a></td>
</tr>
<tr>
<td align="left"><a href="https://arxiv.org/abs/2410.02401">SynCo</a></td>
<td align="center">800</td>
<td align="center">70.7</td>
<td align="center"><a href="https://drive.google.com/file/d/1ZOoUmB6slrQxGRA9AdaCeIN3J-r6NaWI/view?usp=drive_link">download</a></td>
</tr>
</tbody>
</table>

### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.
