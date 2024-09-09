## SynCo: Contrastive Learning with Synthetic Hard Negatives

This is a PyTorch implementation of the [SynCo paper](https://arxiv.org/abs/XXXX.XXXXX):
```
@misc{giakoumoglou2024synco,
  author  = {Nikolaos Giakoumoglou and Tania Stathaki},
  title   = {{SynCo}: Contrastive Learning with Synthetic Hard Negatives},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
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

Linear classification results on ImageNet using this repo:

<table>
<tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Model</th>
<th valign="bottom">epochs</th>
<th valign="bottom">top-1 acc.</th>
<!-- TABLE BODY -->
<tr>
<td align="left">ResNet-50</td>
<td align="center">200</td>
<td align="center">68.1 ± 0.1</td>
</tr>
<tr>
<td align="left">ResNet-50</td>
<td align="center">800</td>
<td align="center">70.6 ± 0.0</td>
</tr>
</tbody>
</table>

Here we run 3 trials (of linear classification) and report mean&plusmn;std: the 3 results of SynCo (200 epochs) are {68.1, 68.0, 68.2}, and of SynCo (800 epochs) are {70.6, 70.6, 70.6}.

### Semi-supervised Learning

To fine-tune the model end-to-end, including training a linear classifier on features/weights using a pre-trained model on an 8-GPU machine with a subset of the ImageNet training set, run:
```
python main_semisup.py \
  -a resnet50 \
  --lr-backbone 0.005 --lr-classifier 0.005 \
  --train-percent 1 --weights finetune \
  --batch-size 256 \
  --pretrained [your checkpoint path]/checkpoint_0199.pth.tar \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  [your imagenet-folder with train and val folders]
```

We sweep over the learning rate `{0.01, 0.02, 0.05, 0.1, 0.005}` and the number of epochs `{30, 60}` to select the hyperparameters achieving the best performance on our local validation set to report test performance.

```
learning_rates=(0.01 0.02 0.05 0.1 0.005)

for lr in "${learning_rates[@]}"; do
    echo "========== LR: $lr, Percentage 1% ==========="
    python main_semisup.py -a resnet50 --lr-backbone $lr --lr-classifier 0.5 --epochs 60 --train-percent 1 --weights finetune --batch-size 1024 --pretrained [your checkpoint path]/checkpoint_0799.pth.tar --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 [your imagenet-folder with train and val folders]
    
    echo "========== LR: $lr, Percentage 10% ==========="
    python main_semisup.py -a resnet50 --lr-backbone $lr --lr-classifier 0.5 --epochs 30 --train-percent 10 --weights finetune --batch-size 1024 --pretrained [your checkpoint path]/checkpoint_0799.pth.tar --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 [your imagenet-folder with train and val folders]
done
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
<td align="left"><a href="https://arxiv.org/abs/XXXX.XXXXX">SynCo</a></td>
<td align="center">200</td>
<td align="center">68.1</td>
<td align="center"><a href="https://drive.google.com/file/d/1sdc9Q5zIOdyEEL47pq9aJrCkrN6RVPOe/view?usp=drive_link">download</a></td>
</tr>
<tr>
<td align="left"><a href="https://arxiv.org/abs/XXXX.XXXXX">SynCo</a></td>
<td align="center">800</td>
<td align="center">70.6</td>
<td align="center"><a href="https://drive.google.com/file/d/1ZOoUmB6slrQxGRA9AdaCeIN3J-r6NaWI/view?usp=drive_link">download</a></td>
</tr>
</tbody>
</table>



### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.
