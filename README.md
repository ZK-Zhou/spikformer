# Spikformer: When Spiking Neural Network Meets Transformer [arxiv](https://arxiv.org/abs/2209.15425)

## Reference
If you find this repo useful, please consider citing:
```
@article{zhou2022spikformer,
  title={Spikformer: When Spiking Neural Network Meets Transformer},
  author={Zhou, Zhaokun and Zhu, Yuesheng and He, Chao and Wang, Yaowei and Yan, Shuicheng and Tian, Yonghong and Yuan, Li},
  journal={arXiv preprint arXiv:2209.15425},
  year={2022}
}
```
Our codes are based on the official imagenet example by PyTorch, pytorch-image-models by Ross Wightman and SpikingJelly by Wei Fang.

<p align="center">
<img src="https://github.com/ZK-Zhou/spikformer/blob/main/images/overview01.png">
</p>

### Requirements
timm==0.5.4

cupy==10.3.1

pytorch==1.10.0+cu111

spikingjelly==0.0.0.0.12

pyyaml

data prepare: ImageNet with the following folder structure, you can extract imagenet by this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4).
```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```


Setting hyper-parameters in imagenet.yml
### Training  on ImageNet
```
cd imagenet
python -m torch.distributed.launch --nproc_per_node=8 train.py
```

### Testing ImageNet Val data 
```
cd imagenet
python test.py
```


