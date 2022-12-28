# Spikformer: When Spiking Neural Network Meets Transformer [arxiv](https://arxiv.org/abs/2209.15425)

spiking transformer

The code is being sorted and will be released as soon as possible.

Thanks for your attention :)


### Requirements
cupy==10.3.1

pytorch==1.10.0+cu111

spikingjelly==0.0.0.0.12

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


