# Pytorch Implementation of Robust Motion In-betweening

This is the unofficial implementation of the approach described in the paper:
> Felix G. Harvey, Mike Yurick, Derek Nowrouzezahrai, and Christopher Pal [Robust Motion In-betweening](https://static-wordpress.akamaized.net/montreal.ubisoft.com/wp-content/uploads/2020/07/09155337/RobustMotionInbetweening.pdf). In *ACM Transactions on Graphics (TOG)*, 2020.

We provide the code for reproducing the main results, as well as pre-trained models.

## Dependencies
- Python 3+ distribution
- PyTorch >= 1.4.0
- NumPy 
- PIL
- TensorboardX
- Pyyaml

Please follow [this repo](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) to download the data. Pretrained model is available at [this link](https://drive.google.com/file/d/1_eqiIJA9NFrHfDGnOoo5s0BdKLhgBz1U/view?usp=sharing).

## For training
```
python train.py
```

## For testing
```
python test.py
```

## The contribution of foot sliding loss
The original papaer, foot sliding problem is only post processed. Here I add the [foot sliding loss](https://github.com/xjwxjw/Pytorch-Robust-Motion-In-betweening/blob/386df7490ca2dfe89122952dd75b84506eedf700/train.py#L299) which turned to be effective to further enhance visual quality. Here is an exmaple:
<img src="https://drive.google.com/file/d/1UsIvPuJtuGKvScHE7QTNuDvFnQ79hMjP/view?usp=sharing" align="center"> <br>

## Work status
[This sheet](https://docs.google.com/spreadsheets/d/1UhpiTP2QyN1eut8PT26ld14hbsh1qCkTzvO-jR4Dr7I/edit?usp=sharing)

## Demo results
Synthesized resutls without foot sliding constraint could be downloaded from [this link](https://drive.google.com/file/d/1137bH0L-_Ri1cpJOjMi9oTUYC0x8v4fG/view?usp=sharing), the results with foot sliding constraint could be downloaded from [this link](https://drive.google.com/file/d/1FIoNyx-_SseJNWrzRjmDqeEpblyYrh3y/view?usp=sharing).
