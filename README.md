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

Please follow [this repo](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) to download the data. Pretrained model is available at [this link](https://drive.google.com/file/d/1-BjR9CjOwZnpJQs7uW5xY5PAoKF9neC_/view?usp=sharing).

## For training
```
python train.py
```

## For testing
```
python test.py
```

## Work status
[This sheet](https://docs.google.com/spreadsheets/d/1UhpiTP2QyN1eut8PT26ld14hbsh1qCkTzvO-jR4Dr7I/edit?usp=sharing)

## Demo results
Synthesized resutls could be downloaded from [this link](https://drive.google.com/file/d/191laFzLBGxE2W3a8uJr7m8GPw-dpnfh1/view?usp=sharing).
