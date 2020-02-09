# Adversarial Distributional Training

This repository contains the code for adversarial distributional training (ADT) of our submission: *Adversarial Distributional Training for Robust Deep Learning* to ICML 2020.

Our code is built upon https://github.com/yaodongyu/TRADES and .

## Prerequisites
* Python (3.6.8)
* Pytorch (1.3.0)
* torchvision (0.4.1)
* numpy

## Training

We have proposed three different methods for ADT. The command for each training method is specified below.

### Training ADT<sub>EXP</sub>

```
python adt_exp.pt --model-dir adt-exp
```

### Training ADT<sub>EXP-AM</sub>

```
python adt_expam.pt --model-dir adt-expam
```

### Training ADT<sub>IMP-AM</sub>

```
python adt_impam.pt --model-dir adt-impam
```
