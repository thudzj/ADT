# Adversarial Distributional Training

This repository contains the code for adversarial distributional training (ADT) of our submission: *Adversarial Distributional Training for Robust Deep Learning* to ICML 2020.

## Prerequisites
* Python (3.6.8)
* Pytorch (1.3.0)
* torchvision (0.4.1)
* numpy

## Training

We have proposed three different methods for ADT. The command for each training method is specified below.

### Training ADT<sub>EXP</sub>

```
python adt_exp.py --model-dir adt-exp
```

### Training ADT<sub>EXP-AM</sub>

```
python adt_expam.py --model-dir adt-expam
```

### Training ADT<sub>IMP-AM</sub>

```
python adt_impam.py --model-dir adt-impam
```

## Evaluation

### Evaluation under White-box Attacks

```
python evaluate_attacks.py --model-path ${MODEL-PATH} --attack-method PGD (or FGSM/MIM/CW)
```

### Evaluation under Transfer-based Black-box Attacks

First change the `--white-box-attack` argument in `evaluate_attacks.py` to `False`. Then run
```
python evaluate_attacks.py --source-model-path ${SOURCE-MODEL-PATH} --target-model-path ${TARGET-MODEL-PATH} --attack-method PGD (or FGSM/MIM/CW)
```

### Evaluation under SPSA

```
python spsa.py --model-path ${MODEL-PATH}
```
