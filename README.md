![](https://img.shields.io/badge/-Python-3776AB?style=flat&logo=Python&logoColor=ffffff)
![](https://img.shields.io/github/license/xiaofeng-github/IGAN.svg?logo=github)


# IGAN (Unsupervised Anomaly Detection Using Inverse Generative Adversarial Networks)
A GANs-based unsupervised anomaly detection method.

## Getting Started
### Prerequisites
- Python 3.10
- `pip install requirements.txt`

### Data Preparation

> [!Tip]
> Give an example: 'thyroid'
```
cd /.../IGAN
mkdir ./dataset/thyroid
```

> [!NOTE]
> Download [thyroid](https://odds.cs.stonybrook.edu/thyroid-disease-dataset/) and put the data file in ''./dataset/thyroid".

```
cd ./dataset
python data_preparation.py --dataset thyroid
cd ..
```

## Implementation

```
python main.py --dataset thyroid --latent_dim 4
```




