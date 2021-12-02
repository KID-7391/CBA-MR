# When False Positive is Intolerant

An implement of the NeurIPS 2021 paper: [**When False Positive is Intolerant: End-to-End Optimization with Low FPR for Multipartite Ranking**](https://papers.nips.cc/paper/2021/file/28267ab848bcf807b2ed53c3a8f8fc8a-Paper.pdf)

## Environments
* **Ubuntu** 16.04
* **CUDA** 11.1
* **Python** 3.8.0
* **Pytorch** 1.7.0

See `requirement.txt`.

## Data preparation
Download datasets from [Google drive](https://drive.google.com/file/d/1Ec9QoiGSp70RjPaDVptcOkvyne4VsdOg/view?usp=sharing) and unzip it to ./data.

## Training
1. Modify configs in `scripts/[dataset]/train_cba.sh`
2. Run the script:
```shell
sh `scripts/[dataset]/train_cba.sh`
```

The model and log are saved in `output/[dataset]/logit_cba` by default.

## Evaluation
1. Download the pretrained model from [Google drive](https://drive.google.com/file/d/1ERRwBDeVYm4ZxswCbgHOsfVehCF92ule/view?usp=sharing).
2. Modify configures in `scripts/[dataset]/eval_cba.yaml`: change `--checkpoint` to the path where the model is saved.
3. Run
```shell
sh `scripts/[dataset]/eval_cba.sh`
```

The results might slightly differ from the above due to the environment difference in the training process.
