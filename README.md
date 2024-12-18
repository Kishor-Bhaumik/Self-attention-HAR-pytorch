# Human Activity Recognition from Wearable Sensor Data Using Self-Attention

Pytorch implementation of "Human Activity Recognition from Wearable Sensor Data Using Self-Attention",
[24th European Conference on Artificial Intelligence, ECAI 2020](https://digital.ecai2020.eu/) by Saif Mahmud, M Tanjid Hasan Tonmoy, [Kishor Kumar Bhaumik](https://kishor-bhaumik.github.io/), A K M Mahbubur Rahman, and M Ashraful Amin, Mohammad Shoyaib,
Muhammad Asif Hossain Khan and Amin Ahsan Ali

[ [arXiV](https://arxiv.org/abs/2003.09018) ] [ [IOS Press](https://ebooks.iospress.nl/publication/55031) ]


## Dataset Download

To download dataset and place it under `data` directory for model training and inference, run the
script `dataset_download.py` with following commad:

```shell
python dataset_download.py --dataset DATASET --unzip
```

Here, the name of dataset in command line argument `DATASET` of this project will be as follows:

    DATASET = < pamap2 / opp / uschad / skoda >

For example, to download `PAMAP2` dataset and unzip under `data` directory, run the following command from project root:

```shell
python dataset_download.py --dataset opp --unzip
```

## Pretrained Models

The `saved_model` directory contains pretrained models for `PAMAP2`, `Opportuninty`, `USC-HAD` and `Skoda` dataset.
These models can be used directly for inference and performance evaluation as described in the following section.

## Training and Evaluation

Python script `main.py` will be used for model training, inference and performance evaluation. The arguments for this
script are as follows:

For example, in order to train model for `75` epochs on `PAMAP2` dataset and evaluate model performance, run the
following command:

```shell
python main.py --epochs 75 --dataset opp
```

## Citation

    @inproceedings{ECAI2020HAR-SaifTanjid,
      title={Human Activity Recognition from Wearable Sensor Data Using Self-Attention},
      author={Saif Mahmud and M. T. H. Tonmoy and Kishor Kumar Bhaumik and A. M. Rahman and M. A. Amin and M. Shoyaib and Muhammad Asif Hossain Khan and A. Ali},
      booktitle = {{ECAI} 2020 - 24th European Conference on Artificial Intelligence, 29 August-8 September 2020, Santiago de Compostela, Spain},
      year={2020}
    }
