# Automatic Speech Recognition

Implementation of models for the Automatic Speech Recognition problem.

1. [QuartzNet](https://arxiv.org/abs/1910.10261) with (BxS)xR architecture:

<p>
    <img src="img/quartznet_arch.png">
</p>

2. [Deepspeech]():

<p>
    <img src="img/deepspeech.png">
</p>

## Notebook

[![Open In Colab](https://colab.research.google.com/github/khaykingleb/ASR/blob/master/notebooks/notebook.ipynb#scrollTo=vuUOQ7zmOurV)

## Getting Started

Clone the repository and step into it:

```shell
git clone https://github.com/khaykingleb/ASR.git
cd ASR
```

Install requirements and modules.

```shell
pip install -r requirements.txt
python setup.py install
```

Use for training:

```shell
python train.py -c configs/cofig_name.json
```

Use for testing:

```shell
python test.py \
      -c default_test_model/config.json \
      -r default_test_model/checkpoint.pth \
      -o result.json
```

Please, note that for testing the model you need to specify the dataset in `test.py`, for instance `LibrispeechDataset`:

```
config.config["data"] = {
        "test": {
            "batch_size": args.batch_size,
            "num_workers": args.jobs,
            "datasets": [
                {
                    "type": "LibrispeechDataset",
                    "args": {
                        "part": "test-clean"
                    }
                }
            ]
        }
    }
```

## Data Used

* [Mozilla Common Voice](https://commonvoice.mozilla.org/).
* [LJ Speech](https://keithito.com/LJ-Speech-Dataset/).
* [Librispeech](https://paperswithcode.com/dataset/librispeech).
