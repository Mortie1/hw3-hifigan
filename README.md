# Neural vocoder (Hi-Fi GAN) with PyTorch

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#download-model">Download model</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## Installation

Follow these steps to install the project:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env/bin/activate
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install hydra (due to broken dependencies in fairseq you need to install hydra separatly. There will be error, its ok):

   ```bash
   pip install hydra-core==1.3.2
   ```

## Download model

To download my pretrained model, use commands below:

1. Install `gdown`

```bash
pip install gdown
```

2. Download model
```bash
gdown https://drive.google.com/uc?id=1ZasGiyOPqx_LJPvdL0eNxIjhC8NB2s0r
```

3. Do not forget to pass path to your model in configs (`src/configs/synthesize.yaml` and `src/configs/resynthesize.yaml` - change `from_pretrained` value)

## How To Use

#### Training

To train a model, run the following command:

```bash
python3 train.py
```

This command will run training with `hw3-hifigan/src/configs/hifigan.yaml` config.
You might want to change some configs, for example, path to pretrained model (if you want to finetune), and/or datasets/dataloaders/optimizers etc.


#### Synthesizing audio

IMPORTANT: resynthesizing works with datasets of following structure (you need to pass path to NameOfTheDirectoryWithUtterances):

```bash
NameOfTheDirectoryWithUtterances
├── UtteranceID1.wav
├── UtteranceID2.wav
.
.
.
└── UtteranceIDn.wav
```

If you want to resynthesize audio from its spectrogram, use:
```bash
python3 synthesize.py -cn=resynthesize 'datasets.test.audio_dir=<YOUR-PATH-TO-DIR-WITH-ORIGINAL-WAVS>' 'inferencer.save_path=<YOUR-OUTPUT-NAME>'
```

You can find your outputs in `/data/saved/<YOUR-OUTPUT-NAME>` folder.

IMPORTANT: synthesize command works with datasets of following structure (you need to pass path to NameOfTheDirectoryWithUtterances):

```bash
NameOfTheDirectoryWithUtterances
└── transcriptions
    ├── UtteranceID1.txt
    ├── UtteranceID2.txt
    .
    .
    .
    └── UtteranceIDn.txt
```

If you want to synthesize wavs for texts that are located in .txt files, use the following command:
```bash
python3 synthesize.py -cn=synthesize 'datasets.test.data_dir=<YOUR-PATH-TO-DIR-WITH-TXT-FILES>' 'inferencer.save_path=<YOUR-OUTPUT-PATH>'
```

If you want to pass text from cli instead of dataset, use the following command:
```bash
python3 synthesize.py -cn=synthesize 'text="Your text that you want to synthesize here."' 'inferencer.save_path=<YOUR-OUTPUT-PATH>'
```

You can find your outputs in `/data/saved/<YOUR-OUTPUT-NAME>` folder.

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
