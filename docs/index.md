# Wav2Rec 🎸

## Overview

Wav2Rec is a library for music recommendation based on recent advances
in self-supervised neural networks.

## Installation

```shell
pip install git+git://github.com/TariqAHassan/wav2rec@main
```

Requires Python 3.7+

## How it Works

Wav2Rec is built on top of recently developed techniques for 
[self-supervised learning](https://en.wikipedia.org/wiki/Self-supervised_learning),
whereby rich representations can be learned from data without explict labels.
In particular, Wav2Rec leverages the simple siamese (or _SimSam_) neural network
architecture proposed by Chen and He (2020), which is trained with the objective
of maximizing the similarity between two augmentations of the same image.

In order to adapt SimSam to work with audio, Wav2Rec introduces two modifications.
First, raw audio waveforms are converted into (mel)[spectrograms](https://en.wikipedia.org/wiki/Spectrogram),
which can be seen as a form of image. This adaption allows the use of a standard image model
_encoders_, such as ResNet50 or Vision Transformer (see [audionets.py](../wav2rec/nn/audionets.py)). 
Second, while spectrograms _can_ been seen as form of image, in actuality their statistical properties
are quite different from those found in natural images. For instance, because spectrograms have a temporal 
structure, flipping along this temporal dimension is not a coherent augmentation to perform. Thus, only
augmentations which respect the unique statistical properties of spectrograms have been used 
(see [transforms.py](../wav2rec/data/transforms.py)).

Once trained, music recommendation is simply a matter of performing nearest neighbour search
on the projections obtained from the model.

## Quick Start

### Training

The [Wav2RecNet](../wav2rec/nn/lightening.py) model, which underlies
`Wav2Rec()` (below), can be trained using any audio dataset. For an example
of training the model using the [FMA](https://github.com/mdeff/fma) dataset
see [experiments/fma/train.ipynb](../experiments/fma/train.ipynb).

### Inference

The `Wav2Rec()` class along with a `Wav2RecDataset()` dataset can 
be used to generate recommendations of similar music.

```python
from pathlib import Path
from wav2rec import Wav2Rec, Wav2RecDataset

MUSIC_PATH = Path("music")  # directory of music
MODEL_PATH = Path("checkpoints/my_trained_model.ckpt")  # trained model

my_dataset = Wav2RecDataset(MUSIC_PATH, ext="mp3").scan()

model = Wav2Rec(MODEL_PATH)
model.fit(my_dataset)
```

Once fit, we can load a piece of sample piece of audio

```python
waveform = my_dataset.load_audio(Path("my_song.mp3"))
```

and get some recommendations for similar music.

```python
metrics, paths = model.recommend(waveform, n=3)
```

Above, `metrics` is a 2D array which stores the similarity
metrics (cosine similarity by default) between `waveform`
and each recommendation. The `paths` object is also a 2D array,
but it contains the paths to the recommended music files.

**Note**: To get an intuition for the representations that will underlie 
these recommendations, check out [experiments/fma/inference.ipynb](../experiments/fma/inference.ipynb).
