# orca-embeddings
extraction pipelines and experiments with audio embeddings. The models used in this repo are [Vggish](https://tfhub.dev/google/vggish/1), [Yamnet](https://tfhub.dev/google/yamnet/1) and [Humbpack whale model](https://tfhub.dev/google/humpback_whale/1), each model requires the waveform to have an appropiate sample rate and tensor shape.

| model | sample rate | input representation | window size        | embedding size |
|-------|-------------|----------------------|--------------------|----------------|
|Vggish | 16 Khz      | Log-mel Spectrogram  | 0.96 s (no overlap)| 128            |
|Yamnet | 16 Khz      | Log-mel Spectrogram  | 0.96 s (0.48s hop) | 1024           |
|Humpack Whale| 10 Khz| Pcen Spectrogram     | 3.84 s             | 2048           |

# Usage

The extraction function receives an audio file and a string with the model name('vggish','yamnet','humpback'). Returns the embedding and time-frequency representation that the model uses as input. 

A really great tool to visualize embeddings, is the [embedding projector](https://projector.tensorflow.org/). You can run this tool either locally with tensorboard or directly uploading your files to a public accesible cloud resource. The embedding projector needs 3 files to do the complete visualization:

* .tsv file where each row is a point represented in a n dimensional space defined from the number of columns.
* .tsv file with metada for each of the points, first row is must be the header with column labels.
* .jpg or .png sprite image, that should be in the same order as the previous files.
* .json config file that contains the paths to the previous files.

This example in colab generates the files for the projector on a single audio file and loads the tensorboard extensions within the notebook to visualize the embeddings directly on the colab notebook. Sometime you have to delete the logs folder and re-run the cell of the projector.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1tAoBD-WbXa1PFPd0J598xxWlvZJxgCl4?usp=sharing)]

# Online Experiments

Using this workflow an projector visualization for each of the embedding models was generated using the train dataset of the Orcaml repo, you can see in the notebook how the files were generated althoug they're very similat to the colab example and just differ in the metadata generation. 

[Vggish embeddings](https://projector.tensorflow.org/?config=https://t3st-jos3.s3.us-east-2.amazonaws.com/projector_config.json)
[Yamnet embeddings](https://projector.tensorflow.org/?config=https://t3st-jos3.s3.us-east-2.amazonaws.com/yamnet_embeddings/yamnet_config.json)
[Humpback embeddings](https://projector.tensorflow.org/?config=https://t3st-jos3.s3.us-east-2.amazonaws.com/humpback_embeddings/humpback_config.json)





