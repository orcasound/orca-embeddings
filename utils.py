import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio
import matplotlib.pyplot as plt
import pandas as pd
import soundfile as sf

# load models from tensorflow hub
vggish_model = hub.load('https://tfhub.dev/google/vggish/1')
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
humpback_model = hub.load('https://tfhub.dev/google/humpback_whale/1')

def extract_mean_embedding(audio_file,model):
    """Extract embeddings from one of the selected models hosted on tensorflow hub"""
    # load mono file using tensorflow
    try:
      waveform, sample_rate = tf.audio.decode_wav(tf.io.read_file(audio_file),desired_channels=1)
    
    except:
      x, sample_rate = sf.read(audio_file)
      waveform = tf.Variable(x.reshape([-1,1]),dtype=tf.float32)

    if model == 'vggish':
        waveform = tf.squeeze(waveform, axis=-1)
        #resample to 16khz
        sample_rate = tf.cast(sample_rate, dtype=tf.int64)
        waveform = tfio.audio.resample(waveform, rate_in=sample_rate, rate_out=16000)
        embeddings = vggish_model(waveform)
        spectrogram,_,_,_ = plt.specgram(waveform.numpy(),Fs=int(sample_rate))
        return embeddings, spectrogram.T

    elif model == 'yamnet':
        waveform = tf.squeeze(waveform, axis=-1)
        #resample to 16khz
        sample_rate = tf.cast(sample_rate, dtype=tf.int64)
        waveform = tfio.audio.resample(waveform, rate_in=sample_rate, rate_out=16000)
        scores, embeddings, log_mel_spectrogram = yamnet_model(waveform)
        return embeddings, log_mel_spectrogram

    elif model == 'humpback':
        waveform = tf.expand_dims(waveform, 0)  # makes a batch of size 1
        pcen_spectrogram = humpback_model.front_end(waveform)
        # zero pad if lenght not a multiple of 128
        w_size = 128 # 3.84 seconds context window
        if pcen_spectrogram.shape[1] % w_size != 0:
          n = w_size - pcen_spectrogram.shape[1] % w_size
          pcen_spectrogram = tf.concat([pcen_spectrogram,tf.zeros([1,n,64])], axis=1)
        
        n_frames = int(pcen_spectrogram.shape[1]/w_size)
        
        pcen_spectrogram_r = tf.reshape(pcen_spectrogram,shape=(n_frames,w_size,64)) 
        embeddings = humpback_model.features(pcen_spectrogram_r)

        return embeddings, pcen_spectrogram
     
    else:
        "print model not specified"


