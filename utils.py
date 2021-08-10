import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio
import matplotlib.pyplot as plt
import pandas as pd
import soundfile as sf

LOCAL_LOADING = False

if LOCAL_LOADING:
    vggish_path = 'D:/orcaml/vggish_1/' # example local dir with models
    yamnet_path = 'D:/orcaml/yamnet_1/'
    humpback_path = 'D:/orcaml/humpback_whale_1/'

    vggish_model = hub.load(vggish_path)
    yamnet_model = hub.load(yamnet_path)
    humpback_model = hub.load(humpback_path)
    
else:
    # load models from tensorflow hub
    vggish_model = hub.load('https://tfhub.dev/google/vggish/1')
    yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
    humpback_model = hub.load('https://tfhub.dev/google/humpback_whale/1')

def extract_embedding_from_audio(audio_file,model):
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
        if int(sample_rate) != 16000:
            sample_rate = tf.cast(sample_rate, dtype=tf.int64)
            waveform = tfio.audio.resample(waveform, rate_in=sample_rate, rate_out=16000)
            
        embeddings = vggish_model(waveform)
        #spectrogram,_,_,_ = plt.specgram(waveform.numpy(),Fs=16000)
        return embeddings

    elif model == 'yamnet':
        waveform = tf.squeeze(waveform, axis=-1)
        #resample to 16khz
        if int(sample_rate) != 16000:
            sample_rate = tf.cast(sample_rate, dtype=tf.int64)
            waveform = tfio.audio.resample(waveform, rate_in=sample_rate, rate_out=16000)
            
        scores, embeddings, log_mel_spectrogram = yamnet_model(waveform)
        return embeddings, log_mel_spectrogram

    elif model == 'humpback':
        if int(sample_rate) != 10000:
            sample_rate = tf.cast(sample_rate, dtype=tf.int64)
            waveform = tfio.audio.resample(waveform, rate_in=sample_rate, rate_out=10000)
        
        waveform = tf.expand_dims(waveform, 0)  # makes a batch of size 1
        pcen_spectrogram = humpback_model.front_end(waveform)
        # zero pad if lenght not a multiple of 128
        w_size = 128 # 3.84 seconds context window
        if pcen_spectrogram.shape[1] % w_size != 0:
          even_context_window = w_size - pcen_spectrogram.shape[1] % w_size
          pcen_spectrogram = tf.concat([pcen_spectrogram,tf.zeros([1,even_context_window,64])], axis=1)
        
        n_frames = int(pcen_spectrogram.shape[1]/w_size)
        
        batch_pcen_spectrogram = tf.reshape(pcen_spectrogram,shape=(n_frames,w_size,64)) 
        embeddings = humpback_model.features(batch_pcen_spectrogram)

        return embeddings, pcen_spectrogram
     
    else:
        "print model not specified"
        
def extract_embedding_from_signal(waveform,model):
    "Assumes the waveform has the correct sample rate"
    
    if model == 'vggish':
        embedding = vggish_model(waveform)
    if model == 'yamnet':
        scores, embedding, log_mel_spectrogram = yamnet_model(waveform)
    if model == 'humpback':
        
        waveform = tf.Variable(waveform.reshape([-1,1]),dtype=tf.float32)
        waveform = tf.expand_dims(waveform, 0)  # makes a batch of size 1
        pcen_spectrogram = humpback_model.front_end(waveform)
        
        # zero pad if lenght not a multiple of 128
        w_size = 128 # 3.84 seconds context window
        
        if pcen_spectrogram.shape[1] % w_size != 0:
            even_n = w_size - pcen_spectrogram.shape[1] % w_size
            pcen_spectrogram = tf.concat([pcen_spectrogram,tf.zeros([1,even_n,64])], axis=1)

        n_frames = int(pcen_spectrogram.shape[1]/w_size)

        batch_pcen_spectrogram = tf.reshape(pcen_spectrogram,shape=(n_frames,w_size,64)) 
        embedding = humpback_model.features(batch_pcen_spectrogram)
    
    return embedding

def create_embedding_animation(x,fs,name='animation',frame_duration=2,time_ratio=2,model='vggish'):
    """ Create animation to the see the output of one embedding model """
    
    hop_size = int(fs/2) # half second
    hop_duration = hop_size/fs
    w_size = fs*frame_duration
    fps = int(1 / hop_duration) * time_ratio 
    steps = np.arange(0,x.size-w_size,hop_size)

    # Writer
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='Orcasound'), bitrate=1800)

    # create figure
    fig = plt.figure(figsize=(10,9))
    # subplots
    ax1 = fig.add_subplot(3,1,1)
    ax2 = fig.add_subplot(3,1,2)
    ax3 = fig.add_subplot(3,1,3)

    max_amp = np.max(np.abs(x))
    ax1.set_ylim([-max_amp,max_amp])
    ax1.set_xlim([0,frame_duration])
    ax1.set_ylabel('{} second window'.format(frame_duration))
    ax2.set_ylabel('{} second STFT'.format(frame_duration))
    
    ax2.set_yticks()

    line, = ax1.plot([], [], lw=1)

    def init():
        line.set_data([], [])
        return line,

    # animate function
    def animate(i):

        frame = x[steps[i]:steps[i]+w_size]
        t = np.arange(0,len(frame))/fs
        # plot waveform
        line.set_data(t,frame)
        line,

        # Run the model.
        embedding = extract_embedding_from_signal(frame,model)
        # calc spectrogram
        f, t, Sxx = signal.spectrogram(frame, fs)
        # plot spectrogram
        ax2.imshow(Sxx, origin='lower',aspect='auto',cmap='jet')   
        # Plot the embedding
        ax3.imshow(embedding.numpy(),origin='lower', aspect='auto', interpolation='nearest', cmap='gray_r')
        

    m = len(steps)
    ani = matplotlib.animation.FuncAnimation(fig, animate,init_func=init, frames=m,interval=int(hop_duration*1000))
    ani.save(name + '_' + model + '.mp4', writer=writer)


def create_animation_all_models(x,fs,name='animation',frame_duration=2,time_ratio=2):
    
    hop_size = int(fs/2) # half second
    hop_duration = hop_size/fs
    w_size = fs*frame_duration
    fps = int(1 / hop_duration) * time_ratio 
    steps = np.arange(0,x.size-w_size,hop_size)

    # Writer
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='Orcasound'), bitrate=1800)

    # create figure
    fig = plt.figure(figsize=(10,9))
    # subplots
    ax1 = fig.add_subplot(5,1,1)
    ax2 = fig.add_subplot(5,1,2)
    ax3 = fig.add_subplot(5,1,3)
    ax4 = fig.add_subplot(5,1,4)
    ax5 = fig.add_subplot(5,1,5)

    max_amp = np.max(np.abs(x))
    ax1.set_ylim([-max_amp,max_amp])
    ax1.set_xlim([0,frame_duration])
    ax1.set_ylabel('{} seconds window'.format(frame_duration))
    ax2.set_ylabel('{} seconds STFT'.format(frame_duration))
    ax3.set_ylabel('Vggish')
    ax4.set_ylabel('Yamnet')
    ax5.set_ylabel('Humpback whale')

    line, = ax1.plot([], [], lw=1)

    def init():
        line.set_data([], [])
        return line,

    # animate function
    def animate(i):

        frame = x[steps[i]:steps[i]+w_size]
        t = np.arange(0,len(frame))/fs
        line.set_data(t,frame)
        line,

        # Run the models.
        vggish_embedding = extract_embedding_from_signal(frame,'vggish')
        yamnet_embedding = extract_embedding_from_signal(frame,'yamnet')
        humpback_embedding = extract_embedding_from_signal(frame,'humpback')
        # calc spectrogram
        f, t, Sxx = signal.spectrogram(frame, fs)
        
        # plot spectrogram
        ax2.imshow(Sxx, origin='lower',aspect='auto',cmap='viridis')   
        
        # Plot the embedding
        ax3.imshow(vggish_embedding.numpy(),origin='lower', aspect='auto', interpolation='nearest', cmap='gray_r')
        # Plot the embedding
        ax4.imshow(yamnet_embedding.numpy(),origin='lower', aspect='auto', interpolation='nearest', cmap='gray_r')
        # Plot the embedding
        ax5.imshow(humpback_embedding.numpy(),origin='lower', aspect='auto', interpolation='nearest', cmap='gray_r')
        

    m = len(steps)
    ani = matplotlib.animation.FuncAnimation(fig, animate,init_func=init, frames=m,interval=int(hop_duration*1000/time_ratio))
    ani.save(name + '_all_models.mp4', writer=writer)
        



