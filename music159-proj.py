import soundfile as sf
from scipy import signal
import librosa
import librosa.display
import numpy as np
import sklearn
import sys
import argparse
import os
#%matplotlib inline
import matplotlib.pyplot as plt


def next_pow_of_2(x):
    if x == 0:
        return 1
    else:
        return 2**(x - 1).bit_length()

def convolution(wav1, wav2, out_wav):
    x, srx = sf.read(wav1)
    h, srh = sf.read(wav2)
    
    if srx != srh:
        sys.exit('sr must be the same in both files')
    if x.shape[0] > h.shape[0]:
        N = next_pow_of_2(x.shape[0])
    else:
        N = newt_pw_of_2(h.shape[0])
    
    scale = 0.1
    direct = 0.7
    
    y = np.fft.irfft(np.fft.rfft(x, N) * np.fft.rfft(h, N))
    y *= scale
    y[0:x.shape[0]] += x * direct
    
    sf.write(out_wav, y, srx)
    print('done with convolution')


def deconvolution(wav1, wav2, out_wav):
    y, syx = sf.read(wav1)
    x, srx = sf.read(wav2)
    
    scale = 3
    N = next_pow_of_2(y.shape[0])
    ft_y = np.fft.fft(y, N)
    ft_x = np.fft.fft(x, N)
    v = max(abs(ft_x)) * 0.005
    
    ir_rebuild = np.fft.irfft(ft_y * np.conj(ft_x) / (v + abs(ft_x)**2))
    
    sig_len = y.shape[0] - x.shape[0]
    ir_rebuild = ir_rebuild[1 : sig_len] * scale
    
    sf.write(out_wav, ir_rebuild, srx)
    print('done with deconvolution')

# when a = 0.5 & c = 1, performs ordinary convolution
# a is the wight to be applied to first file
# 1 - a is the wieght applied to second file
def weighted_geometric_mean_convolution(wav1, wav2, out_wav, a, c):
    x, srx = sf.read(wav1)
    h, srh = sf.read(wav2)
    
    size = x.shape[0]
    x = np.pad(x, (0, h.shape[0] - 1), mode='constant', constant_values=0)
    h = np.pad(h, (0, size - 1), mode='constant', constant_values=0)
   
    if srx != srh:
        sys.exit('sr must be the same in both files')
    if x.shape[0] > h.shape[0]:
        N = next_pow_of_2(x.shape[0])
    if a >= 1 or a <= 0:
        sys.exit('a must be less than 1 and greater than 0')
    if c <= 0:
        sys.exit('c must be greater than 0')
    else:
        N = next_pow_of_2(h.shape[0])
    
    scale = 0.1
    direct = 0.7
    b = 1 - a
    
    f_x = np.fft.rfft(x, N)**a
    f_h = np.fft.rfft(h, N)**b
    y = np.fft.irfft((f_x * f_h)**(2 * c))
    y *= scale
    y[0:x.shape[0]] += x * direct
    
    sf.write(out_wav, y, srx)
    print('done with convolution')

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])


def feature_maps(wav, filename):
    x, srx = sf.read(wav)
    f, t, Sxx = signal.spectrogram(x, srx, nfft=4096)
    Sxx =abs(Sxx[0:1024, :])
    fig_1 = plt.figure(1)
    plt.imshow(np.log (0.0000001 + Sxx), aspect='auto', cmap='plasma', origin='lower')
    fig_1.tight_layout()
    # plt.show ()
  
    kernel_size = 15
    blur_size = 3

    gliss_down = np.eye(kernel_size)
    gliss_down = signal.convolve2d(gliss_down, np.random.random((blur_size, blur_size)))
    gliss_up = np.flipud (gliss_down)
    gliss_up = signal.convolve2d(gliss_up, np.random.random((blur_size, blur_size)))
    steady = np.zeros((kernel_size, kernel_size))
    steady[round(kernel_size/2),:] = 1
    steady = signal.convolve2d(steady, np.random.random((blur_size, blur_size)))
    impulse = np.zeros((kernel_size, kernel_size))
    impulse[:,round(kernel_size/2)] = 1
    impulse = signal.convolve2d(impulse, np.random.random((blur_size, blur_size)))
    
    fig_2 = plt.figure(2)
    plt0 = plt.subplot(2, 2, 1)
    plt0.set_title('Down')
    plt0.imshow(gliss_down)
    plt1 = plt.subplot(2, 2, 2)
    plt1.set_title('Up')
    plt1.imshow(gliss_up)
    plt2 = plt.subplot(2, 2, 3)
    plt2.set_title('Steady')
    plt2.imshow(steady)
    plt3 = plt.subplot(2, 2, 4)
    plt3.set_title('Impulse')
    plt3.imshow(impulse)
    fig_2.tight_layout()

    # threshold = .7
    glissdn_map = np.log(.0000001 + signal.convolve2d (Sxx, gliss_down))
    # glissdn_map[glissdn_map > glissdn_map.max() * threshold] = 0
    glissup_map = np.log(.0000001 + signal.convolve2d (Sxx, gliss_up))
    # glissup_map[glissup_map > glissup_map.max() * threshold] = 0
    steady_map = np.log(.0000001 + signal.convolve2d (Sxx, steady))
    # steady_map[steady_map > steady_map.max() * threshold] = 0
    impulse_map = np.log(.0000001 + signal.convolve2d (Sxx, impulse))
    # impulse_map[impulse_map > impulse_map.max() * threshold] = 0
    
    fig_3 = plt.figure(3)
    plt4 = plt.subplot(2, 2, 1)
    plt4.set_title('MAP Down')
    plt4.imshow(glissdn_map)
    plt5 = plt.subplot(2, 2, 2)
    plt5.set_title ('MAP Up')
    plt5.imshow(glissup_map)
    plt6 = plt.subplot(2, 2, 3)
    plt6.set_title('MAP Steady')
    plt6.imshow (steady_map)
    plt7 = plt.subplot(2, 2, 4)
    plt7.set_title('MAP Impulse')
    plt7.imshow(impulse_map)
    fig_3.tight_layout()
    plt.savefig(filename)
    plt.clf()


def plot_wave(x, sr, filename):
    plt.figure(figsize=(14, 5))
    librosa.display.waveplot(x, sr=sr)
    plt.savefig(filename)
    plt.clf()

def plot_spectrogram(x, sr, filename):
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(Xdb, sr=sr, x_axis = "time", y_axis="hz")
    plt.colorbar();
    plt.savefig(filename)
    plt.clf()

def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)
    
def plot_spectral_centroid(x, sr, filename):
    spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
    plt.figure(figsize=(12, 4))
    frames = range(len(spectral_centroids))
    t = librosa.frames_to_time(frames)
    librosa.display.waveplot(x, sr=sr, alpha=0.4)
    plt.plot(t, normalize(spectral_centroids), color='r');
    plt.savefig(filename)
    plt.clf()

def plot_spectral_bandwidth(x, sr, filename):
    spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr)[0]
    spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr, p=3)[0]
    spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr, p=4)[0]
    spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
    plt.figure(figsize=(15, 9))
    frames = range(len(spectral_centroids))
    t = librosa.frames_to_time(frames)
    librosa.display.waveplot(x, sr=sr, alpha=0.4)
    plt.plot(t, normalize(spectral_bandwidth_2), color='r')
    plt.plot(t, normalize(spectral_bandwidth_3), color='g')
    plt.plot(t, normalize(spectral_bandwidth_4), color='y')
    plt.legend(('p = 2', 'p = 3', 'p = 4'));
    plt.savefig(filename)
    plt.clf()

def zero_crossing_rate(x, sr):
    return sum(librosa.zero_crossings(x, pad=False))

def plot_chromagram(x, sr, hop_length, filename):
    chromagram = librosa.feature.chroma_stft(x, sr=sr, hop_length=hop_length)
    plt.figure(figsize=(15, 5))
    librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm') 
    plt.savefig(filename)  
    plt.clf() 
    
def plot_MFCCs(x, sr, filename):
    mfccs = librosa.feature.mfcc(x, sr=sr)
    plt.figure(figsize=(15, 7))
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.savefig(filename)
    plt.clf()

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('first_wav', help='first .wav file to convolve')
	parser.add_argument('second_wav', help='second .wav file to use as a filter')
	parser.add_argument('output_wav', help='what you want to name the output .wav file')
	parser.add_argument('output_wav_deconv', help='what you want to name the deconvolution output .wav file')
	parser.add_argument('sample_rate', type=int, help='sample rate in hz')
	parser.add_argument('--wgmc', action='store_true', help='set to True to apply weighted geometric mean convolution')
	parser.add_argument('--scalar_a', type=float, default=0.5, help='weight to be applied to first .wav file, 0 < a < 1')
	parser.add_argument('--scalar_c', type=float, default=1, help='0 < c <= 1, as c -> 0, convolution will be more noise-like')
	parser.add_argument('--hop_length', type=int, default=12, help='dimension of element features')

	args = parser.parse_args()
	args = vars(args)
	wav1 = args['first_wav']
	wav2 = args['second_wav']
	output_wav = args['output_wav']
	out_deconv_wav = args['output_wav_deconv']
	sr = args['sample_rate']
	a = args['scalar_a']
	c = args['scalar_c']
	hop_length = args['hop_length']
	print(args)

	if(args['wgmc']):
		weighted_geometric_mean_convolution(wav1, wav2, output_wav, a, c)
	else:
		convolution(wav1, wav2, output_wav)	

	deconvolution(wav1, wav2, out_deconv_wav)	

	x, sr = librosa.load(wav1, sr=sr)
	h, sr = librosa.load(wav2, sr=sr)	
	y, sr = librosa.load(output_wav, sr=sr)	

	os.makedirs('img_data/', exist_ok=True)
	os.makedirs('img_data/feature_maps', exist_ok=True)
	os.makedirs('img_data/waveplots', exist_ok=True)
	os.makedirs('img_data/spectrograms', exist_ok=True)
	os.makedirs('img_data/spectral_centroids', exist_ok=True)
	os.makedirs('img_data/spectral_bandwidths', exist_ok=True)
	os.makedirs('img_data/chromagrams', exist_ok=True)
	os.makedirs('img_data/mfccs', exist_ok=True)

	feature_maps(wav1, 'img_data/feature_maps/feature_maps1.png')
	feature_maps(wav2, 'img_data/feature_maps/feature_maps2.png')
	feature_maps(output_wav, 'img_data/feature_maps/feature_maps_out.png')
	print('done with feature maps')

	plot_wave(x, sr, 'img_data/waveplots/waveplot1.png')
	plot_wave(h, sr, 'img_data/waveplots/waveplot2.png')
	plot_wave(y, sr, 'img_data/waveplots/waveplot_out.png')
	print('done plotting waves')

	plot_spectrogram(x, sr, 'img_data/spectrograms/spectrogram1.png')
	plot_spectrogram(h, sr, 'img_data/spectrograms/spectrogram2.png')
	plot_spectrogram(y, sr, 'img_data/spectrograms/spectrogram_out.png')
	print('done plotting spectrograms')

	plot_spectral_centroid(x, sr, 'img_data/spectral_centroids/spect_cent1.png')
	plot_spectral_centroid(h, sr, 'img_data/spectral_centroids/spect_cent2.png')
	plot_spectral_centroid(y, sr, 'img_data/spectral_centroids/spect_cent_out.png')
	print('done plotting spectral centroids')

	plot_spectral_bandwidth(x, sr, 'img_data/spectral_bandwidths/spect_band1.png')
	plot_spectral_bandwidth(h, sr, 'img_data/spectral_bandwidths/spect_band2.png')
	plot_spectral_bandwidth(y, sr, 'img_data/spectral_bandwidths/spect_band_out.png')
	print('done plotting spectral bandwidths')

	plot_chromagram(x, sr, hop_length, 'img_data/chromagrams/chromagram1.png')
	plot_chromagram(h, sr, hop_length, 'img_data/chromagrams/chromagram2.png')
	plot_chromagram(y, sr, hop_length, 'img_data/chromagrams/chromagram_out.png')
	print('done plotting chromagrams')

	plot_MFCCs(x, sr, 'img_data/mfccs/mfcc1.png')
	plot_MFCCs(h, sr, 'img_data/mfccs/mfcc2.png')
	plot_MFCCs(y, sr, 'img_data/mfccs/mfcc_out.png')
	print('done plotting mfccs')

	print('Success!')


main()


