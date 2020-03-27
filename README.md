# music159-proj

Hello and welcome to my audio synthesizer and audio visualization tool! This is a python3 file that is interactive on the 
command line, which allows you to synthesize new .wav files with various audio processing algorithms such as convolution,
extended convolution cross-synthesis, deconvolution, heterodyning, and spectral freezing. You can perform any and all of these
processes in any order to generate new and unique sounds! Once you are happy with what you've created, you can generate 
various audio visualizations including feature maps, wave plots, spectrograms, spectral centroids, spectral bandwidths, 
chromagrams, and plot Mel-Frequencey Ceptral Coefficients (MFCCs) to compare and contrast different audio effects. 

Some notes on usage, run the .py file using python3 with a -h flag to see the optional arguments to pass in. Once you are
running the file, intructions for how to begin will appear on the command line. It is best to have at least two .wav files in
your directory to use to start sampling. To ensure decent runtime, your .wav files should not excede two minutes long. 
When you generate the audio visualizations, a directory named img_data will appear with subdirectories containing the .png 
files of the various visualizations. 

I implemented convolution and deconvolution very similarly to how it was shown in class. Extended convolution with 
cross-synthesis was inspired by this https://escholarship.org/uc/item/3rq0g07d article by applying weights to the frequencies
with weighted geometric means of the magnitude of the spectra and phase spectra. By changing the scalars used as weights, you
can bring out certain attributes of the sounds during convolution. I implemented heterodyning by taking the fast fourier
transform of the two signals and generating new signals by adding then and subtracting them and then taking the inverse of the 
fourier transforms to convert from the frequencey to the time domain. Spectral freezing was implemented using a python library
known as librosa and slows down the rate of the signal, "freezing" these sounds in time. 
