### IBM/IRM Comparison Tests

This repo will test the results of Ideal Binary Mask and Ideal Ratio Mask on the MUSDB18HQ test dataset with different parameters, as well as propose fragmentation for STFT instead of using the whole song. The evaluation will be done using the museval library.

Tested parameters:
1. n_fft -> The window size for each fft in stft
    tested parameters -> (64, 128, 256, 512, 1024, 2048, 4096)
2. hop_size -> The amount of samples shift for every window
    tested parameters (unit = n_fft) -> (1/16, 1/8, 1/4, 1/2, 1)
    n_overlap is n_fft - hop_size




#Fragmentation
We will also compare the result from doing stft for the whole song, compared to doing it to a small segment of the song, which will then be reconstructed. We will use 1/2 of input_size for the reconstruction hop_size (different from stft hop_size), and use the middle values for both segments...

For example, 
Let's say we have a song with 4096 samples, and we will be using 2560 input_size and output_size
We will first compute samples 1 to 2560 (I), then compute samples 1281 to 3840 (II), then 2561 to 5120 (III) (since higher than the song, the song will be padded with 0s)

Then the full output of the song will be samples 1 to 1920 from I, 1921 to 3200 from II, and 3201 to 4096 from III
This is to prevent clicks.

This may also reduce the performance of IBM, and IRM, but it will be more useful when training a neural network. If the performance loss is negligible, then this kind of fragmentation would be preferable.


After finding the best parameter values (if any), we will then test the fragmentation with more parameters, which are:
1. window_count -> the amount of windows for the STFT
    tested parameters -> (1, 3, 5, 7, 9, ... maybe more?)
    This value will affect the input_size. The input size will be n_fft + (window_count-1)*hop_size
