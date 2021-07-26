%clear, clc, close all
%load preictal_class.mat
% load an audio file
%[x, fs] = audioread('track.wav');   % load an audio file
tic
fd=[];

preictal_class = record1(:,1:7680)                        % get the first channel
fs=256
% define analysis parameters
wlen = 256;                        % window length (recomended to be power of 2)
hop = wlen/4;                       % hop size (recomended to be power of 2)
nfft = 64;                        % number of fft points (recomended to be power of 2)
preictal_stft=[];
% perform STFT
for i=1:1:23
    fd=[];
%for j=1:7680:742000
for j=1:7680:7680
    win = blackman(wlen, 'periodic');
        x= preictal_class(i,j:j+7679);

    [S, f, t] = stft(x, win, hop, nfft, fs);
    fd=cat(3,fd,S);
end
preictal_stft=cat(4,preictal_stft,fd);
end


% calculate the coherent amplification of the window
C = sum(win)/wlen;

% take the amplitude of fft(x) and scale it, so not to be a
% function of the length of the window and its coherent amplification
S = abs(S)/wlen/C;

% correction of the DC & Nyquist component
if rem(nfft, 2)                     % odd nfft excludes Nyquist point
    S(2:end, :) = S(2:end, :).*2;
else                                % even nfft includes Nyquist point
    S(2:end-1, :) = S(2:end-1, :).*2;
end

% convert amplitude spectrum to dB (min = -120 dB)
S = 20*log10(S + 1e-6);

% plot the spectrogram
figure(1)
surf(t, f, S)
shading interp
axis tight
view(0, 90)
set(gca, 'FontName', 'Times New Roman', 'FontSize', 14)
xlabel('Time, s')
ylabel('Frequency, Hz')
title('Amplitude spectrogram of the signal')

hcol = colorbar;
set(hcol, 'FontName', 'Times New Roman', 'FontSize', 14)
ylabel(hcol, 'Magnitude, dB')

toc