clear all
close all
fclose all
clc

%% stationary single frequency wave generation
nData = 150;            % the number of wave data
SNR = 99;               % signal to noise ratio [dB]

% f0 = 1000;            % for mono frequency [Hz]  : between 0 ~ FS/2
f0 = [1000, 2000];      % for multi frequency [Hz]  : between 0 ~ FS/2

nfreq = length(f0);
wavSec = 0.5;           % total wave duration [s]   : between 0.5 ~ 1
sigSec = 0.5;           % signal wave duration [s]

% Default setting
FS = 16000;             % sampling rate [Hz]
wavLen = FS * wavSec;   % total wave length
sigLen = FS * sigSec;   % signal wave length
cover = blackmanharris(sigLen);

nn = 1:1:sigLen;
FrmLeng = 512;
FrmOver = FrmLeng*3/4;
window = hamming(FrmLeng);
nFFT = FrmLeng;

if nfreq > 1
    str_f0 = [];
    for iter = 1:nfreq
        str_f0 = strcat(str_f0, sprintf('%0.1fkHz',f0(iter)/1000));
        if iter < nfreq
            str_f0 = strcat(str_f0, '_');
        end
    end
    saveDir = sprintf('./multi/%02ddB/%s',SNR,str_f0);
else
    saveDir = sprintf('./mono/%02ddB/%0.1fkHz',SNR,f0/1000);
end
if ~exist(saveDir, 'dir')
    mkdir(saveDir);
end

for dter = 1:nData
    % make a signal depending on signal frequency, f0
    signal = zeros(nfreq, sigLen);
    for fter = 1:nfreq
        rnd_f0  = f0(fter) + f0(fter) * randn(1) / 100;
        rnd_pi  = pi * randn(1);
        signal(fter,:) = cos(2*pi*rnd_f0*nn./FS + rnd_pi);
%         signal(fter,:) = cos(2*pi*rnd_f0*nn./FS);
        signal(fter,:) = signal(fter,:) .* cover';
    end
    
    % extend signal to fit wavLen
    ext_signal = signal;
            
    % make a noise
    noise   = 1 * randn([1, wavLen]);

    % combined depending on SNR
    temp_signal = zeros(1,sigLen);
    com_signal = zeros(1,wavLen);
    for fter = 1:nfreq
        temp_signal = temp_signal + signal(fter,:);
        com_signal = com_signal + ext_signal(fter,:);
    end
    temp_signal = temp_signal/nfreq;
    com_signal = com_signal/nfreq;
    
    Ps = sqrt(temp_signal * temp_signal');
    Pn = sqrt(noise * noise');
    we = (Pn/Ps)*10^(SNR/20);
    
    observation = noise + we*com_signal;
    
    % wave normalization to fit [-1 1]
    maxVal = max(abs(observation));
    observation = observation/maxVal;
    
    savePath = sprintf('%s/%04d.wav',saveDir, dter)
    audiowrite(savePath, observation, FS);
    
%     figure(1);
%     cla
%     hold on, plot(com_signal, 'r');
%     hold on, plot(observation, 'b');
%     
%     figure(2);
%     cla
%     specData = spectrogram(com_signal, window, FrmOver, nFFT, FS);
%     specData = abs(specData);
%     subplot(2,1,1), imagesc(log(specData));
%     set(gca, 'YDir', 'normal');
%     specData = spectrogram(observation, window, FrmOver, nFFT, FS);
%     specData = abs(specData);
%     subplot(2,1,2), imagesc(log(specData));
%     set(gca, 'YDir', 'normal');
end
