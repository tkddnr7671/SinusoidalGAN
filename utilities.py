import os, re
import numpy as np
import scipy.signal as sp
import wave, struct
from scipy.io import wavfile
import matplotlib.pyplot as plt

def WaveRead(dirPath):
    wave_list = []
    for (path, dir, files) in os.walk(dirPath):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.wav':
                wave_list.append(path + '/' + filename)
    wave_list.sort()

    nData = len(wave_list)
    output = []
    for dter in range(nData):
        fs, wavData = wavfile.read(wave_list[dter])
        output.append(wavData)

    output = np.asarray(output)
    nLength = output.shape[1]
    return output, nData, nLength

def WaveNormalization(target):
    nData, nDim = target.shape[0], target.shape[1]

    tempMean = np.mean(target, axis=1)
    tempStd  = np.std(target, axis=1, ddof=1)

    out = np.zeros(shape=[nData, nDim], dtype=float)
    for dter in range(nData):
        out[dter,:nDim] = (target[dter,:nDim] - tempMean[dter]) / tempStd[dter]

    return out

def SpecNormToImage(target):
    nData, nDim, nFrm = target.shape[0], target.shape[1], target.shape[2]

    output = []
    for dter in range(nData):
        tempMax = np.max(np.max(target[dter], axis=1))
        tempMin = np.min(np.min(target[dter], axis=1))

        temp = (target[dter] - tempMin) / (tempMax - tempMin)
        output.append(temp)

    output = np.asarray(output)

    nFre, nFrm = output.shape[1], output.shape[2]
    return np.reshape(output, [-1, nFre, nFrm, 1]), nFre, nFrm

def sequence2frame(target, frame_size, frame_over):
    nData = target.shape[0]
    target = np.concatenate([target, np.zeros(shape=[nData, 100])], axis=1)
    nDim = target.shape[1]
    frame_shift = frame_size - frame_over
    nFrame = int((nDim-frame_size)/frame_shift)

    out = np.zeros(shape=[nData, frame_size, nFrame])
    for dter in range(nData):
        for fter in range(nFrame):
            stridx = fter * frame_shift
            endidx = stridx + (frame_size-1)
            out[dter, :(frame_size-1), fter] = target[dter, stridx:endidx]
    return out, nFrame

def frame2sequence(frames, frame_size, frame_over):
    hwindow = np.hamming(frame_size)
    nData, nFrm = frames.shape[0], frames.shape[1]
    frame_shift = frame_size - frame_over
    nLength = (nFrm-1)*frame_shift + frame_size
    output = np.zeros(shape=[nData, nLength], dtype=float)
    for dter in range(nData):
        for fter in range(nFrm):
            stridx = fter*frame_shift
            endidx = stridx + frame_size
            temp_frame = np.multiply(frames[dter, fter, :frame_size], hwindow)
            output[dter, stridx:endidx] = output[dter, stridx:endidx] + temp_frame

    return output

def WriteWave(savePath, nchannels, sampwidth, FS, value, maxValue):
    wav_fp = wave.open(savePath, 'w')
    wav_fp.setnchannels(nchannels)
    wav_fp.setsampwidth(sampwidth)
    wav_fp.setframerate(FS)
    for j in range(value.size):
        sample = int(maxValue * value[j])
        data = struct.pack('<h', sample)
        wav_fp.writeframesraw(data)
    wav_fp.close()


def wav2spec(wavedata, FS, frmLeng, frmOver):
    nData, nLength = wavedata.shape[0], wavedata.shape[1]

    win = sp.get_window('hamming', frmLeng)
    specdata = []
    for dter in range(nData):
        f, t, tempspec = sp.spectrogram(x=wavedata[dter], fs=FS, window=win, nperseg=frmLeng, noverlap=frmOver)
        specdata.append(tempspec)
    specdata = np.asarray(specdata)

    nFre, nFrm = specdata.shape[1], specdata.shape[2]
    return np.reshape(specdata, [-1, nFre, nFrm, 1]), nFre, nFrm
