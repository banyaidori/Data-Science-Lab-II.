import wave

def pcm_to_wav(file_path:str, source_dir:str, target_dir:str, nchannels:int=1, sampwidth=2, framerate:int = 8000,
                        nframes:int=1, comptype:str='NONE', compname:str='NONE'):
        try:
            with open(source_dir + '/' + file_path, 'rb') as pcmfile:
                pcmdata = pcmfile.read()
            with wave.open(target_dir + '/' + file_path[2:-4]+'.wav', 'wb') as wavfile:
                wavfile.setparams((nchannels, sampwidth, framerate, nframes, comptype, compname))
                wavfile.writeframes(pcmdata)

        except Exception as ex:
            print(ex)