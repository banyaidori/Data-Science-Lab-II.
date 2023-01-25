import os, glob
from tqdm import tqdm
import librosa
import numpy as np
import pandas as pd
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import librosa.display

from pcm_to_wav import pcm_to_wav as ptw



class BeeData:
    '''Class used for extracting audio features from a folder of audio files. 
    '''

    def __init__(
            self, 
            folder_name : str,
            swarming_times : dict,
            convert_pcm_wav = False,
            pickle_file_name : str = '',
            read_from_pickle = False
        ):
        '''Initialize the class and create the dataframe.

        Parameters
        ----------
        foldername : str
            The name of the directory containing the audio files or the pickle file.
        swarming_times : dict
            A dictionary containing the swarming times
        convert_pcm_wav : bool, optional
            A flag indicating whether to convert the PCM files to WAV files, by default False
        pickle_file_name : str, optional
            The name of the saved pickle file, by default empty
        read_from_pickle : bool, optional
            A flag indicating whether to read from a pickle file instead of extracting feature, by default False
        '''
        self.swarming_times = swarming_times
        self.dir = folder_name

        if read_from_pickle:
            try: 
                self.df = pd.read_pickle(f'Output/{folder_name}/{pickle_file_name}')
            except:
                print('Can not load the pickle file. Try again with different name.')
        
        else:
            self.df = pd.DataFrame()

            if convert_pcm_wav:
                self.convert_files_from_folder_to_wav()



    def get_dataframe(
            self
        ):
        ''' Returns a copy of the dataframe with features of the class.

        Returns
        -------
        DataFrame
            Copy of the dataframe with features
        '''
        return self.df.copy()



    def convert_files_from_folder_to_wav(
            self
        ):
        '''Convert all PCM files in a specific folder to WAV format.
           The specific folder is the one, which is the the class' filename parameter.
           (can't use it in case of pickle file)
           It searches for .pcm files in the Data/PCM_data folder and writes .wav files 
           to the Data/WAV_data folder.
           Current directory should be the parent folder of Data folder.
        '''
        source_dir = 'Data/PCM_data/' + self.dir

        current_dir = os.getcwd()

        abs_source_path = os.path.abspath(os.path.join(current_dir, source_dir))
        folders = os.listdir(abs_source_path)

        abs_target_path = os.path.abspath(os.path.join(current_dir, 'Data', 'WAV_data', os.path.basename(source_dir)))

        exists = os.path.exists(abs_target_path)
        if not exists:
            os.makedirs(abs_target_path)

        idx = 0
        for folder in folders:
            if folder.startswith('.'):
                break
            os.chdir(os.path.join(abs_source_path, folder))
            for file in glob.glob("*.pcm"):
                    ptw(file_path=file,source_dir=os.path.join(abs_source_path, folder),target_dir=abs_target_path)
                    idx += 1
        print(f'Done with converting #{idx} files')
        os.chdir(current_dir)



    def feature_extraction(
            self,
            ignore_warning : bool = True
        ):
        '''Extracting various audio features with the help of librosa library.

        Parameters
        ----------
        ignore_warning : bool, optional
            Some librosa function gives small warning, by default True

        Returns
        -------
        list[list]
            This list contains the features for all the audio files in the specific folder.
        '''
        folder_path = 'Data/WAV_data/' + self.dir

        if ignore_warning:
            warnings.filterwarnings('ignore')

        features = []
        for filename in tqdm(os.listdir(folder_path), 'Feature extraction: '):

            if filename.endswith(".wav"):
                y,sr = librosa.load(os.path.join(folder_path, filename), sr=None, mono=False)

                if y is None or len(y) == 0:
                    continue

                length = len(y)/sr

                mean_stft = np.mean(librosa.feature.chroma_stft(y=y,sr=sr))
                var_stft = np.var(librosa.feature.chroma_stft(y=y,sr=sr))

                tempo = librosa.beat.tempo(y=y,sr=sr)[0]
                        
                S,phase = librosa.magphase(librosa.stft(y))
                rms = librosa.feature.rms(S=S)
                rms_mean = np.mean(rms)
                rms_var = np.var(rms)
                        
                centroid = librosa.feature.spectral_centroid(S=S)
                centroid_mean = np.mean(centroid)
                centroid_var = np.var(centroid)
                        
                bandwidth = librosa.feature.spectral_bandwidth(S=S)
                bandwidth_mean = np.mean(bandwidth)
                bandwidth_var = np.var(bandwidth)
                        
                rolloff = librosa.feature.spectral_rolloff(y=y,sr=sr,roll_percent=0.85)
                rolloff_mean = np.mean(rolloff)
                rolloff_var = np.var(rolloff)
   
                zerocrossing = librosa.feature.zero_crossing_rate(y=y)
                crossing_mean = np.mean(zerocrossing)
                crossing_var = np.var(zerocrossing)
                        
                y_harmonic = librosa.effects.harmonic(y=y)
                harmonic_mean = np.mean(y_harmonic)
                harmonic_var = np.var(y_harmonic)
                        
                contrast = librosa.feature.spectral_contrast(S=S,sr=sr, fmin=30)
                contrast_mean = np.mean(contrast)
                contrast_var = np.var(contrast)

                mfcc = librosa.feature.mfcc(y=y, sr=sr)
                mfcc_mean =np.mean(mfcc)
                mfcc_var = np.var(mfcc)

                cqt = librosa.feature.chroma_cqt(y=y, sr=sr, fmin=30)
                cqt_mean =np.mean(cqt)
                cqt_var = np.var(cqt)

                flatness = librosa.feature.spectral_flatness(S=S)
                flatness_mean =np.mean(flatness)
                flatness_var = np.var(flatness)

                tonnetz = librosa.feature.tonnetz(y=y, sr=sr, fmin=30)
                tonnetz_mean =np.mean(tonnetz)
                tonnetz_var = np.var(tonnetz)

                features.append([filename, length, mean_stft, var_stft, tempo, rms_mean, rms_var, \
                    centroid_mean, centroid_var, bandwidth_mean, bandwidth_var, rolloff_mean, rolloff_var, \
                    crossing_mean, crossing_var, harmonic_mean, harmonic_var, contrast_mean, contrast_var, \
                    mfcc_mean, mfcc_var, cqt_mean, cqt_var, flatness_mean, flatness_var, tonnetz_mean, tonnetz_var])

        return features



    def normalize_df(
            self, 
            df : pd.DataFrame
        ):
        '''Normalize the values of a DataFrame between 0 and 1.

        Parameters
        ----------
        df : DataFrame
            The DataFrame that needs to be normalized.

        Returns
        -------
        DataFrame
            The DataFrame with normalized values.
        '''
        result = df.copy()
        for feature_name in df.columns:
            if feature_name not in ['filename']:
                max_value = df[feature_name].max()
                min_value = df[feature_name].min()
                result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
        
        return result



    def time_difference(
            self,
            swarming_date : datetime, 
            audio_date : datetime
        ):
        '''Compute the time difference between a swarming time and a datetime objects in minutes.

        Parameters
        ----------
        swarming_date : datetime
            The datetime object for the swarming time
        audio_date : datetime
            The datetime object for the other audio

        Returns
        -------
        float
            The time difference in minutes.
        '''
        delta = swarming_date - audio_date
        return delta.total_seconds()/60
    


    def give_swarming_score(
            self,
            row : pd.Series, 
            swarming_times : dict, 
            penalty_score : float = 1.8
        ):
        '''Compute the swarming score for a given audio file.
            This score shows how close or far the recording time is to the closest swarming time.
            The smaller the number, the closer it is.
            It penalizes when swarming already happened.

        Parameters
        ----------
        row : Series
            The Series object representing a row of the audio dataframe
        swarming_times : dict
            The dictionary containing the swarming times
        penalty_score : float, optional
            The penalty score for negative time differences, by default 1.8

        Returns
        -------
        float
            The swarming score.
        '''
        audio_time_object = datetime.strptime(row['date'] + row['time'], '%y%m%d%H%M%S')

        _, closest_index = min(
            (abs(self.time_difference(datetime.strptime(swarming_times[i]['date'] + swarming_times[i]['time'],\
            '%y%m%d%H%M%S'), audio_time_object)), i) 
            for i in swarming_times)

        best_time_object = datetime.strptime(
                swarming_times[closest_index]['date'] + 
                swarming_times[closest_index]['time'], 
                '%y%m%d%H%M%S')
        closest_time_diff = self.time_difference(best_time_object, audio_time_object)
        
        return abs(closest_time_diff) * penalty_score if closest_time_diff < 0 else closest_time_diff



    def add_extra_features(
            self,
            df : pd.DataFrame
        ):
        '''Add extra features to the audio dataframe. 
           It extracts features from the filename and gives the swarming scores.

        Parameters
        ----------
        df : DataFrame
            The dataframe containing audio data filenames.

        Returns
        -------
        DataFrame
            The dataframe with the added features.
        '''
        result = df.copy()
        result['date'] = result.apply(lambda row : row.filename.split('-')[0], axis = 1)
        result['time'] = result.apply(lambda row : row.filename.split('-')[1], axis = 1)
        result['some_id'] = result.apply(lambda row : row.filename.split('-')[2].replace('.wav', ''), axis = 1)
        result['swarming_score'] = result.apply(lambda row : self.give_swarming_score(row, self.swarming_times), axis = 1)

        return result



    def create_df_from_features(
            self,
            features : list,
            normalize = True
        ):
        '''Create a dataframe from the extracted audio features.

        Parameters
        ----------
        features : list
             List of audio features
        normalize : bool, optional
            Whether to normalize the dataframe or not, by default True

        Returns
        -------
        DataFrame
            The created dataframe with the audio features.
        '''
        df = pd.DataFrame(features, columns=["filename", "length","mean_stft","var_stft","tempo",\
            "rms_mean", "rms_var","centroid_mean","centroid_var","bandwidth_mean","bandwidth_var",\
            "rolloff_mean", "rolloff_var", "crossing_mean","crossing_var","harmonic_mean",\
            "harmonic_var", "contrast_mean","contrast_var", "mfcc_mean","mfcc_var", "cqt_mean",\
            "cqt_var", "flatness_mean", "flatness_var", "tonnetz_mean", "tonnetz_var"])
        
        if normalize:
            df = self.normalize_df(df)

        self.df = self.add_extra_features(df)
        return self.df.copy()



    def save_df_to_pickle(
            self
        ):
        '''Save the dataframe with features in a pickle format for later use.
        '''
        now = datetime.now()
        path = os.getcwd()+'/Output/'+self.dir
        exists = os.path.exists(path)
        if not exists:
            os.makedirs(path)
        self.df.to_pickle(f'Output/{self.dir}/{now.strftime("%Y-%m-%d-%H-%M-%S")}.pkl')



    def visualize_audio(
            self, 
            file_name : str
        ):
        '''Visualize the extracted features for a file.

        Parameters
        ----------
        file_name : str
            The name of the file in the WAV_data's class directory folder
        '''
        full_name =  'Data/WAV_data/' + self.dir + '/' + file_name
        warnings.filterwarnings('ignore')
        y, sr = librosa.load(full_name)
        S,_ = librosa.magphase(librosa.stft(y))

        fig, axs = plt.subplots(6, 2, figsize=(15, 25))
        fig.tight_layout(pad=5.0)

        sound = librosa.display.waveshow(y, sr=sr, ax=axs[0,0])
        axs[0,0].set_title('The original sound')

        #stft
        stft = librosa.display.specshow(librosa.feature.chroma_stft(y=y,sr=sr), y_axis='chroma', x_axis='time', ax=axs[0, 1])
        fig.colorbar(stft)
        axs[0,1].set_title('Chromagram')

        #rms
        axs[1,0].plot(librosa.feature.rms(S=S)[0], color = 'm')
        axs[1,0].set_title('Root-mean-square (RMS) value for each frame')
        
        #Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
        frames = range(len(spectral_centroid))
        t = librosa.frames_to_time(frames)
        axs[1,1].plot(t, spectral_centroid, color='b')
        axs[1,1].set_title('Spectral centroid')

        #spectral_bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y+0.01, sr=sr)[0]
        frames2 = range(len(spectral_bandwidth))
        t2 = librosa.frames_to_time(frames2)
        axs[2,0].plot(t2, spectral_bandwidth, color='r')
        axs[2,0].set_title('Spectral bandwidth')

        #spectral_rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y+0.01, sr=sr)[0]
        frames3 = range(len(spectral_rolloff))
        t3 = librosa.frames_to_time(frames3)
        axs[2,1].plot(t3, spectral_rolloff, color='g')
        axs[2,1].set_title('Spectral rolloff')


        #zero_crossing_rate
        axs[3,0].plot(librosa.feature.zero_crossing_rate(y)[0], color = 'y')
        axs[3,0].set_title('Zero crossing rate')

        #Spectral Contrast
        contrast = librosa.display.specshow(librosa.feature.spectral_contrast(S=S, sr=sr), x_axis='time', ax = axs[5,0])
        fig.colorbar(contrast)
        axs[5,0].set_title('Spectral contrast')
        

        #mfcc
        mfcc = librosa.display.specshow(librosa.feature.mfcc(y=y, sr=sr), sr=sr, x_axis='time', ax = axs[4,0])
        fig.colorbar(mfcc)
        axs[4,0].set_title('Mel-frequency cepstral coefficients (MFCCs)')

        #Chroma_cqt
        cqt = librosa.display.specshow(librosa.feature.chroma_cqt(y=y, sr=sr),y_axis='chroma', x_axis='time', ax = axs[4,1])
        fig.colorbar(cqt)
        axs[4,1].set_title('Constant-Q chromagram')


        #spectral_flatness
        axs[3,1].plot(librosa.feature.spectral_flatness(S=S)[0])
        axs[3,1].set_title('Spectral flatness')

        #tonnetz
        tonnetz = librosa.display.specshow(librosa.feature.tonnetz(y=y, sr=sr), y_axis='tonnetz', x_axis='time', ax = axs[5,1])
        fig.colorbar(tonnetz)
        axs[5,1].set_title('Tonal centroid features - tonnetz')
        

        plt.show()

