import os, sys
from scipy import io
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import matplotlib.pyplot as plt
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
random_state = 1234
base_dir = str(project_root)
# datadir: C:\Users\OEM3\pch\Dataset\[9] 2009 PHM Data Challenge (gearbox)\phm09 dataset\00_Dataset
class PHMGearboxDataProcessor:
    # def __init__(self, config):
    #     self.config = config

    def collect_all_data(base_path, f_list, data_type, dimension_type='time_freq', window_size=1024, overlap=128):
        if dimension_type == 'time_freq':
            pass
        elif dimension_type == 'time_domain':
            all_datasets, all_metadata_dfs = PHMGearboxDataProcessor.process_files_with_origin_time_domain(f_list, base_path,data_type,window_size, overlap)
        elif dimension_type == 'freq_domain':
            sampling_rate = 66670
            all_datasets, all_metadata_dfs = PHMGearboxDataProcessor.process_files_with_origin_freq_domain(f_list, base_path,data_type,window_size, overlap,sampling_rate)
       
        print("\nFinal dataset info:")
        print(f"Data shape: {all_datasets.shape}")
        print(f"Metadata shape: {all_metadata_dfs.shape}")
        
        return all_datasets, all_metadata_dfs


    def print_distribution(metadata):
        print("\nRotating condition distribution:")
        print(metadata['rotating_condition'].value_counts())
        print("\nSeverity level distribution:")
        print(metadata['rotating_condition_severity'].value_counts())
        print("\nBearing condition distribution:")
        print(metadata['bearing_condition'].value_counts())
        print("\nRPM distribution:")
        print(metadata['rpm'].value_counts())
    @staticmethod
    def create_metadata(filename):
        # 메타데이터를 저장할 딕셔너리
        meta = {}
        
        # 파일명에서 정보 추출
        parts = filename.split('_')
        meta['filename'] = filename
        meta['state'] = parts[0]
        meta['group'] = int(parts[-1].split('.')[0]) # 1,2
        meta['load'] = parts[-2] # High, Low       
        meta['rpm'] = int(parts[1].split('hz')[0]) # 30hz
        return meta
    
    @staticmethod
    def process_files_with_origin_time_domain(file_list, data_dir, data_type, window_size=1024, overlap=128):
        """
        시계열 데이터를 윈도우 크기로 자르고 오버랩을 적용하여 처리
        """
        all_segments = []
        metadata_list = []
        
        for file in file_list:
            # 폴더리스트안에 부하, 속도별로 또 폴더가 있음
            if file.find(data_type.split('_')[-1])!=-1:
                data_path_ = data_dir + file+'/'
                state_files = os.listdir(data_path_)
                for state_file in state_files:
                    data_path_path = data_path_ + state_file + '/'
                    file_ = os.listdir(data_path_path)
                    data = pd.read_csv(data_path_path + file_[0],delimiter=' ', header=None, names=['acc1','acc2','tachometer'],skipinitialspace=True) # 모든폴더엔 1개txt만 있음
                    metadata = PHMGearboxDataProcessor.create_metadata(file_[0])
                    time_series = data['acc1'] # acc1 위치만 사용
                    stride = window_size - overlap
                    n_segments = ((len(time_series) - window_size) // stride) + 1
                        
                    for i in range(n_segments):
                        start = i * stride
                        end = start + window_size
                        segment = time_series[start:end]
                        
                        # 세그먼트 저장
                        all_segments.append(segment)
                        
                        # 각 세그먼트에 대한 메타데이터 생성
                        segment_metadata = {
                            'filename': file,
                            'segment_idx': i,
                            'start_idx': start,
                            'end_idx': end,
                            'bearing_condition': metadata['state'],
                            'group': metadata['group'],
                            'load': metadata['load'],
                            'rpm': metadata['rpm']
                        }
                        metadata_list.append(segment_metadata)
        
        # 모든 세그먼트를 numpy 배열로 변환
        stacked_data = np.array(all_segments)
        
        # 메타데이터를 DataFrame으로 변환
        metadata_df = pd.DataFrame(metadata_list)
        
        print(f"Processed time domain data shape: {stacked_data.shape}")
        print(f"Metadata DataFrame shape: {metadata_df.shape}")
        
        return stacked_data, metadata_df
    @staticmethod    
    def process_files_with_origin_freq_domain(file_list, data_dir, data_type, window_size=1024, overlap=128, sampling_rate=25600):
        """
        시계열 데이터를 윈도우 크기로 자르고 오버랩을 적용하여 fft하여 출력
        """
        all_segments = []
        metadata_list = []
        
        for file in file_list:
            # 폴더리스트안에 부하, 속도별로 또 폴더가 있음
            if file.find(data_type.split('_')[-1])!=-1:
                data_path_ = data_dir + file+'/'
                state_files = os.listdir(data_path_)
                for state_file in state_files:
                    data_path_path = data_path_ + state_file + '/'
                    file_ = os.listdir(data_path_path)
                    data = pd.read_csv(data_path_path + file_[0],delimiter=' ', header=None, names=['acc1','acc2','tachometer'],skipinitialspace=True) # 모든폴더엔 1개txt만 있음
                    metadata = PHMGearboxDataProcessor.create_metadata(file_[0])
                    time_series = data['acc1'] # acc1 위치만 사용
                    stride = window_size - overlap
                    n_segments = ((len(time_series) - window_size) // stride) + 1
                        
                    for i in range(n_segments):
                        start = i * stride
                        end = start + window_size
                        segment = time_series[start:end]
                        # FFT 계산
                        segment_fft = np.fft.fft(segment)
                        segment_fft_mag = np.abs(segment_fft[:window_size//2])  # 절반만 사용 (나머지는 대칭)
                        
                        # 주파수 축 계산
                        freqs = np.fft.fftfreq(window_size, d=1/sampling_rate)[:window_size//2]
                        # 진폭 스펙트럼을 파워 스펙트럼으로 변환 (선택사항)
                        segment_power = (segment_fft_mag ** 2) / window_size
                        # 데시벨 스케일로 변환 (선택사항)
                        segment_db = 10 * np.log10(segment_power + 1e-10)  # 0 방지를 위해 작은 값 추가
                        # 저장할 데이터 선택 (magnitude, power, 또는 dB)
                        all_segments.append(segment_power)                  
                        
                        # 각 세그먼트에 대한 메타데이터 생성
                        segment_metadata = {
                            'filename': file,
                            'segment_idx': i,
                            'start_idx': start,
                            'end_idx': end,
                            'bearing_condition': metadata['state'],
                            'group': metadata['group'],
                            'load': metadata['load'],
                            'rpm': metadata['rpm']
                        }
                        metadata_list.append(segment_metadata)
            
        # 모든 세그먼트를 numpy 배열로 변환
        stacked_data = np.array(all_segments)
        
        # 메타데이터를 DataFrame으로 변환
        metadata_df = pd.DataFrame(metadata_list)
        
        print(f"Processed time domain data shape: {stacked_data.shape}")
        print(f"Metadata DataFrame shape: {metadata_df.shape}")
        
        return stacked_data, metadata_df

class PHMGearboxDataLoader:
    @staticmethod
    def load_phm_gearbox_data(stacked_data, metadata_df, load_cond, speed_cond, noise_level=0, label_column='bearing_condition'):
        if "Diff_load" in load_cond:
            X_train, X_test, y_train, y_test, label_to_idx, metadata_train, metadata_test = PHMGearboxDataLoader.split_data_by_load(stacked_data, metadata_df, load_cond, speed_cond, label_column)
        else:
            X_train, X_test, y_train, y_test, label_to_idx, metadata_train, metadata_test = PHMGearboxDataLoader.split_data_stratified(stacked_data, metadata_df, load_cond, speed_cond, label_column)
        # 노이즈 추가
        if noise_level is not None and noise_level != 0 and (type(noise_level) == float or type(noise_level) == int):
            X_train = PHMGearboxDataLoader.add_noise_snr(X_train, noise_level)
            X_test = PHMGearboxDataLoader.add_noise_snr(X_test, noise_level)
            print(f"Added {noise_level*100}% Gaussian noise to the data")
        elif noise_level != 0 and noise_level.startswith('f'):
            X_train = PHMGearboxDataLoader.add_signal_proportional_noise(X_train, noise_level)
            X_test = PHMGearboxDataLoader.add_signal_proportional_noise(X_test, noise_level)
            print(f"Added {noise_level} SNR noise to the data")
        return X_train, X_test, y_train, y_test, label_to_idx, metadata_train, metadata_test

    @staticmethod
    def add_noise_snr(data, snr_db):
        """
        데이터에 지정된 SNR(dB)로 가우시안 노이즈를 추가하는 함수
        
        Parameters:
        -----------
        data : numpy.ndarray
            원본 데이터
        snr_db : float
            목표 Signal-to-Noise Ratio (dB)
            
        Returns:
        --------
        numpy.ndarray
            노이즈가 추가된 데이터
        """
        # 각 샘플별로 노이즈 추가
        noisy_data = np.zeros_like(data)
        # 더 일반적인 SNR 값 (15-25dB) SNR(dB) = 10dB는 신호가 노이즈보다 10배 강하다는 의미입니다 (10-30)
        for i in range(len(data)):
            # 현재 샘플의 신호 파워 계산
            signal_power = np.mean(data[i] ** 2)
            # SNR 공식: SNR = 10 * log10(signal_power/noise_power)
            noise_power = signal_power / (10 ** (snr_db / 10))
            
            # 가우시안 노이즈 생성
            noise = np.random.normal(0, np.sqrt(noise_power), size=data[i].shape)
            # 노이즈 추가
            noisy_data[i] = data[i] + noise
        # PHMGearboxDataLoader.plot_original_vs_noisy(data, noisy_data, snr_db)    
        return noisy_data  

    @staticmethod
    def add_signal_proportional_noise(data, noise_level):
        """
        데이터 값에 비례하여 노이즈 강도를 조정하는 함수.

        Parameters:
        -----------
        data : numpy.ndarray
            원본 데이터.
        scaling_factor : float
            데이터 값에 곱해지는 스케일링 비율.

        Returns:
        --------
        numpy.ndarray
            노이즈가 추가된 데이터.
        """
        scaling_factor =float(noise_level[1:])
        noisy_data = np.zeros_like(data)
        
        # 각 행(샘플)별로 노이즈 추가
        for i in range(len(data)):
            # 현재 샘플의 데이터 값 기반 노이즈 강도 계산
            noise_strength = scaling_factor * np.abs(data[i])#-np.mean(data[i]))
            
            # 현재 샘플에 대한 노이즈 생성
            noise = np.random.normal(0, noise_strength)
            
            # 노이즈 추가
            noisy_data[i] = data[i] + noise
        return noisy_data 
    @staticmethod
    def plot_original_vs_noisy(X_original, X_noisy, noise_level, num_samples=8):
        """
        원본 데이터와 노이즈가 추가된 데이터를 비교하여 시각화
        
        Parameters:
        -----------
        X_original: np.array
            원본 데이터
        X_noisy: np.array
            노이즈가 추가된 데이터
        noise_level: float or str
            노이즈 레벨 (그래프 제목에 표시)
        num_samples: int
            표시할 샘플 수 (기본값: 8)
        """
        # 랜덤하게 샘플 선택
        total_samples = X_original.shape[0]
        selected_indices = np.random.choice(total_samples, num_samples, replace=False)
        
        # 그래프 설정
        fig, axes = plt.subplots(4, 2, figsize=(15, 20))
        fig.suptitle(f'Original vs Noisy Signal Comparison (Noise Level: {noise_level})', fontsize=16)
        
        for idx, ax in zip(selected_indices, axes.ravel()):
            # 원본 데이터 플롯
            ax.plot(X_original[idx], label='Original', alpha=0.7, color='blue')
            # 노이즈 데이터 플롯
            ax.plot(X_noisy[idx], label='Noisy', alpha=0.7, color='red')
            
            # 그래프 꾸미기
            ax.set_title(f'Sample {idx}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Amplitude')
        
        plt.tight_layout()
        plt.show()
        plt.savefig(f'gearbox_original_vs_noisy_{noise_level}.png')    
    def split_data_stratified(spectrograms, metadata_df, load_cond, speed_cond, test_size=0.2, random_state=random_state):
        """
        load_cond: 선택할 load 값의 리스트 (예: [0, 1])
        severity_cond: 선택할 severity 값의 리스트 (예: [0, 1]). None이면 모든 severity 포함
        """
        # Load 조건으로 필터링
        if 'High' in load_cond:
            load_mask = metadata_df['load']=='High'
            filtered_metadata = metadata_df[load_mask]
            filtered_spectrograms = spectrograms[load_mask]
        elif 'Low' in load_cond:
            load_mask = metadata_df['load']=='Low'
            filtered_metadata = metadata_df[load_mask]
            filtered_spectrograms = spectrograms[load_mask]
        else: # all
            filtered_metadata = metadata_df
            filtered_spectrograms = spectrograms
        # Severity 조건으로 필터링 (지정된 경우)
        if isinstance(speed_cond, list) and all(x in [30,35,40,45,50] for x in speed_cond):
            try:
                sensor_mask = filtered_metadata['rpm'].isin(speed_cond)
                filtered_metadata = filtered_metadata[sensor_mask]
                filtered_spectrograms = filtered_spectrograms[sensor_mask]
            except:
                print("Speed condition is not valid")

                                
        # 필터링 결과 확인
        print("\nFiltered data distribution:")
        print(f"Load conditions: {load_cond}")
        print(f"Sensor conditions: {speed_cond}")
        print(f"Remaining samples: {len(filtered_metadata)}")

        filtered_metadata['stratification_label'] = filtered_metadata['bearing_condition']
        print("\nSeverity distribution:")
        print(filtered_metadata['stratification_label'].value_counts())
        
        # Label encoding
        unique_labels = sorted(filtered_metadata['stratification_label'].unique())
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}        # 데이터 분할
        
        # 그룹으로 train/test 분할
        X_train_idx = filtered_metadata['group'] == 1
        X_test_idx = filtered_metadata['group'] == 2
        metadata_train = filtered_metadata[X_train_idx].reset_index(drop=True)
        metadata_test = filtered_metadata[X_test_idx].reset_index(drop=True)
    
        # 비율로 분할
        # X_train_idx, X_test_idx = train_test_split(
        #     np.arange(len(filtered_metadata)),
        #     test_size=0.2,
        #     random_state=random_state,
        #     stratify=filtered_metadata['stratification_label']
        # )
        # metadata_train = filtered_metadata.iloc[X_train_idx].reset_index(drop=True)
        # metadata_test = filtered_metadata.iloc[X_test_idx].reset_index(drop=True)       
        # 데이터 분할 적용
        X_train = filtered_spectrograms[X_train_idx]
        X_test = filtered_spectrograms[X_test_idx]

        
        # 분할 후 데이터 분포 확인
        print("\nAfter splitting - Training set distribution:")
        print_distribution(metadata_train)
        print("\nAfter splitting - Test set distribution:")
        print_distribution(metadata_test)
        y_train = metadata_train['stratification_label'].map(label_to_idx).values
        y_test = metadata_test['stratification_label'].map(label_to_idx).values

        # 클래스별 데이터 수 확인
        unique_train, counts_train = np.unique(y_train, return_counts=True)
        min_samples = np.min(counts_train)
        unique_test, counts_test = np.unique(y_test, return_counts=True)
        min_samples_test = np.min(counts_test)        
        
        # 언더샘플링 수행
        balanced_indices_train = []
        balanced_indices_test = []
        np.random.seed(random_state)  # 재현성을 위한 시드 설정
        
        for label in unique_train:
            label_indices = np.where(y_train == label)[0]
            selected_indices = np.random.choice(label_indices, min_samples, replace=False)
            balanced_indices_train.extend(selected_indices)
        for label in unique_test:
            label_indices = np.where(y_test == label)[0]
            selected_indices = np.random.choice(label_indices, min_samples_test, replace=False)
            balanced_indices_test.extend(selected_indices)
        # 훈련 데이터 균형 맞추기   
        X_train = X_train[balanced_indices_train]
        # X_train = np.array(X_train, dtype=np.float32)
        y_train = y_train[balanced_indices_train]
        metadata_train = metadata_train.iloc[balanced_indices_train].reset_index(drop=True)
        # 테스트 데이터 균형 맞추기
        X_test = X_test[balanced_indices_test]
        # X_test = np.array(X_test, dtype=np.float32)
        y_test = y_test[balanced_indices_test]

        metadata_test = metadata_test.iloc[balanced_indices_test].reset_index(drop=True)

        print("\nTrain class distribution:")
        print(pd.Series(y_train).value_counts())
        print("\nTest class distribution:")
        print(pd.Series(y_test).value_counts())        
        print(f"\nLabel encoding mapping:")
        for label, idx in label_to_idx.items():
            print(f"{label}: {idx}") 

        return X_train, X_test, y_train, y_test, label_to_idx, metadata_train, metadata_test
    @staticmethod
    def split_data_by_load(stacked_data, metadata_df, load_cond, speed_cond, label_column='bearing_condition'):
        """
        load_cond에 따라 train/test를 다르게 분할하는 함수
        
        Parameters:
        -----------
        load_cond : str
            'diff_high': High load를 test set으로 사용
            'diff_low': Low load를 test set으로 사용
        """
        # 초기 속도필터링
        if isinstance(speed_cond, list) and all(x in [30,35,40,45,50] for x in speed_cond):
            try:
                sensor_mask = metadata_df['rpm'].isin(speed_cond)
                filtered_metadata = metadata_df[sensor_mask]
                filtered_spectrograms = stacked_data[sensor_mask]
            except:
                print("Speed condition is not valid")    
        # Load 조건에 따른 분할
        if load_cond == 'Diff_load_high':
            # High load를 test set으로 사용
            train_mask = filtered_metadata['load'] == 'Low'
            test_mask = filtered_metadata['load'] == 'High'
        elif load_cond == 'Diff_load_low':
            # Low load를 test set으로 사용
            train_mask = filtered_metadata['load'] == 'High'
            test_mask = filtered_metadata['load'] == 'Low'
        else:
            raise ValueError("load_cond must be either 'Diff_load_high' or 'Diff_load_low'")
        
        # 데이터 분할
        metadata_train = filtered_metadata[train_mask].reset_index(drop=True)
        metadata_test = filtered_metadata[test_mask].reset_index(drop=True)
        X_train = filtered_spectrograms[train_mask]
        X_test = filtered_spectrograms[test_mask]
        
        # 분포 확인
        print("\nTraining set distribution:")
        print("Load conditions:", metadata_train['load'].unique())
        print("Bearing conditions:", metadata_train['bearing_condition'].value_counts())
        
        print("\nTest set distribution:")
        print("Load conditions:", metadata_test['load'].unique())
        print("Bearing conditions:", metadata_test['bearing_condition'].value_counts())
        
        # Label encoding
        unique_labels = sorted(filtered_metadata['bearing_condition'].unique())
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        
        # 레이블 변환
        y_train = metadata_train['bearing_condition'].map(label_to_idx).values
        y_test = metadata_test['bearing_condition'].map(label_to_idx).values
        
        # 클래스별 데이터 수 확인 및 균형 맞추기
        unique_train, counts_train = np.unique(y_train, return_counts=True)
        min_samples = np.min(counts_train)
        unique_test, counts_test = np.unique(y_test, return_counts=True)
        min_samples_test = np.min(counts_test)
        
        # 언더샘플링
        balanced_indices_train = []
        balanced_indices_test = []
        np.random.seed(random_state)
        
        for label in unique_train:
            label_indices = np.where(y_train == label)[0]
            selected_indices = np.random.choice(label_indices, min_samples, replace=False)
            balanced_indices_train.extend(selected_indices)
        
        for label in unique_test:
            label_indices = np.where(y_test == label)[0]
            selected_indices = np.random.choice(label_indices, min_samples_test, replace=False)
            balanced_indices_test.extend(selected_indices)
        
        # 균형잡힌 데이터셋 생성
        X_train = X_train[balanced_indices_train]
        y_train = y_train[balanced_indices_train]
        metadata_train = metadata_train.iloc[balanced_indices_train].reset_index(drop=True)
        
        X_test = X_test[balanced_indices_test]
        y_test = y_test[balanced_indices_test]
        metadata_test = metadata_test.iloc[balanced_indices_test].reset_index(drop=True)
        
        print("\nAfter balancing:")
        print("Train class distribution:", pd.Series(y_train).value_counts())
        print("Test class distribution:", pd.Series(y_test).value_counts())
        print("\nLabel encoding mapping:")
        for label, idx in label_to_idx.items():
            print(f"{label}: {idx}")
        
        return X_train, X_test, y_train, y_test, label_to_idx, metadata_train, metadata_test    
def print_distribution(metadata):
    print("\nRotor condition distribution:")
    print(metadata['bearing_condition'].value_counts())
