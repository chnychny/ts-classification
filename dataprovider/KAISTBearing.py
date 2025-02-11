import os, sys
from scipy import io
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# THK Bearing Data Import
# source:https://data.mendeley.com/datasets/2cygy6y4rk/1
base_dir = str(project_root)
random_state = 1234
class KAISTDataProcessor:
    # def __init__(self, config):
    #     self.config = config

    def collect_all_data(base_path, f_list, dimension_type='time_freq', window_size=1024, overlap=128):
        if dimension_type == 'time_freq':
            all_datasets, all_metadata_dfs = KAISTDataProcessor.process_files_with_origin_STFT(f_list, base_path)
        elif dimension_type == 'time_domain':
            all_datasets, all_metadata_dfs = KAISTDataProcessor.process_files_with_origin_time_domain(f_list, base_path,window_size, overlap)
        elif dimension_type == 'freq_domain':
            sampling_rate = 25600
            all_datasets, all_metadata_dfs = KAISTDataProcessor.process_files_with_origin_freq_domain(f_list, base_path,window_size, overlap,sampling_rate)
       
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
        
        # 부하 조건 추출 (파일명 끝의 .mat 앞 숫자)
        severity = ''.join(filter(str.isdigit, filename.split('.')[0][-1]))  # 숫자만 추출
        meta['severity'] = int(severity) if severity else 0  # 숫자가 있으면 int로 변환, 없으면 0
        load = filename.split('Nm')[0][0]  # 마지막 숫자 추출
        meta['load'] = int(load)    
        # 베어링 상태 추출
        if 'normal' in filename.lower():
            meta['bearing_condition'] = 'normal'
        elif 'bpfi' in filename.lower():
            meta['bearing_condition'] = 'inner_race_fault'
        elif 'bpfo' in filename.lower():
            meta['bearing_condition'] = 'outer_race_fault'
        elif 'unbal' in filename.lower():
            meta['bearing_condition'] = 'unbalance'
        elif 'misalign' in filename.lower():
            meta['bearing_condition'] = 'misalignment'
        else:        
            meta['bearing_condition'] = 'unknown'
        
        return meta

    @staticmethod
    def process_files_with_origin_STFT(file_list, data_path, rpm_folder):
        """
        STFT 데이터 처리
        
        Parameters:
        -----------
        file_list : list
            처리할 파일 목록
        data_path : str
            데이터 경로
        rpm_folder : list
            RPM 폴더 목록
            
        Returns:
        --------
        all_spectrograms : numpy.ndarray
            처리된 STFT 데이터
        metadata_df : pandas.DataFrame
            메타데이터
        """
        metadata_list = []
        spectrograms = []
        
        for file in file_list:
            if file.endswith('.mat'):
                # 메타데이터 생성
                metadata = KAISTDataProcessor.create_metadata(file.replace('.mat', ''))
                
                # .mat 파일 로드
                raw = io.loadmat(data_path + rpm_folder[0] + '/' + file)
                
                # Spectrogram 데이터 추출
                spec_data = raw['Spectrogram']  # shape: (freq_bins, time_steps, num_samples)
                num_samples = spec_data.shape[0]  # 한 파일 내의 샘플 수
                
                # 각 샘플에 대해 메타데이터 복제
                sample_metadata = [{
                    'filename': file,
                    'sample_idx': i,
                    'rotating_condition': metadata['rotating_condition'],
                    'rotating_condition_severity': metadata['rotating_condition_severity'],
                    'bearing_condition': metadata['bearing_condition'],
                    'sampling_rate': metadata['sampling_rate'],
                    'bearing_model': metadata['bearing_model'],
                    'rpm': metadata['rpm']
                } for i in range(num_samples)]
                
                metadata_list.extend(sample_metadata)
                spectrograms.append(spec_data)
        
        # 모든 spectrogram 데이터 연결
        all_spectrograms = np.concatenate(spectrograms, axis=0)
        
        # 메타데이터를 DataFrame으로 변환
        metadata_df = pd.DataFrame(metadata_list)
        
        print(f"Final spectrogram shape: {all_spectrograms.shape}")
        print(f"Metadata DataFrame shape: {metadata_df.shape}")
        
        return all_spectrograms, metadata_df
    
    @staticmethod
    def process_files_with_origin_time_domain(file_list, data_dir, window_size=1024, overlap=128):
        """
        시계열 데이터를 윈도우 크기로 자르고 오버랩을 적용하여 처리
        """
        all_segments = []
        metadata_list = []
        
        for file in file_list:
            if file.endswith('.mat'):
                # 메타데이터 생성
                metadata = KAISTDataProcessor.create_metadata(file)
                
                # .mat 파일 로드
                data = io.loadmat(data_dir + file)
                
                # DE_time 키 찾기
                signal_struct = data['Signal'][0,0] # (time_steps,)
                x_values = signal_struct['x_values'].flatten()  # 시간 값
                y_values = signal_struct['y_values'].flatten()  # 신호 값
                function_record = signal_struct['function_record']  # 기록 정보    
                time_series_set = y_values['values'][0] # 구조체 조사할땐 dtype.name 으로 이름확인
                # 열 각각 point 1~4 위치의 진동신호 
                # ‘Time Stamp’, x_values['values'][0]
                # y_values ‘x_direction_housing_A’, ‘y_direction_housing_A’, ‘x_direction_housing_B’, and ‘y_direction_housing_B’. 
                # 고장은 A housing 안에 있는 곳에 냈다. x방향이 수평, y방향이 수직
                time_series = time_series_set[:, 1]
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
                        'bearing_condition': metadata['bearing_condition'],
                        'load': metadata['load'],
                        'severity': metadata['severity']
                    }
                    metadata_list.append(segment_metadata)
        
        # 모든 세그먼트를 numpy 배열로 변환
        stacked_data = np.array(all_segments)
        
        # 메타데이터를 DataFrame으로 변환
        metadata_df = pd.DataFrame(metadata_list)
        
        # bearing_condition 별로 severity 값을 1,2,3으로 치환
        for condition in metadata_df['bearing_condition'].unique():
            condition_mask = metadata_df['bearing_condition'] == condition
            severity_values = sorted(metadata_df.loc[condition_mask, 'severity'].unique())
            severity_map = {val: i+1 for i, val in enumerate(severity_values)}
            metadata_df.loc[condition_mask, 'severity'] = metadata_df.loc[condition_mask, 'severity'].map(severity_map)
        
        print(f"Processed time domain data shape: {stacked_data.shape}")
        print(f"Metadata DataFrame shape: {metadata_df.shape}")
        
        return stacked_data, metadata_df
    @staticmethod    
    def process_files_with_origin_freq_domain(file_list, data_dir, window_size=1024, overlap=128, sampling_rate=25600):
        """
        시계열 데이터를 윈도우 크기로 자르고 오버랩을 적용하여 fft하여 출력
        """
        all_segments = []
        metadata_list = []
        
        for file in file_list:
            if file.endswith('.mat'):
                # 메타데이터 생성
                metadata = KAISTDataProcessor.create_metadata(file)
                
                # .mat 파일 로드
                data = io.loadmat(data_dir + file)
                
                # DE_time 키 찾기
                signal_struct = data['Signal'][0,0] # (time_steps,)
                x_values = signal_struct['x_values'].flatten()  # 시간 값
                y_values = signal_struct['y_values'].flatten()  # 신호 값
                function_record = signal_struct['function_record']  # 기록 정보    
                time_series_set = y_values['values'][0] # 구조체 조사할땐 dtype.name 으로 이름확인
                # 열 각각 point 1~4 위치의 진동신호 
                # ‘Time Stamp’, x_values['values'][0]
                # y_values ‘x_direction_housing_A’, ‘y_direction_housing_A’, ‘x_direction_housing_B’, and ‘y_direction_housing_B’. 
                # 고장은 A housing 안에 있는 곳에 냈다. x방향이 수평, y방향이 수직
                time_series = time_series_set[:, 1]
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
                    all_segments.append(segment_power)  # 또는 segment_power 또는 segment_db

                    # 각 세그먼트에 대한 메타데이터 생성
                    segment_metadata = {
                        'filename': file,
                        'segment_idx': i,
                        'start_idx': start,
                        'end_idx': end,
                        'bearing_condition': metadata['bearing_condition'],
                        'load': metadata['load'],
                        'severity': metadata['severity']
                    }
                    metadata_list.append(segment_metadata)
        
        # 모든 세그먼트를 numpy 배열로 변환
        stacked_data = np.array(all_segments)
        
        # 메타데이터를 DataFrame으로 변환
        metadata_df = pd.DataFrame(metadata_list)
        
        # bearing_condition 별로 severity 값을 1,2,3으로 치환
        for condition in metadata_df['bearing_condition'].unique():
            condition_mask = metadata_df['bearing_condition'] == condition
            severity_values = sorted(metadata_df.loc[condition_mask, 'severity'].unique())
            severity_map = {val: i+1 for i, val in enumerate(severity_values)}
            metadata_df.loc[condition_mask, 'severity'] = metadata_df.loc[condition_mask, 'severity'].map(severity_map)
        
        print(f"Processed time domain data shape: {stacked_data.shape}")
        print(f"Metadata DataFrame shape: {metadata_df.shape}")
        
        return stacked_data, metadata_df
class KAISTDataLoader:
    @staticmethod
    def load_kaist_data(stacked_data, metadata_df, load_cond=[2], severity_cond='max', noise_level=0.1, label_column='bearing_condition'):
        """
        데이터를 로드하고 필요한 경우 SNR dB 기준으로 노이즈를 추가하는 함수
        
        Parameters:
        -----------
        snr_db : float or None
            Signal-to-Noise Ratio (dB). None이면 노이즈를 추가하지 않음
            예: 20은 20dB SNR의 노이즈 추가 (높을수록 노이즈가 적음) (0, -2. -4, -8)
        """        
        if "diff" in severity_cond: 
            speed_list = list(sorted(metadata_df['load'].unique()))
            X_train, X_test, y_train, y_test, label_to_idx, metadata_train, metadata_test = KAISTDataLoader.split_data_by_load(stacked_data, metadata_df, load_cond, severity_cond)
        else: # even
            X_train, X_test, y_train, y_test, label_to_idx, metadata_train, metadata_test = KAISTDataLoader.split_data_stratified(stacked_data, metadata_df, load_cond,severity_cond, label_column)
        # 노이즈 추가
        if noise_level is not None and noise_level != 0 and (type(noise_level) == float or type(noise_level) == int):
            X_train = KAISTDataLoader.add_noise_snr(X_train, noise_level)
            X_test = KAISTDataLoader.add_noise_snr(X_test, noise_level)
            print(f"Added {noise_level*100}% Gaussian noise to the data")
        elif noise_level != 0 and noise_level.startswith('f'):
            X_train = KAISTDataLoader.add_signal_proportional_noise(X_train, noise_level)
            X_test = KAISTDataLoader.add_signal_proportional_noise(X_test, noise_level)
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
        # KAISTDataLoader.plot_original_vs_noisy(data, noisy_data, snr_db)               
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
        plt.savefig(f'kaistbearing_original_vs_noisy_{noise_level}.png')      
    @staticmethod
    def split_data_by_load(spectrograms, metadata_df, load_cond, severity_cond=None):
        """
        load_cond: 선택할 load 값의 리스트 (예: [0, 1])
        severity_cond: 선택할 severity 값의 리스트 (예: [0, 1]). None이면 모든 severity 포함
        """
        metadata_df = metadata_df.reset_index(drop=True)
        metadata_df['original_index'] = metadata_df.index   
        metadata_df['combined_label'] = metadata_df['bearing_condition'] + metadata_df['load'].astype(str)
        if severity_cond.find('max') != -1 or severity_cond.find('min') != -1:
            try:
                # 각 bearing_condition별로 필터링된 데이터 저장할 리스트
                filtered_metadata_list = []
                filtered_spectrograms_list = []
                # 각 bearing_condition별로 처리
                for condition in metadata_df['bearing_condition'].unique():
                    # 해당 condition의 데이터만 선택
                    condition_mask = metadata_df['bearing_condition'] == condition
                    condition_metadata = metadata_df[condition_mask]
                    condition_spectrograms = spectrograms[condition_mask]
                    
                    # severity의 unique 값 구하기
                    severity_values = condition_metadata['severity'].unique()
                    
                    # max 또는 min severity 값 선택
                    target_severity = max(severity_values) if 'max' in severity_cond else min(severity_values)
                    
                    # 선택된 severity 값에 해당하는 데이터만 필터링
                    severity_mask = condition_metadata['severity'] == target_severity
                    filtered_metadata_list.append(condition_metadata[severity_mask])
                    filtered_spectrograms_list.append(condition_spectrograms[severity_mask])
                
                # 모든 bearing_condition의 필터링된 결과 합치기
                filtered_metadata = pd.concat(filtered_metadata_list, axis=0)
                filtered_spectrograms = np.concatenate(filtered_spectrograms_list, axis=0)
                
            except Exception as e:
                print(f"Severity filtering failed: {str(e)}")
                # severity로 계산할 수 없는 경우 모든 데이터 포함
                pass
        else: # all
            pass       
        
        # Label encoding
        unique_labels = sorted(filtered_metadata['combined_label'].unique())
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}        # 데이터 분할
        valid_indices = filtered_metadata['original_index'].values

        spectrograms = spectrograms[valid_indices]          
        test_mask = filtered_metadata['load'].isin(load_cond)
        train_mask = ~test_mask
        X_train = spectrograms[train_mask]
        X_test = spectrograms[test_mask]
        metadata_train = filtered_metadata[train_mask].reset_index(drop=True)
        metadata_test = filtered_metadata[test_mask].reset_index(drop=True)
 
        y_train = metadata_train['combined_label'].map(label_to_idx).values
        y_test = metadata_test['combined_label'].map(label_to_idx).values

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
            if len(label_indices) > 0:  # 빈 배열이 아닌 경우에만 선택
                selected_indices = np.random.choice(label_indices, min_samples, replace=False)
            else:
                selected_indices = np.array([])  # 빈 배열 반환
            balanced_indices_train.extend(selected_indices)
        for label in unique_test:
            label_indices = np.where(y_test == label)[0]
            selected_indices = np.random.choice(label_indices, min_samples_test, replace=False)
            balanced_indices_test.extend(selected_indices)
        # 훈련 데이터 균형 맞추기   
        X_train = X_train[balanced_indices_train]

        metadata_train = metadata_train.iloc[balanced_indices_train].reset_index(drop=True)
        # 테스트 데이터 균형 맞추기
        X_test = X_test[balanced_indices_test]

        metadata_test = metadata_test.iloc[balanced_indices_test].reset_index(drop=True)
        # 다시 y_train, y_test 변경
        unique_labels_last = sorted(metadata_train['bearing_condition'].unique())
        label_to_idx_last = {label: idx for idx, label in enumerate(unique_labels_last)}    
        y_train = metadata_train['bearing_condition'].map(label_to_idx_last).values
        y_test = metadata_test['bearing_condition'].map(label_to_idx_last).values              
        print("\nTrain class distribution:")
        print(pd.Series(y_train).value_counts())
        print("\nTest class distribution:")
        print(pd.Series(y_test).value_counts())        
        print(f"\nLabel encoding mapping:")
        for label, idx in label_to_idx_last.items():
            print(f"{label}: {idx}") 

        return X_train, X_test, y_train, y_test, label_to_idx_last, metadata_train, metadata_test
    @staticmethod
    def split_data_stratified(spectrograms, metadata_df, load_cond, severity_cond=None, test_size=0.2, label_column='bearing_condition', random_state=random_state):
        """
        load_cond: 선택할 load 값의 리스트 (예: [0, 1])
        severity_cond: 선택할 severity 값의 리스트 (예: [0, 1]). None이면 모든 severity 포함
        """
        # Load 조건으로 필터링
        load_mask = metadata_df['load'].isin(load_cond)
        filtered_metadata = metadata_df[load_mask]
        filtered_spectrograms = spectrograms[load_mask]
        
        # Severity 조건으로 필터링 (지정된 경우)
        # if severity_cond in ['max', 'min']:
        #     try:
        #         # 각 bearing_condition별로 필터링된 데이터 저장할 리스트
        #         filtered_metadata_list = []
        #         filtered_spectrograms_list = []
                
        #         # 각 bearing_condition별로 처리
        #         for condition in filtered_metadata['bearing_condition'].unique():
        #             # 해당 condition의 데이터만 선택
        #             condition_mask = filtered_metadata['bearing_condition'] == condition
        #             condition_metadata = filtered_metadata[condition_mask]
        #             condition_spectrograms = filtered_spectrograms[condition_mask]
                    
        #             # severity의 unique 값 구하기
        #             severity_values = condition_metadata['severity'].unique()
                    
        #             # max 또는 min severity 값 선택
        #             target_severity = max(severity_values) if severity_cond == 'max' else min(severity_values)
                    
        #             # 선택된 severity 값에 해당하는 데이터만 필터링
        #             severity_mask = condition_metadata['severity'] == target_severity
        #             filtered_metadata_list.append(condition_metadata[severity_mask])
        #             filtered_spectrograms_list.append(condition_spectrograms[severity_mask])
                
        #         # 모든 bearing_condition의 필터링된 결과 합치기
        #         filtered_metadata = pd.concat(filtered_metadata_list, axis=0)
        #         filtered_spectrograms = np.concatenate(filtered_spectrograms_list, axis=0)
                
        #     except Exception as e:
        #         print(f"Severity filtering failed: {str(e)}")
        #         # severity로 계산할 수 없는 경우 모든 데이터 포함
        #         pass
        # else:
        #     pass
         # Severity 조건으로 필터링 (지정된 경우)
        if severity_cond.find('max') != -1 or severity_cond.find('min') != -1:
            try:
                # 각 bearing_condition별로 필터링된 데이터 저장할 리스트
                filtered_metadata_list = []
                filtered_spectrograms_list = []
                
                # 각 bearing_condition별로 처리
                for condition in filtered_metadata['bearing_condition'].unique():
                    # 해당 condition의 데이터만 선택
                    condition_mask = filtered_metadata['bearing_condition'] == condition
                    condition_metadata = filtered_metadata[condition_mask]
                    condition_spectrograms = filtered_spectrograms[condition_mask]
                    
                    # severity의 unique 값 구하기
                    severity_values = condition_metadata['severity'].unique()
                    
                    # max 또는 min severity 값 선택
                    target_severity = max(severity_values) if severity_cond == 'max' else min(severity_values)
                    
                    # 선택된 severity 값에 해당하는 데이터만 필터링
                    severity_mask = condition_metadata['severity'] == target_severity
                    filtered_metadata_list.append(condition_metadata[severity_mask])
                    filtered_spectrograms_list.append(condition_spectrograms[severity_mask])
                
                # 모든 bearing_condition의 필터링된 결과 합치기
                filtered_metadata = pd.concat(filtered_metadata_list, axis=0)
                filtered_spectrograms = np.concatenate(filtered_spectrograms_list, axis=0)
                
            except Exception as e:
                print(f"Severity filtering failed: {str(e)}")
                # severity로 계산할 수 없는 경우 모든 데이터 포함
                pass
        else: # all
            pass       
        # 필터링 결과 확인
        print("\nFiltered data distribution:")
        print(f"Load conditions: {load_cond}")
        print(f"Severity conditions: {severity_cond}")
        print(f"Remaining samples: {len(filtered_metadata)}")
        print("\nLoad distribution:")
        print(filtered_metadata['load'].value_counts())

        filtered_metadata['stratification_label'] = filtered_metadata['bearing_condition'] + filtered_metadata['load'].astype(str)
        print("\nSeverity distribution:")
        print(filtered_metadata['stratification_label'].value_counts())
        
        # Label encoding
        unique_labels = sorted(filtered_metadata['bearing_condition'].unique())
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}        # 데이터 분할
        
        
        X_train_idx, X_test_idx = train_test_split(
            np.arange(len(filtered_metadata)),
            test_size=0.2,
            random_state=random_state,
            stratify=filtered_metadata['stratification_label']
        )
        
        # 데이터 분할 적용
        X_train = filtered_spectrograms[X_train_idx]
        X_test = filtered_spectrograms[X_test_idx]
        metadata_train = filtered_metadata.iloc[X_train_idx].reset_index(drop=True)
        metadata_test = filtered_metadata.iloc[X_test_idx].reset_index(drop=True)
        
        # 분할 후 데이터 분포 확인
        print("\nAfter splitting - Training set distribution:")
        print_distribution(metadata_train)
        print("\nAfter splitting - Test set distribution:")
        print_distribution(metadata_test)
        y_train = metadata_train['bearing_condition'].map(label_to_idx).values
        y_test = metadata_test['bearing_condition'].map(label_to_idx).values

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
            if len(label_indices) > 0:  # 빈 배열이 아닌 경우에만 선택
                selected_indices = np.random.choice(label_indices, min_samples, replace=False)
            else:
                selected_indices = np.array([])  # 빈 배열 반환
            balanced_indices_train.extend(selected_indices)
        for label in unique_test:
            label_indices = np.where(y_test == label)[0]
            selected_indices = np.random.choice(label_indices, min_samples_test, replace=False)
            balanced_indices_test.extend(selected_indices)
        # 훈련 데이터 균형 맞추기   
        X_train = X_train[balanced_indices_train]
        y_train = y_train[balanced_indices_train]
        metadata_train = metadata_train.iloc[balanced_indices_train].reset_index(drop=True)
        # 테스트 데이터 균형 맞추기
        X_test = X_test[balanced_indices_test]
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
    
def print_distribution(metadata):
    print("\nBearing condition distribution:")
    print(metadata['bearing_condition'].value_counts())
    print("\nLoad distribution:")
    print(metadata['load'].value_counts())
    print("\nSeverity distribution:")
    print(metadata['severity'].value_counts())    
