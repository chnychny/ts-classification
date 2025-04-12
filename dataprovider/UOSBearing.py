import os, sys
from scipy import io
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
random_state = 1234
# THK Bearing Data Import
# source:https://data.mendeley.com/datasets/2cygy6y4rk/1
base_dir = str(project_root)

class UOSDataProcessor:
    # def __init__(self, config):
    #     self.config = config

    def collect_all_data(base_path, f_list, dimension_type='time_freq', window_size=1024, overlap=128):
        """
        모든 데이터 수집 및 처리
        
        Parameters:
        -----------
        base_path : str
            기본 데이터 경로
        f_list : list
            처리할 폴더 목록
        dimension_type : str
            데이터 처리 타입 ('time_freq' 또는 'time_domain')
        window_size : int
            time_domain 처리시 윈도우 크기
        overlap : int
            time_domain 처리시 오버랩 크기
            
        Returns:
        --------
        final_datasets : numpy.ndarray
            처리된 최종 데이터셋
        final_metadata : pandas.DataFrame
            최종 메타데이터
        """
        all_datasets = []
        all_metadata_dfs = []
        
        for rpm_folder in f_list:
            print(f"Processing folder: {rpm_folder}")
            # 각 rpm 폴더 내의 파일 리스트 가져오기
            file_list = os.listdir(base_path + rpm_folder)
            
            # 각 rpm 폴더의 데이터 처리
            if dimension_type == 'time_freq':
                stacked_data, metadata_df = UOSDataProcessor.process_files_with_origin_STFT(file_list, base_path, [rpm_folder])
            elif dimension_type == 'time_domain':
                stacked_data, metadata_df = UOSDataProcessor.process_files_with_origin_time_domain(file_list, base_path,
                                                                                [rpm_folder], window_size, overlap)
            elif dimension_type == 'freq_domain':
                sampling_rate = 8000
                stacked_data, metadata_df = UOSDataProcessor.process_files_with_origin_freq_domain(file_list, base_path,
                                                                                [rpm_folder], window_size, overlap,sampling_rate)
            all_datasets.append(stacked_data)
            all_metadata_dfs.append(metadata_df)
            
            print(f"Current folder data shape: {stacked_data.shape}")
        
        # 모든 데이터를 axis=0으로 연결
        final_datasets = np.concatenate(all_datasets, axis=0)
        
        # 모든 메타데이터 DataFrame 연결
        final_metadata = pd.concat(all_metadata_dfs, axis=0, ignore_index=True)
        
        print("\nFinal dataset info:")
        print(f"Data shape: {final_datasets.shape}")
        print(f"Metadata shape: {final_metadata.shape}")
        
        return final_datasets, final_metadata


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
        
        # 파일명을 '_'로 분리
        parts = filename.split('_')
        
        # 회전 부품 상태 (Rotating component condition)
        rotating_state = parts[0][0]  # 첫 번째 문자
        meta['rotating_condition'] = {
            'H': 'healthy',
            'M': 'misalignment',
            'U': 'unbalance',
            'L': 'looseness'
        }.get(rotating_state)
        
        # 회전 부품 심각도 (Severity level)
        meta['rotating_condition_severity'] = int(parts[0][1]) if len(parts[0]) > 1 and parts[0][0] in ['M', 'U'] else 0
        
        # 베어링 상태 (Bearing condition)
        bearing_state = parts[1]
        meta['bearing_condition'] = {
            'H': 'healthy',
            'B': 'ball_fault',
            'IR': 'inner_race_fault',
            'OR': 'outer_race_fault'
        }.get(bearing_state)
        
        # 샘플링 레이트 (Sampling rate)
        meta['sampling_rate'] = int(parts[2]) * 1000  # kHz를 Hz로 변환
        
        # 베어링 모델 (Bearing model)
        meta['bearing_model'] = parts[3]
        
        # 회전 속도 (Rotating speed)
        meta['rpm'] = int(parts[4])
        
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
                metadata = UOSDataProcessor.create_metadata(file.replace('.mat', ''))
                
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
    
    def process_files_with_origin_freq_domain(file_list, data_path, rpm_folder, window_size=1024, overlap=128, sampling_rate=8000):
        """
        시계열 데이터를 윈도우 크기로 자르고 오버랩을 적용하여 처리한다음 fft하여 출력
        """
 
        all_segments = []
        metadata_list = []
        
        for file in file_list:
            if file.endswith('.mat'):
                # 메타데이터 생성
                metadata = UOSDataProcessor.create_metadata(file.replace('.mat', ''))
                # .mat 파일 로드
                raw = io.loadmat(data_path + rpm_folder[0] + '/' + file)
                # 시계열 데이터 추출 (진동 신호)
                time_series = raw['Data'].squeeze()  # (time_steps,)
                # 윈도우 크기와 오버랩을 사용하여 세그먼트로 분할
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
                        'rotating_condition': metadata['rotating_condition'],
                        'rotating_condition_severity': metadata['rotating_condition_severity'],
                        'bearing_condition': metadata['bearing_condition'],
                        'sampling_rate': metadata['sampling_rate'],
                        'bearing_model': metadata['bearing_model'],
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
    
    def process_files_with_origin_time_domain(file_list, data_path, rpm_folder, window_size=1024, overlap=128):
        """
        시계열 데이터를 윈도우 크기로 자르고 오버랩을 적용하여 처리
        
        Parameters:
        -----------
        file_list : list
            처리할 파일 목록
        data_path : str
            데이터 경로
        rpm_folder : list
            RPM 폴더 목록
        window_size : int
            윈도우 크기 (default: 1024)
        overlap : int
            오버랩 크기 (default: 128)
        
        Returns:
        --------
        stacked_data : numpy.ndarray
            처리된 시계열 데이터 (samples, window_size)
        metadata_df : pandas.DataFrame
            메타데이터
        """
        all_segments = []
        metadata_list = []
        
        for file in file_list:
            if file.endswith('.mat'):
                # 메타데이터 생성
                metadata = UOSDataProcessor.create_metadata(file.replace('.mat', ''))
                
                # .mat 파일 로드
                raw = io.loadmat(data_path + rpm_folder[0] + '/' + file)
                
                # 시계열 데이터 추출 (진동 신호)
                time_series = raw['Data'].squeeze()  # (time_steps,)
                
                # 윈도우 크기와 오버랩을 사용하여 세그먼트로 분할
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
                        'rotating_condition': metadata['rotating_condition'],
                        'rotating_condition_severity': metadata['rotating_condition_severity'],
                        'bearing_condition': metadata['bearing_condition'],
                        'sampling_rate': metadata['sampling_rate'],
                        'bearing_model': metadata['bearing_model'],
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

class UOSDataLoader:
    @staticmethod
    def load_uos_data(sys_path, stacked_data, metadata_df, data_type='UOS_ALL', condition='diff_speed', test_condition='[1000, 1600]', label_column='bearing_condition'):
        """UOS 관련 데이터 로드 함수"""
        if data_type == 'UOS_RTES_tapered_roller_bearing':
            X_train, X_test, y_train, y_test, label_to_idx, metadata_train, metadata_test = UOSDataLoader.process_split(stacked_data, metadata_df, condition,test_condition, label_column)
        elif data_type == 'UOS_RTES_deep_groove_ball_bearing':
            # data_dir = sys_path + '/data/UOS_RTES_deep_groove_ball_bearing/SamplingRate_8000/'
            X_train, X_test, y_train, y_test, label_to_idx, metadata_train, metadata_test = UOSDataLoader.process_split(stacked_data, metadata_df, condition,test_condition, label_column)
        elif data_type == 'UOS_ALL':
            bearing_types = [
                'deep_groove_ball_bearing',
                'tapered_roller_bearing',
                'cylindrical_roller_bearing'
            ]
            X_train, X_test, y_train, y_test, label_to_idx, metadata_train, metadata_test = UOSDataLoader.load_all_bearing_data(sys_path, bearing_types, condition, test_condition, label_column)
        else:
            raise ValueError("Unsupported data type")

        return X_train, X_test, y_train, y_test, label_to_idx, metadata_train, metadata_test

    @staticmethod
    def load_all_bearing_data(sys_path, bearing_types, condition, label_column):
        # ... (기존 UOS_ALL 데이터 로드 로직 유지)
        pass

    @staticmethod
    def process_split(stacked_data, metadata_df, condition, test_condition, label_column):

        healthy_mask = (metadata_df[label_column].str.contains('healthy', case=False) | 
                            metadata_df['rotating_condition'].str.contains('healthy', case=False))
        
        if "Uni" in condition: # Healthy 데이터만 필터링 (단일고장모드 추출)
            stacked_data = stacked_data[healthy_mask]
            metadata_df = metadata_df[healthy_mask]
            metadata_df.loc[:, 'combined_label'] = metadata_df[label_column] + '_' + metadata_df['rotating_condition'] # UOS는 단일고장에 I, O, B와 L, M, U 가 있다. 

        elif "Compound" in condition:
            metadata_df.loc[:, 'combined_label'] = metadata_df[label_column] + '_' + metadata_df['rotating_condition']

        if "diff_speed" in condition:
            speed_list = list(sorted(metadata_df['rpm'].unique()))
            X_train, X_test, metadata_train, metadata_test = UOSDataLoader.split_data_by_rpm(stacked_data, metadata_df, test_condition)

        elif "total" in condition:
            selected_speed=test_condition
            test_size =0.2
            X_train, X_test, metadata_train, metadata_test = UOSDataLoader.split_data_stratified(stacked_data, metadata_df, selected_speed,test_size, random_state)
        # Label encoding
        unique_labels = sorted(metadata_df['combined_label'].unique())
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}       
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
            selected_indices = np.random.choice(label_indices, min_samples, replace=False)
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
    
    def split_data_by_rpm(spectrograms, metadata_df, test_speed):
        """
        test_speed: list of RPM values for test set (e.g., [1200, 1800])
        """
        # 심각도조건 선별
        metadata_df = metadata_df.reset_index(drop=True)
        metadata_df['original_index'] = metadata_df.index        
        def filter_max_severity(group):
            if len(group['rotating_condition_severity'].unique()) > 1:
                # 여러 종류의 severity가 있는 경우 최대값만 선택
                max_severity = group['rotating_condition_severity'].max()
                return group[group['rotating_condition_severity'] == max_severity]
            # 한 종류의 severity만 있는 경우 그대로 반환
            return group        
        # 인덱스 저장      
        filtered_metadata = metadata_df.groupby('combined_label', group_keys=False).apply(filter_max_severity)
        # 필터링된 인덱스만 spectrograms에서 선택
        valid_indices = filtered_metadata['original_index'].values
        spectrograms = spectrograms[valid_indices]          
        test_mask = filtered_metadata['rpm'].isin(test_speed)
        train_mask = ~test_mask
        X_train = spectrograms[train_mask]
        X_test = spectrograms[test_mask]
        metadata_train = filtered_metadata[train_mask].reset_index(drop=True)
        metadata_test = filtered_metadata[test_mask].reset_index(drop=True)
        
        # 분할 결과 출력
        print("\nTrain set RPM distribution:")
        print(metadata_train['rpm'].value_counts())
        print("\nTest set RPM distribution:")
        print(metadata_test['rpm'].value_counts())
        
        return X_train, X_test, metadata_train, metadata_test

    def split_data_stratified(spectrograms, metadata_df, test_speed,test_size=0.2, random_state=random_state):
        speed_mask = metadata_df['rpm'].isin(test_speed)
        spectrograms = spectrograms[speed_mask]
        metadata_df = metadata_df[speed_mask]   
        # 심각도조건 선별
        def filter_max_severity(group):
            if len(group['rotating_condition_severity'].unique()) > 1:
                # 여러 종류의 severity가 있는 경우 최대값만 선택
                max_severity = group['rotating_condition_severity'].max()
                return group[group['rotating_condition_severity'] == max_severity]
            # 한 종류의 severity만 있는 경우 그대로 반환
            return group        
        metadata_df = metadata_df.groupby('combined_label', group_keys=False).apply(filter_max_severity)
            
        # 데이터 분할
        X_train_idx, X_test_idx = train_test_split(
            np.arange(len(metadata_df)),
            test_size=test_size,
            random_state=random_state,
            stratify=metadata_df['combined_label']
        )  
        # 스펙트로그램 데이터 분할
        X_train = spectrograms[X_train_idx]
        X_test = spectrograms[X_test_idx]
        
        # 메타데이터 분할
        metadata_train = metadata_df.iloc[X_train_idx].reset_index(drop=True)
        metadata_test = metadata_df.iloc[X_test_idx].reset_index(drop=True)
        
        return X_train, X_test, metadata_train, metadata_test