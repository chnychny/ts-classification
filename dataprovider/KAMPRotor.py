import os, sys
from scipy import io
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

base_dir = str(project_root)

class KAMPRotorDataProcessor:
    # def __init__(self, config):
    #     self.config = config

    def collect_all_data(base_path, f_list, dimension_type='time_freq', window_size=1024, overlap=128):
        if dimension_type == 'time_freq':
            pass
        elif dimension_type == 'time_domain':
            all_datasets, all_metadata_dfs = KAMPRotorDataProcessor.process_files_with_origin_time_domain(f_list, base_path,window_size, overlap)
        elif dimension_type == 'freq_domain':
            sampling_rate = 1400
            all_datasets, all_metadata_dfs = KAMPRotorDataProcessor.process_files_with_origin_freq_domain(f_list, base_path,window_size, overlap,sampling_rate)
       
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
        meta['group'] = parts[0] # g1, g2       
        meta['sensor_cond'] = int(parts[1].split('.')[0][-1]) # 1,2,3,4
        return meta
    
    @staticmethod
    def process_files_with_origin_time_domain(file_list, data_dir, window_size=1024, overlap=128):
        """
        시계열 데이터를 윈도우 크기로 자르고 오버랩을 적용하여 처리
        """
        all_segments = []
        metadata_list = []
        
        for file in file_list:
            if file.endswith('.csv'):
                # 메타데이터 생성
                metadata = KAMPRotorDataProcessor.create_metadata(file)
                # .mat 파일 로드
                data = pd.read_csv(data_dir + file)
                condition_list = ['time','normal', 'unbalance', 'looseness','unbalance_looseness']
                for col in range(data.shape[1]):
                    if col!=0: # 0은 
                        time_series = data.iloc[:,col]
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
                                'bearing_condition': condition_list[col],
                                'group': metadata['group'],
                                'sensor_cond': metadata['sensor_cond']
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
    def process_files_with_origin_freq_domain(file_list, data_dir, window_size=1024, overlap=128, sampling_rate=25600):
        """
        시계열 데이터를 윈도우 크기로 자르고 오버랩을 적용하여 fft하여 출력
        """
        all_segments = []
        metadata_list = []
        
        for file in file_list:
            if file.endswith('.csv'):
                # 메타데이터 생성
                metadata = KAMPRotorDataProcessor.create_metadata(file)
                
                # .mat 파일 로드
                data = pd.read_csv(data_dir + file)
                condition_list = ['normal', 'unbalance', 'looseness','unbalance_looseness']
                for col in range(data.shape[1]):
                    if col!=0:
                        time_series = data.iloc[:,col]
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
                                'bearing_condition': condition_list[col],
                                'group': metadata['group'],
                                'sensor_cond': metadata['sensor_cond']
                            }
                            metadata_list.append(segment_metadata)
        # 모든 세그먼트를 numpy 배열로 변환
        stacked_data = np.array(all_segments)
        
        # 메타데이터를 DataFrame으로 변환
        metadata_df = pd.DataFrame(metadata_list)

        return stacked_data, metadata_df
    
class KAMPRotorDataLoader:
    @staticmethod
    def load_kamp_rotor_data(sys_path, stacked_data, metadata_df, load_cond, sensor_cond, label_column='bearing_condition'):

        X_train, X_test, y_train, y_test, label_to_idx, metadata_train, metadata_test = KAMPRotorDataLoader.split_data_stratified(stacked_data, metadata_df, load_cond, sensor_cond, label_column)

        return X_train, X_test, y_train, y_test, label_to_idx, metadata_train, metadata_test


    def split_data_stratified(spectrograms, metadata_df, load_cond, sensor_cond, test_size=0.2, random_state=42):
        """
        load_cond: 선택할 load 값의 리스트 (예: [0, 1])
        severity_cond: 선택할 severity 값의 리스트 (예: [0, 1]). None이면 모든 severity 포함
        """
        # Load 조건으로 필터링
        if load_cond == 'unifault':
            load_mask = metadata_df['bearing_condition'].isin(['normal','unbalance','looseness'])
            filtered_metadata = metadata_df[load_mask]
            filtered_spectrograms = spectrograms[load_mask]
        else:
            filtered_metadata = metadata_df
            filtered_spectrograms = spectrograms
        # Severity 조건으로 필터링 (지정된 경우)
        if isinstance(sensor_cond, list) and all(x in [1,2,3,4] for x in sensor_cond):
            try:
                sensor_mask = filtered_metadata['sensor_cond'].isin(sensor_cond)
                filtered_metadata = filtered_metadata[sensor_mask]
                filtered_spectrograms = filtered_spectrograms[sensor_mask]
            except:
                print("Sensor condition is not valid")

                                
        # 필터링 결과 확인
        print("\nFiltered data distribution:")
        print(f"Load conditions: {load_cond}")
        print(f"Sensor conditions: {sensor_cond}")
        print(f"Remaining samples: {len(filtered_metadata)}")

        filtered_metadata['stratification_label'] = filtered_metadata['bearing_condition']
        print("\nSeverity distribution:")
        print(filtered_metadata['stratification_label'].value_counts())
        
        # Label encoding
        unique_labels = sorted(filtered_metadata['stratification_label'].unique())
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}        # 데이터 분할
        
        # 그룹으로 train/test 분할
        X_train_idx = filtered_metadata['group'] == 'g1'
        X_test_idx = filtered_metadata['group'] == 'g2'
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
        np.random.seed(42)  # 재현성을 위한 시드 설정
        
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
    
def print_distribution(metadata):
    print("\nRotor condition distribution:")
    print(metadata['bearing_condition'].value_counts())
