import json


import os, sys
from scipy import io
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
random_state = 1234
base_dir = str(project_root)

class SNUWaveGloveDataProcessor:
    # def __init__(self, config):
    #     self.config = config
    @staticmethod
    def collect_all_data(base_path, local_dir, data_type, dimension_type='time_domain', seg_cut = 20):
        if dimension_type == 'time_freq':
            pass
        elif dimension_type == 'time_domain':
            
            all_datasets, all_metadata_dfs = SNUWaveGloveDataProcessor.process_files_with_origin_time_domain(local_dir, base_path,data_type,seg_cut)
        else:
            pass
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
        return meta
    @staticmethod
    def extract_segments(df, x, label_num):
    # label이 6인 데이터 필터링
        df_label_6 = df[df['label'] == label_num].copy()
        
        # beforeGrip에서 afterGrip으로 바뀌는 idx 찾기
        change_indices = df_label_6.index[(df_label_6['grip_time'].shift(1) == 'beforeGrip') & (df_label_6['grip_time'] == 'afterGrip')].tolist()
        
        # 새로운 df_seg 정의
        df_seg = pd.DataFrame()
        for i,idx in enumerate(change_indices):
            # 앞의 x개와 뒤의 x개 추출
            segment = df_label_6.loc[max(0, idx - x):idx + x].copy()# + 1]
            segment['seg_idx'] = i
            segment['change_idx'] = idx
            df_seg = pd.concat([df_seg, segment], ignore_index=True)
        
        return df_seg
    @staticmethod
    def expand_sensor_data(df):
        """
        리스트 형태의 센서 데이터를 개별 컬럼으로 분해하는 함수
        """
        # 새로운 데이터프레임 생성
        expanded_df = df.copy()
        
        # raw_acc 처리 (11개 센서, 3축)
        for sensor in range(11):
            for axis in range(3):
                col_name = f'raw_acc_s{sensor+1}_axis{axis+1}'
                expanded_df[col_name] = df['raw_acc'].apply(lambda x: x[sensor][axis]).copy()
        
        # raw_gyr 처리 (11개 센서, 3축)
        for sensor in range(11):
            for axis in range(3):
                col_name = f'raw_gyr_s{sensor+1}_axis{axis+1}'
                expanded_df[col_name] = df['raw_gyr'].apply(lambda x: x[sensor][axis]).copy()
        
        # joint_pos 처리 (16개 관절, 3차원)
        for joint in range(16):
            for dim in range(3):
                col_name = f'joint_pos_j{joint+1}_dim{dim+1}'
                expanded_df[col_name] = df['joint_pos'].apply(lambda x: x[joint][dim]).copy()
        
        # joint_angles 처리 (16개 관절, 4개 쿼터니언)
        for joint in range(16):
            for angle in range(4):
                col_name = f'joint_angles_j{joint+1}_q{angle+1}'
                expanded_df[col_name] = df['joint_angles'].apply(lambda x: x[joint][angle]).copy()
        
        # 원본 리스트 컬럼 삭제
        expanded_df = expanded_df.drop(['raw_acc', 'raw_gyr', 'joint_pos', 'joint_angles'], axis=1)
        
        return expanded_df    
    def plot_sensor_data_comparison(df_seg_cat, label_num, seg_idx_num, seg_cut, fig_save_dir):
        """
        특정 label과 seg_idx에 대해 change_idx를 기준으로 센서 데이터를 시각화합니다.
        
        Parameters:
        -----------
        df_seg_cat : pandas.DataFrame
            분석할 데이터프레임
        label_num : int
            분석할 라벨 번호
        seg_idx_num : int
            분석할 세그먼트 인덱스 번호
        """
        # 특정 label과 seg_idx에 해당하는 데이터 필터링
        filtered_df = df_seg_cat[(df_seg_cat['label'] == label_num) & (df_seg_cat['seg_idx'] == seg_idx_num)]
        filtered_df=filtered_df.reset_index(drop=True)
        
        if filtered_df.empty:
            print(f"No data found for label {label_num}, segment index {seg_idx_num}.")
            return
        
        # change_idx 값 가져오기
        change_idx = filtered_df['change_idx'].iloc[0]
        
        # 데이터 시각화를 위한 준비
        # 각 행의 인덱스를 change_idx를 기준으로 한 상대적 위치로 변환
        relative_change_idx = filtered_df[filtered_df['idx']==change_idx].index[0]
        # 센서 데이터 시각화
        # fig, axes = plt.subplots(4, 1, figsize=(15, 20))
        # fig.suptitle(f'Sensor Data Comparison (Label: {label_num}, Segment: {seg_idx_num}, Change Index: {change_idx})', fontsize=16)
        # 4개의 개별 figure 생성
        plt.style.use('seaborn')  # 더 현대적인 스타일 적용
        # 1. raw_acc 시각화
        fig_acc = plt.figure(figsize=(15, 8))
        # 3개의 subplot 생성 (x, y, z 축 분리)
        axes = [plt.subplot(3, 1, i+1) for i in range(3)]
        
        # 데이터 준비 (기존 코드와 동일)
        data_dict = {'idx': []}
        for sensor in range(11):
            for axis in range(3):
                data_dict[f'sensor_{sensor+1}_axis_{axis+1}'] = []
        
        # 데이터 수집 (기존 코드와 동일)
        for i in range(len(filtered_df)):
            row = filtered_df.iloc[i]
            idx_value = row['idx']
            acc_data = np.array(row['raw_acc'])
            data_dict['idx'].append(idx_value)
            for sensor in range(11):
                for axis in range(3):
                    data_dict[f'sensor_{sensor+1}_axis_{axis+1}'].append(acc_data[sensor, axis])
        
        sensor_df = pd.DataFrame(data_dict)
        # 각 축별로 그래프 그리기
        axis_names = ['X', 'Y', 'Z']
        colors = plt.cm.rainbow(np.linspace(0, 1, 11))  # 11개 센서를 위한 색상
        
        # 모든 라인들을 저장할 리스트
        lines = []
        labels = []

        for axis in range(3):
            ax = axes[axis]
            for sensor in range(11):
                column_name = f'sensor_{sensor+1}_axis_{axis+1}'
                line = ax.plot(sensor_df['idx'], 
                    sensor_df[column_name],
                    '-',  
                    color=colors[sensor],
                    label=f'Sensor {sensor+1}',
                    linewidth=1)
                
                # 첫 번째 subplot의 라인들만 legend에 추가
                if axis == 0:
                    lines.append(line[0])
                    labels.append(f'Sensor {sensor+1}')
            
            # Change Point 라인 추가 (첫 번째 subplot에서만)
            if axis == 0:
                change_line = ax.axvline(x=change_idx, color='r', linestyle='--', label='Change Point')
                lines.append(change_line)
                labels.append('Change Point')
            else:
                ax.axvline(x=change_idx, color='r', linestyle='--')
            
            ax.set_xlabel('Index')
            ax.set_ylabel(f'{axis_names[axis]}-axis')
            ax.grid(True, alpha=0.3)

        # 모든 subplot의 legend를 한번에 표시
        fig_acc.legend(lines, labels, 
                bbox_to_anchor=(1.01, 0.5),  # 오른쪽 중앙에 배치
                loc='center left',
                title='Sensors')
        plt.suptitle(f'Acceleration Data (Label: {label_num}, Segment: {seg_idx_num})')
        # 여백 조정
        plt.tight_layout(rect=[0, 0, 0.92, 1])  # legend를 위한 공간 확보
        plt.savefig(f'{fig_save_dir}/figure_raw_data/[Label {label_num}, Segment {seg_idx_num}] cut{seg_cut} Acceleration.png',bbox_inches='tight')
        # 2. raw_gyr 시각화 (비슷한 방식으로)
        fig_gyr = plt.figure(figsize=(15, 8))
        axes = [plt.subplot(3, 1, i+1) for i in range(3)]
        # 데이터를 저장할 딕셔너리 초기화
        data_dict = {'idx': []}
        for sensor in range(11):  # 11개 센서 가정
            for axis in range(3):  # 3축 가정
                data_dict[f'gyr_sensor_{sensor+1}_axis_{axis+1}'] = []
        
        # 각 행의 데이터를 수집
        for i in range(len(filtered_df)):
            row = filtered_df.iloc[i]
            idx_value = row['idx']  # 현재 행의 idx 값
            gyr_data = np.array(row['raw_gyr'])  # 형태: (11, 3) 가정
            
            # idx 값 저장
            data_dict['idx'].append(idx_value)
            
            # 각 센서와 축의 데이터 저장
            for sensor in range(gyr_data.shape[0]):
                for axis in range(gyr_data.shape[1]):
                    data_dict[f'gyr_sensor_{sensor+1}_axis_{axis+1}'].append(gyr_data[sensor, axis])
        
        # 데이터프레임으로 변환
        gyr_df = pd.DataFrame(data_dict)  

       # 모든 라인들을 저장할 리스트
        lines = []
        labels = []

        for axis in range(3):
            ax = axes[axis]
            for sensor in range(11):
                column_name = f'gyr_sensor_{sensor+1}_axis_{axis+1}'
                line = ax.plot(gyr_df['idx'], 
                    gyr_df[column_name],
                    '-',  
                    color=colors[sensor],
                    label=f'Sensor {sensor+1}',
                    linewidth=1)
                
                # 첫 번째 subplot의 라인들만 legend에 추가
                if axis == 0:
                    lines.append(line[0])
                    labels.append(f'Sensor {sensor+1}')
            
            # Change Point 라인 추가 (첫 번째 subplot에서만)
            if axis == 0:
                change_line = ax.axvline(x=change_idx, color='r', linestyle='--', label='Change Point')
                lines.append(change_line)
                labels.append('Change Point')
            else:
                ax.axvline(x=change_idx, color='r', linestyle='--')
            
            ax.set_xlabel('Index')
            ax.set_ylabel(f'{axis_names[axis]}-axis')
            ax.grid(True, alpha=0.3)

        # 모든 subplot의 legend를 한번에 표시
        fig_gyr.legend(lines, labels, 
                bbox_to_anchor=(1.01, 0.5),  # 오른쪽 중앙에 배치
                loc='center left',
                title='Sensors')
        plt.suptitle(f'Gyroscope Data (Label: {label_num}, Segment: {seg_idx_num})')
        # 여백 조정
        plt.tight_layout(rect=[0, 0, 0.92, 1])  # legend를 위한 공간 확보
        plt.savefig(f'{fig_save_dir}/figure_raw_data/[Label {label_num}, Segment {seg_idx_num}] cut{seg_cut} Gyroscope.png',bbox_inches='tight')

        # 3. joint_pos 시각화
        fig_pos = plt.figure(figsize=(15, 8))
        axes = [plt.subplot(3, 1, i+1) for i in range(3)]
        colors = plt.cm.rainbow(np.linspace(0, 1, 16))  # 16개 관절을 위한 색상
        
        # 데이터를 저장할 딕셔너리 초기화
        data_dict = {'idx': []}
        for joint in range(16):  # 16개 관절 가정
            for dim in range(3):  # 3차원 위치 가정
                data_dict[f'joint_{joint+1}_dim_{dim+1}'] = []
        
        # 각 행의 데이터를 수집
        for i in range(len(filtered_df)):
            row = filtered_df.iloc[i]
            idx_value = row['idx']  # 현재 행의 idx 값
            joint_pos_data = np.array(row['joint_pos'])  # 형태: (16, 3) 가정
            
            # idx 값 저장
            data_dict['idx'].append(idx_value)
            
            # 각 관절과 차원의 데이터 저장
            for joint in range(joint_pos_data.shape[0]):
                for dim in range(joint_pos_data.shape[1]):
                    data_dict[f'joint_{joint+1}_dim_{dim+1}'].append(joint_pos_data[joint, dim])
        
        # 데이터프레임으로 변환
        joint_pos_df = pd.DataFrame(data_dict) 
        # 모든 라인들을 저장할 리스트
        lines = []
        labels = []

        for axis in range(3):
            ax = axes[axis]
            for sensor in range(16):
                column_name = f'joint_{sensor+1}_dim_{axis+1}'
                line = ax.plot(joint_pos_df['idx'], 
                    joint_pos_df[column_name],
                    '-',  
                    color=colors[sensor],
                    label=f'Joint {sensor+1}',
                    linewidth=1)
                
                # 첫 번째 subplot의 라인들만 legend에 추가
                if axis == 0:
                    lines.append(line[0])
                    labels.append(f'Joint {sensor+1}')
            
            # Change Point 라인 추가 (첫 번째 subplot에서만)
            if axis == 0:
                change_line = ax.axvline(x=change_idx, color='r', linestyle='--', label='Change Point')
                lines.append(change_line)
                labels.append('Change Point')
            else:
                ax.axvline(x=change_idx, color='r', linestyle='--')
            
            ax.set_xlabel('Index')
            ax.set_ylabel(f'{axis_names[axis]}-axis')
            ax.grid(True, alpha=0.3)

        # 모든 subplot의 legend를 한번에 표시
        fig_pos.legend(lines, labels, 
                bbox_to_anchor=(1.01, 0.5),  # 오른쪽 중앙에 배치
                loc='center left',
                title='Sensors')
        plt.suptitle(f'Joint Position Data (Label: {label_num}, Segment: {seg_idx_num})')
        # 여백 조정
        plt.tight_layout(rect=[0, 0, 0.92, 1])  # legend를 위한 공간 확보
        plt.savefig(f'{fig_save_dir}/figure_raw_data/[Label {label_num}, Segment {seg_idx_num}] cut{seg_cut} Joint Position.png',bbox_inches='tight')
      
        # 4. joint_angles 시각화
        fig_angles = plt.figure(figsize=(15, 10))
        axes = [plt.subplot(4, 1, i+1) for i in range(4)]
        colors = plt.cm.rainbow(np.linspace(0, 1, 16))  # 16개 관절을 위한 색상
        data_dict = {'idx': []}
        for joint in range(16):  # 11개 관절 각도 가정
            for angle in range(4):  # 4개 각도 값 가정
                data_dict[f'angle_joint_{joint+1}_angle_{angle+1}'] = []
        
        # 각 행의 데이터를 수집
        for i in range(len(filtered_df)):
            row = filtered_df.iloc[i]
            idx_value = row['idx']  # 현재 행의 idx 값
            joint_angles_data = np.array(row['joint_angles'])  # 형태: (11, 4) 가정
            
            # idx 값 저장
            data_dict['idx'].append(idx_value)
            
            # 각 관절과 각도의 데이터 저장
            for joint in range(joint_angles_data.shape[0]):
                for angle in range(joint_angles_data.shape[1]):
                    data_dict[f'angle_joint_{joint+1}_angle_{angle+1}'].append(joint_angles_data[joint, angle])
        
        # 데이터프레임으로 변환
        joint_angles_df = pd.DataFrame(data_dict)
        angle_names = ['W', 'X', 'Y', 'Z']  # 쿼터니언 각도 성분

        lines = []
        labels = []

        for axis in range(4):
            ax = axes[axis]
            for sensor in range(16):
                column_name = f'angle_joint_{sensor+1}_angle_{axis+1}'
                line = ax.plot(joint_angles_df['idx'], 
                    joint_angles_df[column_name],
                    '-',  
                    color=colors[sensor],
                    label=f'Joint {sensor+1}',
                    linewidth=1)
                
                # 첫 번째 subplot의 라인들만 legend에 추가
                if axis == 0:
                    lines.append(line[0])
                    labels.append(f'Joint {sensor+1}')
            
            # Change Point 라인 추가 (첫 번째 subplot에서만)
            if axis == 0:
                change_line = ax.axvline(x=change_idx, color='r', linestyle='--', label='Change Point')
                lines.append(change_line)
                labels.append('Change Point')
            else:
                ax.axvline(x=change_idx, color='r', linestyle='--')
            
            ax.set_xlabel('Index')
            ax.set_ylabel(f'{angle_names[axis]}-component')
            ax.grid(True, alpha=0.3)

        # 모든 subplot의 legend를 한번에 표시
        fig_angles.legend(lines, labels, 
                bbox_to_anchor=(1.01, 0.5),  # 오른쪽 중앙에 배치
                loc='center left',
                title='Sensors')
        plt.suptitle(f'Joint Angles Data (Label: {label_num}, Segment: {seg_idx_num})')
        # 여백 조정
        plt.tight_layout(rect=[0, 0, 0.92, 1])  # legend를 위한 공간 확보
        plt.savefig(f'{fig_save_dir}/figure_raw_data/[Label {label_num}, Segment {seg_idx_num}] cut{seg_cut} Joint Angles.png',bbox_inches='tight')

    @staticmethod
    def process_files_with_origin_time_domain(local_dir, f_list, data_type, seg_cut = 20):
        """
        시계열 데이터를 윈도우 크기로 자르고 오버랩을 적용하여 처리
        """
        # f_list = [x for x in f_list if not x.find('figure')!=-1] # figure가 아닌 파일
        f_list = [x for x in f_list if x.find('log_seq_')!=-1] # log_seq_ 파일만 추출 (키보드실험)
        df_seg_cat = pd.DataFrame()
        processed_data = []  # 모든 파일의 데이터를 누적할 리스트

        # 여러 JSON 파일 처리를 위한 새로운 코드
        for f_list_name in f_list:
            file_exp = local_dir+f_list_name
            # file_path = file_exp+"/log_seq.json"
            file_path = local_dir + '/'+f_list_name
            print(f"Processing file: {f_list_name}")  # 현재 처리 중인 파일 출력
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # 각 파일의 데이터 처리
                for entry in data:
                    label = entry["label"]

                    for grip_time, grip_data in [("beforeGrip", entry["beforeGrip"]), ("afterGrip", entry["afterGrip"])]:
                        for grip_entry in grip_data:
                            bbox = grip_entry["bbox"]
                            raw_acc = grip_entry["raw_acc"]  # list 11, 3x1
                            raw_gyr = grip_entry["raw_gyr"] # list 11, 3x1
                            joint_pos = grip_entry["joint_pos"] # list 16, 3x1
                            joint_angles = grip_entry["joint_angles"] # list 11, 4x1
                            
                            processed_data.append({
                                "label": label,
                                "grip_time": grip_time,
                                "bbox": bbox,
                                "raw_acc": raw_acc,
                                "raw_gyr": raw_gyr,
                                "joint_pos": joint_pos,
                                "joint_angles": joint_angles,
                                "f_list": f_list_name
                                })
            except Exception as e:
                print(f"Error processing file {f_list_name}: {str(e)}")
                continue

        # 누적된 모든 데이터를 DataFrame으로 변환
        df = pd.DataFrame(processed_data)    
        df['idx'] = df.index  # idx 열 추가
        print(df.columns)
        print(df.head())
        ## timestamt(index별 라벨) (before after)
        # 전체 label 종류 파악 라벨의 type이 리스트인 경우, 첫 번째 요소만 추출하여 단일 값으로 변환
        if isinstance(df['label'].iloc[0], list):
            print("Warning: Labels are in list format, converting to single values...")
            df['label'] = df['label'].apply(lambda x: x[0] if isinstance(x, list) else x)        
        unique_labels = sorted(df['label'].unique())
        num_labels = len(unique_labels)
        print(f"Total number of unique labels: {num_labels}")
        print("Labels:", unique_labels)
        # idx별로 label을 그리는 그래프
        plt.figure(figsize=(10, 6))
        plt.scatter(df[df['grip_time']=='beforeGrip']['idx'], df[df['grip_time']=='beforeGrip']['label'], marker='o', color='blue', alpha=0.5)
        plt.scatter(df[df['grip_time']=='afterGrip']['idx'], df[df['grip_time']=='afterGrip']['label'], marker='x', color='red', alpha=0.5)
        plt.title(f'{f_list_name} idx-label (beforeGrip:blue, afterGrip:red)')
        plt.xlabel('idx')
        plt.ylabel('label')
        plt.grid(True)
        plt.savefig(f'{base_dir}/combined_figure/{f_list_name.split(".")[0]} idx-label.png',bbox_inches='tight')
        plt.close()
        # 모든 label과 segment에 대해 그래프 생성

        for label_num in unique_labels:
            df_seg = SNUWaveGloveDataProcessor.extract_segments(df, seg_cut, label_num)
            df_seg_cat = pd.concat([df_seg_cat, df_seg], ignore_index=True)
        aa = df_seg_cat.groupby(['label','seg_idx'])['change_idx'].value_counts()
        # 모든 label과 segment에 대해 그래프 생성
        for label_num in unique_labels:  # label 1부터 12까지
            # 현재 label에 대한 segment 개수 확인
            seg_indices = df_seg_cat[df_seg_cat['label'] == label_num]['seg_idx'].unique()
            
            # for seg_idx_num in seg_indices:
            #     SNUWaveGloveDataProcessor.plot_sensor_data_comparison(
            #         df_seg_cat, 
            #         label_num=label_num, 
            #         seg_idx_num=seg_idx_num, 
            #         seg_cut=seg_cut, 
            #         fig_save_dir=file_exp
            #     )
            #     plt.close('all')  # 메모리 관리를 위해 그래프 객체 정리
        # df_seg_cat의 raw_acc, raw_gyr, joint_pos, joint_angles 데이터를 모두 더해서 하나의 데이터로 만들기기
        df_seg_cat_expanded =SNUWaveGloveDataProcessor.expand_sensor_data(df_seg_cat)
        # raw_acc, raw_gyr, joint_pos, joint_angles로 시작하는 컬럼만 필터링
        sensor_cols = [col for col in df_seg_cat_expanded.columns if col.startswith(('raw_acc', 'raw_gyr', 'joint_pos', 'joint_angles'))]
        x_data = df_seg_cat_expanded[sensor_cols]  # 필터링된 센서 데이터만 로드
        y_data = df_seg_cat_expanded['label']  # 레이블 데이터 로드
        # 시계열 데이터 처리 tsne

        # numpy 배열로 변환
        stacked_data = np.array(x_data)
        # 메타데이터를 DataFrame으로 변환
        metadata_df = df_seg_cat_expanded
        
        print(f"Processed time domain data shape: {stacked_data.shape}")
        print(f"Metadata DataFrame shape: {metadata_df.shape}")
        

        return stacked_data, metadata_df

    # @staticmethod
    # def process_files_with_origin_time_domain(local_dir, f_list, data_type, seg_cut = 20):
    #     """
    #     시계열 데이터를 윈도우 크기로 자르고 오버랩을 적용하여 처리
    #     """
    #     f_list = [x for x in f_list if not x.find('figure')!=-1]
        
    #     df_seg_cat = pd.DataFrame()
    #     # Load the new JSON file
    #     for f_list_name in f_list:
    #         file_exp = local_dir+f_list_name
    #         file_path = file_exp+"/log_seq.json"
    #         with open(file_path, "r", encoding="utf-8") as f:
    #             data = json.load(f)

    #     # Extracting relevant data and merging beforeGrip and afterGrip
    #         processed_data = []
    #         for entry in data:
    #             label = entry["label"]

    #             for grip_time, grip_data in [("beforeGrip", entry["beforeGrip"]), ("afterGrip", entry["afterGrip"])]:
    #                 for grip_entry in grip_data:
    #                     bbox = grip_entry["bbox"]
    #                     raw_acc = grip_entry["raw_acc"]  # list 11, 3x1
    #                     raw_gyr = grip_entry["raw_gyr"] # list 11, 3x1
    #                     joint_pos = grip_entry["joint_pos"] # list 16, 3x1
    #                     joint_angles = grip_entry["joint_angles"] # list 11, 4x1
                        
    #                     processed_data.append({
    #                         "label": label,
    #                         "grip_time": grip_time,
    #                         "bbox": bbox,
    #                         "raw_acc": raw_acc,
    #                         "raw_gyr": raw_gyr,
    #                         "joint_pos": joint_pos,
    #                         "joint_angles": joint_angles,
    #                         "f_list": f_list_name
    #                         })

    #         # Convert to DataFrame for better analysis
    #         df = pd.DataFrame(processed_data)    
    #         df['idx'] = df.index  # idx 열 추가
    #         ## timestamt(index별 라벨) (before after)

    #         # idx별로 label을 그리는 그래프
    #         # plt.figure(figsize=(10, 6))
    #         # plt.scatter(df[df['grip_time']=='beforeGrip']['idx'], df[df['grip_time']=='beforeGrip']['label'], marker='o', color='blue', alpha=0.5)
    #         # plt.scatter(df[df['grip_time']=='afterGrip']['idx'], df[df['grip_time']=='afterGrip']['label'], marker='x', color='red', alpha=0.5)
    #         # plt.title(f'{f_list_name} idx-label (beforeGrip:blue, afterGrip:red)')
    #         # plt.xlabel('idx')
    #         # plt.ylabel('label')
    #         # plt.grid(True)
    #         # plt.savefig(f'{local_dir}/combined_figure/{f_list_name} idx-label.png',bbox_inches='tight')
    #         # plt.close()
    #     # 모든 label과 segment에 대해 그래프 생성

    #         for label_num in range(1,13):
    #             df_seg = SNUWaveGloveDataProcessor.extract_segments(df, seg_cut, label_num)
    #             df_seg_cat = pd.concat([df_seg_cat, df_seg], ignore_index=True)
    #         aa = df_seg_cat.groupby(['label','seg_idx'])['change_idx'].value_counts()
    #         # 모든 label과 segment에 대해 그래프 생성
    #         for label_num in range(1,13):  # label 1부터 12까지
    #             # 현재 label에 대한 segment 개수 확인
    #             seg_indices = df_seg_cat[df_seg_cat['label'] == label_num]['seg_idx'].unique()
                
    #             # for seg_idx_num in seg_indices:
    #             #     SNUWaveGloveDataProcessor.plot_sensor_data_comparison(
    #             #         df_seg_cat, 
    #             #         label_num=label_num, 
    #             #         seg_idx_num=seg_idx_num, 
    #             #         seg_cut=seg_cut, 
    #             #         fig_save_dir=file_exp
    #             #     )
    #             #     plt.close('all')  # 메모리 관리를 위해 그래프 객체 정리
    #     # df_seg_cat의 raw_acc, raw_gyr, joint_pos, joint_angles 데이터를 모두 더해서 하나의 데이터로 만들기기
    #     df_seg_cat_expanded =SNUWaveGloveDataProcessor.expand_sensor_data(df_seg_cat)
    #     # raw_acc, raw_gyr, joint_pos, joint_angles로 시작하는 컬럼만 필터링
    #     sensor_cols = [col for col in df_seg_cat_expanded.columns if col.startswith(('raw_acc', 'raw_gyr', 'joint_pos', 'joint_angles'))]
    #     x_data = df_seg_cat_expanded[sensor_cols]  # 필터링된 센서 데이터만 로드
    #     y_data = df_seg_cat_expanded['label']  # 레이블 데이터 로드
    #     # 시계열 데이터 처리 tsne

    #     # numpy 배열로 변환
    #     stacked_data = np.array(x_data)
    #     # 메타데이터를 DataFrame으로 변환
    #     metadata_df = df_seg_cat_expanded
        
    #     print(f"Processed time domain data shape: {stacked_data.shape}")
    #     print(f"Metadata DataFrame shape: {metadata_df.shape}")
        

    #     return stacked_data, metadata_df
    @staticmethod    
    def process_files_with_origin_freq_domain(file_list, data_dir, data_type, window_size=1024, overlap=128, sampling_rate=25600):
        """
        시계열 데이터를 윈도우 크기로 자르고 오버랩을 적용하여 fft하여 출력 => 작성 안함.
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
                    metadata = SNUWaveGloveDataProcessor.create_metadata(file_[0])
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

class SNUWaveGloveDataLoader:
    @staticmethod
    def load_waveglove_data(stacked_data, metadata_df, load_cond, speed_cond, noise_level=0, label_column='bearing_condition'):
        if "even" in load_cond:
            X_train, X_test, y_train, y_test, label_to_idx, metadata_train, metadata_test = SNUWaveGloveDataLoader.split_data_stratified(stacked_data, metadata_df, load_cond, speed_cond, label_column)
        else:
            X_train, X_test, y_train, y_test, label_to_idx, metadata_train, metadata_test = SNUWaveGloveDataLoader.split_data_stratified(stacked_data, metadata_df, load_cond, speed_cond, label_column)
        # 노이즈 추가
        if noise_level is not None and noise_level != 0 and (type(noise_level) == float or type(noise_level) == int):
            X_train = SNUWaveGloveDataLoader.add_noise_snr(X_train, noise_level)
            X_test = SNUWaveGloveDataLoader.add_noise_snr(X_test, noise_level)
            print(f"Added {noise_level*100}% Gaussian noise to the data")
        elif noise_level != 0 and noise_level.startswith('f'):
            X_train = SNUWaveGloveDataLoader.add_signal_proportional_noise(X_train, noise_level)
            X_test = SNUWaveGloveDataLoader.add_signal_proportional_noise(X_test, noise_level)
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
        plt.savefig(f'original_vs_noisy_{noise_level}.png')   
    @staticmethod
    def split_data_stratified(spectrograms, metadata_df, load_cond, speed_cond, test_size=0.2, random_state=random_state):
        """
        load_cond: 선택할 load 값의 리스트 (예: [0, 1])
        severity_cond: 선택할 severity 값의 리스트 (예: [0, 1]). None이면 모든 severity 포함
        """
        filtered_metadata = metadata_df
        filtered_spectrograms = spectrograms
                              
        # 필터링 결과 확인
        print("\nFiltered data distribution:")
        print(f"Load conditions: {load_cond}")
        print(f"Sensor conditions: {speed_cond}")
        print(f"Remaining samples: {len(filtered_metadata)}")

        filtered_metadata['stratification_label'] = filtered_metadata['y']
        print("\ncondition distribution:")
        print(filtered_metadata['stratification_label'].value_counts())
        
        # Label encoding
        unique_labels = sorted(filtered_metadata['stratification_label'].unique())
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}        # 데이터 분할
    
        # 비율로 분할
        X_train_idx, X_test_idx = train_test_split(
             np.arange(len(filtered_metadata)),
             test_size=0.2,
             random_state=random_state,
             stratify=filtered_metadata['stratification_label']
         )
        metadata_train = filtered_metadata.iloc[X_train_idx].reset_index(drop=True)
        metadata_test = filtered_metadata.iloc[X_test_idx].reset_index(drop=True)       
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
def print_distribution(metadata):
    print("\nSNU Waveglove condition distribution:")
    print(metadata['y'].value_counts())