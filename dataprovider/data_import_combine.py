import sys, os
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from dataprovider.KAISTBearing import KAISTDataProcessor, KAISTDataLoader
from dataprovider.UOSBearing import UOSDataProcessor, UOSDataLoader
from dataprovider.CWRUBearing import CWRUBearingDataProcessor, CWRUDataLoader
from dataprovider.KAMPRotor import KAMPRotorDataProcessor, KAMPRotorDataLoader
from dataprovider.PHMGearbox import PHMGearboxDataProcessor, PHMGearboxDataLoader
from dataprovider.IPMSM_MPARN import IPMSM_MPARNDataProcessor, IPMSM_MPARNDataLoader
from dataprovider.KAISTBearing import KAISTDataProcessor, KAISTDataLoader
from dataprovider.WaveGlove import WaveGloveDataProcessor, WaveGloveDataLoader
from dataprovider.SNUWaveGlove import SNUWaveGloveDataProcessor, SNUWaveGloveDataLoader
def data_import_combine(data_type,base_dir, condition, test_condition, noise_level, label_column, dim_type, window_size, overlap):
    if 'UOS' in data_type:
        sampling_rate = 8000
        data_path = base_dir+data_type+"/SamplingRate_"+str(sampling_rate)+"/"
        f_list = os.listdir(data_path)
        
        save_dir = str(project_root)
        save_path = save_dir + "/data/"+data_type+"/SamplingRate_"+str(sampling_rate)+"/"
        os.makedirs(save_path, exist_ok=True)
        
        all_datastacks, all_metadata = UOSDataProcessor.collect_all_data(
            data_path, f_list, dimension_type=dim_type, window_size=window_size, overlap=overlap)

        (X_train, X_test, y_train, y_test, label_to_idx, meta_train, meta_test) = UOSDataLoader.load_uos_data(
            str(project_root), all_datastacks, all_metadata, data_type, condition, test_condition, label_column)

    elif 'KAIST' in data_type:
        data_path = base_dir+data_type+"/"
        f_list = os.listdir(data_path)
        
        all_datastacks, all_metadata = KAISTDataProcessor.collect_all_data(
            data_path, f_list, dimension_type=dim_type, window_size=window_size, overlap=overlap)

        load_cond = test_condition
        severity_cond = condition
        (X_train, X_test, y_train, y_test, label_to_idx, meta_train, meta_test) = KAISTDataLoader.load_kaist_data(
            all_datastacks, all_metadata, load_cond, severity_cond, noise_level, label_column)

    elif 'CWRU' in data_type:
        data_path = base_dir+data_type+"/"
        f_list = os.listdir(data_path)
        
        all_datastacks, all_metadata = CWRUBearingDataProcessor.collect_all_data(
            data_path, f_list, dimension_type=dim_type, window_size=window_size, overlap=overlap)

        severity_cond = condition # "min_even, max_even, all_even, diff_load" (#severity를 어떻게 고를건지)
        load_cond = test_condition # [0,1,2,3] load를 어떻게 고를건지
        (X_train, X_test, y_train, y_test, label_to_idx, meta_train, meta_test) = CWRUDataLoader.load_cwru_data(
            all_datastacks, all_metadata, severity_cond, load_cond, noise_level, label_column)
    elif 'KAMP_Rotor' in data_type:
        data_path = base_dir+data_type+"/"
        f_list = os.listdir(data_path)

        all_datastacks, all_metadata = KAMPRotorDataProcessor.collect_all_data(
            data_path, f_list, dimension_type=dim_type, window_size=window_size, overlap=overlap)
        sensor_cond = condition
        load_cond = test_condition
        
        (X_train, X_test, y_train, y_test, label_to_idx, meta_train, meta_test) = KAMPRotorDataLoader.load_kamp_rotor_data(
            str(project_root), all_datastacks, all_metadata, load_cond, sensor_cond, label_column)
    elif 'PHM_Gearbox' in data_type:
        data_path = base_dir+"[9] 2009 PHM Data Challenge (gearbox)/phm09 dataset/00_Dataset/"
        f_list = os.listdir(data_path)

        all_datastacks, all_metadata = PHMGearboxDataProcessor.collect_all_data(
            data_path, f_list, data_type, dimension_type=dim_type, window_size=window_size, overlap=overlap)
        load_cond = condition
        speed_cond = test_condition        
        (X_train, X_test, y_train, y_test, label_to_idx, meta_train, meta_test) = PHMGearboxDataLoader.load_phm_gearbox_data(
            all_datastacks, all_metadata, load_cond, speed_cond, noise_level, label_column)
    elif 'IPMSM_MPARN' in data_type:
        data_path = base_dir+"IPMSM/MPARN_final_selected/"
        f_list = os.listdir(data_path)
        save_dir = str(project_root)
        save_path = save_dir + "/data/"+data_type+"/"
        os.makedirs(save_path, exist_ok=True)

        all_datastacks, all_metadata = IPMSM_MPARNDataProcessor.collect_all_data(
            data_path, f_list, data_type, dimension_type=dim_type, window_size=window_size, overlap=overlap)
        speed_cond = test_condition
        load_cond = condition
        (X_train, X_test, y_train, y_test, label_to_idx, meta_train, meta_test) = IPMSM_MPARNDataLoader.load_ipmsm_mparn_data(
            str(project_root), all_datastacks, all_metadata, load_cond, speed_cond,label_column)
    elif 'Waveglove_m' in data_type:
        data_path = base_dir+"Wavegloveset/"
        f_list = os.listdir(data_path)

        all_datastacks, all_metadata = WaveGloveDataProcessor.collect_all_data(
            data_path, f_list, data_type, dimension_type=dim_type, window_size=window_size, overlap=overlap)
        load_cond = condition
        speed_cond = test_condition        
        (X_train, X_test, y_train, y_test, label_to_idx, meta_train, meta_test) = WaveGloveDataLoader.load_waveglove_data(
            all_datastacks, all_metadata, load_cond, speed_cond, noise_level, label_column) 
    elif 'SNUWG_selftouch' in data_type:
        data_path = "G:/My Drive/2. 연구/3. hand gesture recognition using IMU data/log_seq_250311/w_o object/"
        f_list = os.listdir(data_path)  # 250310_1_hr/, prof/ ..

        all_datastacks, all_metadata = SNUWaveGloveDataProcessor.collect_all_data(
            data_path, f_list, data_type, dimension_type=dim_type, window_size=window_size, overlap=overlap)
        load_cond = condition
        speed_cond = test_condition        
        (X_train, X_test, y_train, y_test, label_to_idx, meta_train, meta_test) = SNUWaveGloveDataLoader.load_waveglove_data(
            all_datastacks, all_metadata, load_cond, speed_cond, noise_level, label_column)                  
    else:
        print("Unsupported data set")                
    
    print("\nFinal dataset shapes:")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Train metadata shape: {meta_train.shape}")
    print(f"Test metadata shape: {meta_test.shape}")    

    return X_train, X_test, y_train, y_test, label_to_idx, meta_train, meta_test
