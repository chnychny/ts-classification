import os# 결과 저장을 위한 디렉토리 구조 생성
def create_experiment_dirs(data_type):
    dirs = {
        'metrics': 'results/'+data_type+'/metrics',
        'plots': 'results/'+data_type+'/plots',
        'model_info': 'results/'+data_type+'/model_info',
        'checkpoints': 'results/'+data_type+'/checkpoints'
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        
    return dirs
def create_rep_experiment_dirs(data_type,result_folder_name):
    dirs = {
        'metrics': result_folder_name+'/'+data_type+'/metrics',
        'plots': result_folder_name+'/'+data_type+'/plots',
        'model_info': result_folder_name+'/'+data_type+'/model_info',
        'checkpoints': result_folder_name+'/'+data_type+'/checkpoints'
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        
    return dirs    
