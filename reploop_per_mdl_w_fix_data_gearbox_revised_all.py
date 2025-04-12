import torch
import numpy as np
import sys, os, gc
from pathlib import Path
project_root = Path(__file__).parent
sys.path.append(str(project_root))
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report
from dataprovider.base_preprocess import BaseTimeDataset
from utils.training_tools import TrainingHistory
from utils.training_tools import evaluate
from utils.training_tools import train_epoch, get_model_size
from utils.training_tools_revised import train_epoch_revised
from utils.utils import create_rep_experiment_dirs  #
from models.att_resnet import projDilResnet3, projDil22Resnet3, projDil3Resnet3, projDil2ConvResnet3,projDilConv2Resnet3,DilResnet
from models.projDilResTransformer_rev import ImprovedProjDilResTransformer2, DilResTransformer
from models.custom_resnet import baseResnet, projResnet
from models.basictransformer import BasicTransformer
from models.signal_transformers import SignalTransformer
from dataprovider.data_import_combine import data_import_combine
from datetime import datetime
from sklearn.model_selection import train_test_split
import pandas as pd 

def train_model(model, model_name, label_to_idx, mdl_rep_num, exp_rep_num):
    # 모델 설정은 외부에서 받음
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )       
 
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99) 
    cycle_update = False
    class_names = [label for label, _ in sorted(label_to_idx.items(), key=lambda x: x[1])]
    
    # exp_name 설정
    exp_name = f'{data_type}-{condition}-{str(test_condition)}-n{noise_level}-{model_name}-exp_rep{exp_rep_num}-mdl_rep{mdl_rep_num}-epoch{epochs}-lr{lr}-bsize{batch_size}-{suffix}'
    
    # 모델 정보 저장
    model_info = get_model_size(model)
    print(f"\nTraining {model_name}")
    print("Model Information:")
    print(f"Total Parameters: {model_info['total_params']:,}")
    print(f"Trainable Parameters: {model_info['trainable_params']:,}")
    print(f"Model Size: {model_info['model_size_mb']:.2f} MB")
    
    model_info_path = os.path.join(result_dirs['model_info'], f'model_info_{exp_name}.txt')
    with open(model_info_path, 'w') as f:
        f.write(f"Experiment Name: {exp_name}\n")
        f.write(f"Model Architecture: {model.__class__.__name__}\n")
        f.write(f"Total Parameters: {model_info['total_params']:,}\n")
        f.write(f"Trainable Parameters: {model_info['trainable_params']:,}\n")
        f.write(f"Model Size: {model_info['model_size_mb']:.2f} MB\n")
        f.write("\nModel Structure:\n")
        f.write(str(model))
    
    history = TrainingHistory()
    best_acc = 0
    
    # 메트릭 저장을 위한 로그 파일
    metrics_log_path = os.path.join(result_dirs['metrics'], f'training_log_{exp_name}.csv')
    with open(metrics_log_path, 'w') as f:
        f.write('epoch,train_loss,train_acc,val_loss,val_acc,test_loss,test_acc,learning_rate\n')

    
    accumulation_steps = 4
    for epoch in range(epochs):
        
        # Training with gradient accumulation
        train_loss, train_acc = train_epoch_revised(
            model, train_loader, criterion, optimizer, scheduler, 
            device, accumulation_steps, cycle_update=cycle_update
        )
        
        # Validation
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        if not cycle_update:
            scheduler.step()
        # Learning rate 기록
        current_lr = optimizer.param_groups[0]['lr']
        
        # 메트릭 기록
        history.train_losses.append(train_loss)
        history.train_accs.append(train_acc)
        history.val_losses.append(val_loss)
        history.val_accs.append(val_acc)
        
        # 로그 저장
        with open(metrics_log_path, 'a') as f:
            f.write(f'{epoch},{train_loss},{train_acc},{val_loss},{val_acc},{test_loss},{test_acc},{current_lr}\n')

        print(f'\nEpoch: {epoch+1}/{epochs}')
        print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'Val Loss: {val_loss:.3f} | Val Acc: {val_acc*100:.2f}%')
        print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
        print(f'Current learning rate: {current_lr}')
        
        # Best model 저장
        if val_acc > best_acc:
            print('Saving model...')
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
            }, os.path.join(result_dirs['checkpoints'], f'model_{exp_name}_best.pth'))
            
            # 현재 최고 성능에서의 혼동 행렬 저장
            history.plot_confusion_matrix(
                exp_name=exp_name,
                save_path=os.path.join(result_dirs['plots'], f'confusion_matrix_{exp_name}.png')
            )
        
        # Confusion matrix 업데이트
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                outputs, _ = model(inputs)
                _, preds = outputs.max(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        history.add_confusion_matrix(all_labels, all_preds, class_names)
        
        # 학습 곡선 저장
        history.plot_metrics(
            exp_name=exp_name,
            save_path=os.path.join(result_dirs['plots'], f'training_metrics_{exp_name}.png')
        )
    
    return model, history, criterion, exp_name
# 실험 조합 정의

experiment_configs = [
   
    {
        "data_type": "PHM_Gearbox_spur",
        "condition": "All_load_even",
        "test_condition": [50],
        "noise_level": 10
    }, 

    {
        "data_type": "PHM_Gearbox_spur",
        "condition": "All_load_even",
        "test_condition": [50],
        "noise_level": 5
    }, 

    {
        "data_type": "PHM_Gearbox_spur",
        "condition": "All_load_even",
        "test_condition": [50],
        "noise_level": 0
    },     


    {
        "data_type": "PHM_Gearbox_spur",
        "condition": "All_load_even",
        "test_condition": [50],
        "noise_level": 2
    },     
    {
        "data_type": "PHM_Gearbox_spur",
        "condition": "All_load_even",
        "test_condition": [50],
        "noise_level": 1
    },            
]


# 공통 파라미터 설정
common_params = {
    "window_size": 1024,
    "overlap": 0,
    "dim_type": 'time_domain',
    "batch_size": 64,
    "epochs": 150,
    "lr": 1e-3,
    "base_dir": "C:/Users/USER/pch/DataSet/",
    "save_path": str(project_root) + "/data/",
    "label_column": 'bearing_condition',
    "exp_rep_time": 3,
    "mdl_rep_time": 2
}
random_state_list = [42, 1234,628]
# 메인 실행 코드
if __name__ == "__main__":
    # suffix 설정
    suffix = f'ts_ws{common_params["window_size"]}_ov{common_params["overlap"]}_revised'
    for exp_rep in range(common_params['exp_rep_time']):
        random_state = random_state_list[exp_rep]
        for exp_config in experiment_configs:
            print(f"\n{'='*50}")
            print(f"Starting experiment with configuration:")
            print(f"Data Type: {exp_config['data_type']}")
            print(f"Condition: {exp_config['condition']}")
            print(f"Test Condition: {exp_config['test_condition']}")
            print(f"Suffix: {suffix}")
            print(f"{'='*50}\n")

            # 실험 파라미터 설정
            data_type = exp_config['data_type']
            condition = exp_config['condition']
            test_condition = exp_config['test_condition']
            
            # 공통 파라미터 적용
            window_size = common_params['window_size']
            overlap = common_params['overlap']
            dim_type = common_params['dim_type']
            batch_size = common_params['batch_size']
            epochs = common_params['epochs']
            lr = common_params['lr']
            base_dir = common_params['base_dir']
            save_path = common_params['save_path']
            label_column = common_params['label_column']
            noise_level = exp_config['noise_level']
            # 각 exp_config에 대한 결과를 저장할 DataFrame 생성
            c_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            results_df = pd.DataFrame(columns=[
                'model_name', 'exp_rep_num', 'mdl_rep_num', 
                'test_accuracy','test_loss', 
                'train_accuracy','train_loss',  # train metrics 추가
                'data_type', 'condition', 'test_condition','noise_level',
                'batch_size', 'epochs', 'learning_rate', 'window_size',
                'overlap', 'label_column', 'dim_type', 'log_time',
                'class_names', 'precision_per_class', 'recall_per_class', 
                'f1_per_class', 'support_per_class',
                'macro_precision', 'macro_recall', 'macro_f1',
                'weighted_precision', 'weighted_recall', 'weighted_f1'
            ])
            try:
                # 데이터 로드
                X_train, X_test, y_train, y_test, label_to_idx, meta_train, meta_test = data_import_combine(
                    data_type=data_type,
                    base_dir=base_dir,
                    condition=condition,
                    test_condition=test_condition,
                    noise_level=noise_level,
                    label_column=label_column,
                    window_size=window_size,
                    overlap=overlap,
                    dim_type=dim_type
                )

                indices = np.arange(len(X_train))
                
                # Train/Valid split with indices
                X_train_split, X_valid, y_train_split, y_valid, train_idx, valid_idx = train_test_split(
                    X_train, y_train, indices,
                    test_size=0.1, 
                    random_state=random_state, 
                    stratify=y_train
                )
                # meta_train에 split 정보 추가
                meta_train['split'] = 'train'  # 기본값을 'train'으로 설정
                meta_train.loc[valid_idx, 'split'] = 'valid'  # valid 인덱스에 해당하는 행을 'valid'로 설정
                # 저장 경로 생성 (data_type/exp_data 아래에 저장)
                exp_data_dir = os.path.join('loop_cls_results', exp_config['data_type'], 'exp_data')
                os.makedirs(exp_data_dir, exist_ok=True) 
                exp_data_name = f"{exp_config['data_type']}_{exp_config['condition']}_{exp_config['test_condition']}_n{exp_config['noise_level']}_{suffix}_{c_time}_exp_rep{exp_rep}"
                # meta 데이터와 실험 설정 저장
                exp_data_save_path = os.path.join(exp_data_dir, exp_data_name)
                meta_train.to_csv(f'{exp_data_save_path}_meta_train_exp_rep{exp_rep}.csv', index=True)
                meta_test.to_csv(f'{exp_data_save_path}_meta_test_exp_rep{exp_rep}.csv', index=True)
                # 모든 데이터를 하나의 npz 파일로 저장
                np.savez_compressed(
                    f'{exp_data_save_path}_data_exp_rep{exp_rep}.npz',
                    X_train=X_train_split,
                    X_valid=X_valid,
                    X_test=X_test,
                    y_train=y_train_split,
                    y_valid=y_valid,
                    y_test=y_test,
                    label_to_idx=np.array([label_to_idx], dtype=object),  # 딕셔너리를 numpy 배열로 변환
                    train_indices=train_idx,
                    valid_indices=valid_idx
                )
                # 데이터 로드 예시 코드를 log에 저장
                with open(f'{exp_data_save_path}_data_loading_example_exp_rep{exp_rep}.txt', 'w') as f:
                    f.write("# 데이터 로드 예시 코드\n")
                    f.write("data = np.load('데이터경로/데이터파일명_data.npz')\n")
                    f.write("\n# 개별 데이터 접근\n")
                    f.write("X_train = data['X_train']\n")
                    f.write("X_valid = data['X_valid']\n")
                    f.write("X_test = data['X_test']\n")
                    f.write("y_train = data['y_train']\n")
                    f.write("y_valid = data['y_valid']\n")
                    f.write("y_test = data['y_test']\n")
                    f.write("label_to_idx = data['label_to_idx'][0].item()\n")
                    f.write("train_indices = data['train_indices']\n")
                    f.write("valid_indices = data['valid_indices']\n")                       
                # 데이터셋 및 데이터로더 생성
                train_dataset = BaseTimeDataset(X_train_split, y_train_split, normalize='no')
                valid_dataset = BaseTimeDataset(X_valid, y_valid, normalize='no')
                test_dataset = BaseTimeDataset(X_test, y_test, normalize='no')
                
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                # 모델 설정
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                num_classes = len(np.unique(y_train))

                # 결과 저장 디렉토리 생성
                result_dirs = create_rep_experiment_dirs(data_type, 'loop_cls_results')

                # 실험할 모델들 정의
                models_to_train = [
                    (ImprovedProjDilResTransformer2(in_channels=1, output_channel=64, num_classes=num_classes, stride=2).to(device), 
                    "ImprovedProjDilResTransformer2"),  
                    (projDilResnet3(in_channels=1, output_channel=64, num_classes=num_classes, stride=2).to(device), 
                    "projDilResnet3"),                                              
                    (projDil22Resnet3(in_channels=1, output_channel=64, num_classes=num_classes, stride=2).to(device), 
                    "projDil22Resnet3"), 
                    (projDil3Resnet3(in_channels=1, output_channel=64, num_classes=num_classes, stride=2).to(device), 
                    "projDil3Resnet3"), 
                    (projDil2ConvResnet3(in_channels=1, output_channel=64, num_classes=num_classes, stride=2).to(device), 
                    "projDil2ConvResnet3"), 
                    (projDilConv2Resnet3(in_channels=1, output_channel=64, num_classes=num_classes, stride=2).to(device), 
                    "projDilConv2Resnet3"), 
                    (baseResnet(in_channels=1, output_channel=64, num_classes=num_classes).to(device), 
                    "baseResnet"),   
                    (projResnet(in_channels=1, output_channel=64, num_classes=num_classes, stride=2).to(device), 
                    "projResnet"), 
                    (DilResnet(in_channels=1, output_channel=64, num_classes=num_classes, stride=2).to(device), 
                    "DilResnet"),   
                    (DilResTransformer(in_channels=1, output_channel=64, num_classes=num_classes, stride=2).to(device), 
                    "DilResTransformer"),      
                    (BasicTransformer(in_channels=1, output_channel=64, num_classes=num_classes, d_model=64, n_head=8, dim_feedforward=64, dropout=0.1, num_transformer_layers=3).to(device), 
                    "BasicTransformer"),        
                    (SignalTransformer(in_channels=1, output_channel=64, num_classes=num_classes, segment_length=16, d_model=64, n_head=4,  dim_feedforward=64,dropout=0.1,num_transformer_layers=3).to(device), 
                    "SignalTransformer"),                                                 
                ]
                
                results_save_path = os.path.join('loop_cls_results', data_type, 
                    f'rep_exp_results_{data_type}_{condition}_{test_condition}_n{noise_level}_{suffix}_{c_time}.csv')
                os.makedirs(os.path.dirname(results_save_path), exist_ok=True)
                # 각 모델별로 학습 진행
                for model, model_name in models_to_train:
                    print(f"\n{'='*50}")
                    print(f"Training {model_name} for {exp_config['data_type']}")
                    print(f"{'='*50}")
                    if model_name == "BasicTransformer":
                         common_params['epochs'] = 500
                         common_params['batch_size'] = 16
                    elif model_name == "SignalTransformer":
                         common_params['epochs'] = 300
                         common_params['batch_size'] = 64
                    for rep in range(common_params['mdl_rep_time']): # same dataset, repeat train
                        print(f"\nRepetition {rep + 1}/{common_params['mdl_rep_time']} in exp_rep{exp_rep+1}/{common_params['exp_rep_time']}")
                        
                        try:
                            # CUDA 캐시 메모리 초기화
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                import gc
                                gc.collect()
                                
                            trained_model, history, criterion, exp_name = train_model(model, model_name, label_to_idx, rep, exp_rep)
                            # 학습 세트에 대한 최종 평가 - 훈련
                            trained_model.eval()
                            train_total_loss = 0
                            train_correct = 0
                            train_total = 0
                            
                            with torch.no_grad():
                                for inputs, targets in train_loader:
                                    inputs = inputs.to(device)
                                    targets = targets.to(device)
                                    outputs, _ = trained_model(inputs)
                                    loss = criterion(outputs, targets)
                                    
                                    train_total_loss += loss.item() * inputs.size(0)
                                    _, predicted = outputs.max(1)
                                    train_correct += predicted.eq(targets).sum().item()
                                    train_total += targets.size(0)
                            
                            final_train_loss = train_total_loss / train_total
                            final_train_acc = train_correct / train_total

                            # 최종 평가 - test
                            total_test_loss = 0
                            total_correct = 0
                            total_samples = 0
                            all_preds = []
                            all_targets = []
                            
                            with torch.no_grad():
                                for inputs, targets in test_loader:
                                    inputs = inputs.to(device)
                                    targets = targets.to(device)
                                    outputs, _ = trained_model(inputs)
                                    loss = criterion(outputs, targets)
                                    
                                    total_test_loss += loss.item() * inputs.size(0)
                                    _, predicted = outputs.max(1)
                                    total_correct += predicted.eq(targets).sum().item()
                                    total_samples += targets.size(0)
                                    
                                    all_preds.extend(predicted.cpu().numpy())
                                    all_targets.extend(targets.cpu().numpy())
                            
                            final_test_loss = total_test_loss / total_samples
                            final_test_acc = total_correct / total_samples
                            
                            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                            # classification report 생성
                            class_names = [label for label, _ in sorted(label_to_idx.items(), key=lambda x: x[1])]
                            report = classification_report(
                                all_targets, 
                                all_preds, 
                                target_names=class_names, 
                                output_dict=True,
                                zero_division=0
                            )
                            # 각 클래스별 메트릭 추출
                            precision_per_class = [report[class_name]['precision'] for class_name in class_names]
                            recall_per_class = [report[class_name]['recall'] for class_name in class_names]
                            f1_per_class = [report[class_name]['f1-score'] for class_name in class_names]
                            support_per_class = [report[class_name]['support'] for class_name in class_names]

                            # DataFrame에 결과 추가
                            results_df = pd.concat([results_df, pd.DataFrame([{
                                'model_name': model_name,
                                'exp_rep_num': exp_rep,
                                'mdl_rep_num': rep,
                                'train_loss': final_train_loss,
                                'train_accuracy': final_train_acc * 100,
                                'test_loss': final_test_loss,
                                'test_accuracy': final_test_acc * 100,
                                'batch_size': batch_size,
                                'epochs': epochs,
                                'learning_rate': lr,
                                'window_size': window_size,
                                'overlap': overlap,
                                'data_type': data_type,
                                'condition': condition,
                                'test_condition': str(test_condition),
                                'label_column': label_column,
                                'dim_type': dim_type,
                                'noise_level': noise_level,
                                'log_time': current_time,
                                'class_names': class_names,
                                'precision_per_class': precision_per_class,
                                'recall_per_class': recall_per_class,
                                'f1_per_class': f1_per_class,
                                'support_per_class': support_per_class,
                                'macro_precision': report['macro avg']['precision'],
                                'macro_recall': report['macro avg']['recall'],
                                'macro_f1': report['macro avg']['f1-score'],
                                'weighted_precision': report['weighted avg']['precision'],
                                'weighted_recall': report['weighted avg']['recall'],
                                'weighted_f1': report['weighted avg']['f1-score']
                            }])], ignore_index=True)
                            results_df.to_csv(results_save_path, index=False)   
                        except Exception as e:
                            print(f"Error in repetition {rep} for {model_name}: {str(e)}")
                            continue
            except Exception as e:
                print(f"Error in experiment {exp_config['data_type']}: {str(e)}")
                continue

            print(f"\nCompleted experiment for {exp_config['data_type']}")                    
