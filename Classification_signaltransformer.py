import torch
import numpy as np
import sys, os
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report
from dataprovider.base_preprocess import BaseTimeDataset
from utils.training_tools import TrainingHistory
from utils.training_tools import evaluate
from utils.training_tools import train_epoch, get_model_size
from utils.utils import create_experiment_dirs  #
from models.att_resnet import projDil22Resnet3, projDil2ConvResnet3, projDil3Resnet3, projDilResnet3
from models.signal_transformers import SignalTransformer
from models.projDilResTransformer_rev import ImprovedProjDilResTransformer2
from utils.training_tools_revised import train_epoch_revised
from dataprovider.data_import_combine import data_import_combine
from datetime import datetime
from sklearn.model_selection import train_test_split

def train_model(label_to_idx):

    model = SignalTransformer(in_channels=1, output_channel=64, num_classes=num_classes, segment_length=16, d_model=64, n_head=4,  dim_feedforward=64,dropout=0.1,num_transformer_layers=3).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.01
    )
    
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99) 
    cycle_update = False
    # scheduler = optim.lr_scheduler.OneCycleLR(
    # optimizer,
    # max_lr=1e-3,
    # epochs=epochs,
    # steps_per_epoch=len(train_loader),
    # pct_start=0.3,  # warm-up 기간
    # div_factor=25,  # 초기 lr = max_lr/25
    # final_div_factor=1e4  # 최종 lr = 초기 lr/1e4
    # )   
    # cycle_update = True
    

    criterion = nn.CrossEntropyLoss()   
    class_names = [label for label, _ in sorted(label_to_idx.items(), key=lambda x: x[1])]
    
    # 모델 정보 저장
    model_info = get_model_size(model)
    print("\nModel Information:")
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
        # train_loss, train_acc = train_epoch(
        #     model, train_loader, criterion, optimizer, scheduler, 
        #     device, accumulation_steps, grad_step_update=False
        # )
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
        # scheduler.step(val_loss)
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
    
    return model, history, criterion
# 메인 실행 코드
if __name__ == "__main__":   
    # UOS 데이터 로드 및 분할
    base_dir ="C:/Users/USER/pch/DataSet/"
    save_path = str(project_root) + "/data/"
    label_column='bearing_condition'


    # data_type = "PHM_Gearbox_spur" 
    # condition='All_load_even' # laad (diff_high - high로 테스트)
    # test_condition=[50] # speed select
    # noise_level=0


    data_type = "KAIST_Bearing"
    condition='max_even'
    test_condition=[0,2,4]
    noise_level=0

    # data_type = "UOS_RTES_deep_groove_ball_bearing"
    # condition = "Compound_fault_total"
    # test_condition = [800]
    # noise_level = 0


    window_size = 1024
    overlap = 0
    
    suffix = f'ts_ws{window_size}_ov{overlap}'

    dim_type='time_domain'
    
    # 학습 파라미터 설정
    batch_size = 16
    epochs = 500
    lr = 1e-3#4e-4
    exp_name = f'{data_type}-{condition}-{str(test_condition)}-noise{noise_level}-SignalTransformer-epoch{epochs}-explr-lr{lr}-bsize{batch_size}-{suffix}'
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
    X_train_split, X_valid, y_train_split, y_valid = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
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
    result_dirs = create_experiment_dirs(data_type)

    # 모델 학습
    trained_model, history, criterion = train_model(label_to_idx)
    
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    final_results_path = os.path.join('results/'+data_type, f'final_evaluation_{exp_name}_{current_time}.txt')

    with open(final_results_path, 'w') as f:
        # 전체 테스트 셋에 대한 평가
        trained_model.eval()
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
        
        # 결과 저장
        f.write(f'Exp name: {exp_name}\n\n')
        # 하이퍼 파라미터 정보 저장
        f.write('Hyperparameters:\n')
        f.write(f'Batch Size: {batch_size}\n')
        f.write(f'Epochs: {epochs}\n')
        f.write(f'Learning Rate: {lr}\n')
        f.write(f'Window Size: {window_size}\n')
        f.write(f'Overlap: {overlap}\n')
        f.write(f'Data Type: {data_type}\n')
        f.write(f'Condition: {condition}\n')
        f.write(f'Test Condition: {test_condition}\n')
        f.write(f'Label Column: {label_column}\n')
        f.write(f'Dim Type: {dim_type}\n')
        f.write(f'Log Time: {current_time}\n')
        f.write('\n')
        
        # 최종 테스트 결과 저장
        f.write(f'Final Test Loss: {final_test_loss:.4f}\n')
        f.write(f'Final Test Accuracy: {final_test_acc*100:.2f}%\n\n')
        
        # 분류 리포트 저장
        f.write('Classification Report:\n')
        f.write(classification_report(all_targets, all_preds, zero_division=0))