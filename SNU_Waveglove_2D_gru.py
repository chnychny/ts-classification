import torch
import numpy as np
import pandas as pd
import sys, os
from pathlib import Path
project_root = Path(__file__).parent
sys.path.append(str(project_root))
print(project_root)
from sklearn.model_selection import train_test_split
from dataprovider.SNUWaveGlove import SNUWaveGloveDataProcessor, SNUWaveGloveDataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import itertools
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime

## start 
base_dir = "/gpfs/home1/chny1216/ts-classification/"
base_dir = str(project_root)
save_path = str(project_root)
dim_type='time_domain'

# SNU waveglove
label_column= 'label'# waveglove
data_type = "SNUWG_Rkeyboard"
condition = 'All_even'
seg_cut = 30

# 데이터 로드 /gpfs/home1/chny1216/data/SNU-WaveGlove/dataset_keyboard_250403/
data_path = "/gpfs/home1/chny1216/data/SNU-WaveGlove/dataset_keyboard_250403/"
data_path = "G:/My Drive/1. 연구/3. hand gesture recognition using IMU data/dataset_keyboard_250403/"
f_list = os.listdir(data_path)
local_dir = data_path
base_path = str(project_root)

# 데이터 처리
stacked_data, metadata_df = SNUWaveGloveDataProcessor.process_files_with_origin_time_domain(local_dir, f_list, data_type, seg_cut)
df = stacked_data.copy()
unit_data = {
    # 'unit': df['f_list'] + '-' + df['label'].astype(str) + '-' + df['seg_idx'].astype(str)
    'unit': df['f_list'] + '-' + df['label'].astype(str)
}
# 3. 한 번에 컬럼 추가
df = pd.concat([df, pd.DataFrame(unit_data)], axis=1)

def prepare_data(df, seq_length=10, anomaly_context=10):
    """
    데이터를 시퀀스로 변환하고 train/val/test로 분할합니다.
    anomaly 데이터의 전후 anomaly_context개 window는 test set에 포함됩니다.
    """
    df_expand = SNUWaveGloveDataProcessor.expand_sensor_data(df)
    # 센서 데이터 선택
    sensor_cols = [col for col in df_expand.columns if col.startswith(('raw_acc', 'raw_gyr', 'joint_pos', 'joint_angles'))]

   # 균형잡힌 데이터셋 생성
    balanced_train_data = []

    for idx in range(len(df_expand)):
        df_expand_unit = df_expand.iloc[idx]
        # Series에서 직접 센서 컬럼 선택
        x_temp = np.array([df_expand_unit[col] for col in sensor_cols])
        balanced_train_data.append(x_temp)
    
    # numpy 배열로 변환
    X = np.stack(balanced_train_data, axis=0)
    y = np.array(df_expand['anomaly_label'].values).astype(np.float32)
    
    print(f"데이터 shape - X: {X.shape}, y: {y.shape}")
    
    # 시퀀스 생성
    X_seq = create_sequences(X, seq_length)
    y_seq = create_sequences(y, seq_length)
    
    # anomaly 인덱스 찾기
    anomaly_indices = np.where(y == 1)[0]
    
    # anomaly 데이터를 validation과 test set에 적절히 분배
    test_anomaly, val_anomaly = train_test_split(anomaly_indices, test_size=0.3)
    
    # test_indices와 val_indices에 각각 anomaly와 그 주변 데이터 포함
    test_indices = set()
    val_indices = set()
    
    # test set anomaly 처리
    for idx in test_anomaly:
        start_idx = max(0, idx - anomaly_context)
        end_idx = min(len(X), idx + anomaly_context + 1)
        for i in range(start_idx, end_idx):
            if i + seq_length <= len(X):
                test_indices.add(i)
                
    # validation set anomaly 처리
    for idx in val_anomaly:
        start_idx = max(0, idx - anomaly_context)
        end_idx = min(len(X), idx + anomaly_context + 1)
        for i in range(start_idx, end_idx):
            if i + seq_length <= len(X):
                val_indices.add(i)
    
    # train/val/test 분할
    all_indices = set(range(len(X_seq)))
    test_indices = list(test_indices)
    remaining_indices = list(all_indices - set(test_indices))
    
    # remaining_indices를 train과 val로 분할
    train_indices, val_indices = train_test_split(
        remaining_indices, 
        test_size=0.1, 
        random_state=42
    )
    
    # 데이터 분할
    X_train = X_seq[train_indices]
    y_train = y_seq[train_indices]
    X_val = X_seq[val_indices]
    y_val = y_seq[val_indices]
    X_test = X_seq[test_indices]
    y_test = y_seq[test_indices]
    
    print(f"Train set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")

    print(f"Train set anomaly ratio: {np.mean(y_train[:, -1]):.4f}")
    print(f"Validation set anomaly ratio: {np.mean(y_val[:, -1]):.4f}")
    print(f"Test set anomaly ratio: {np.mean(y_test[:, -1]):.4f}")
    
    # 데이터 타입 확인
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
   
    
    # 데이터 형태 변환: [batch, seq_len, channel] -> [batch, channel, seq_len, 1]

    X_train_reshaped = X_train.transpose(0, 2, 1).reshape(-1, 178, X_train.shape[1], 1)
    X_val_reshaped = X_val.transpose(0, 2, 1).reshape(-1, 178, X_val.shape[1], 1)
    X_test_reshaped = X_test.transpose(0, 2, 1).reshape(-1, 178, X_test.shape[1], 1)    
    # Dataset 생성
    print(f"X_train shape: {X_train_reshaped.shape}")
 
    train_dataset = TensorDataset(torch.FloatTensor(X_train_reshaped), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val_reshaped), torch.FloatTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test_reshaped), torch.FloatTensor(y_test))
    
    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    return train_loader, val_loader, test_loader, sensor_cols

def train_epoch(model, train_loader, optimizer, device):
    """한 에포크의 학습을 수행합니다."""
    model.train()
    total_loss = 0
    
    for batch_x, batch_y in train_loader:
        # 정상 데이터만 선택
        normal_idx = batch_y[:, -1] == 0
        batch_x = batch_x[normal_idx].to(device)
        
        optimizer.zero_grad()
        reconstructed, encoded = model(batch_x)
        
        loss = model.loss_function(batch_x, reconstructed, encoded)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def evaluate(model, data_loader, device):
    """모델을 평가하고 loss와 anomaly score를 반환합니다."""
    model.eval()
    total_loss = 0
    scores = []
    labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            reconstructed, encoded = model(batch_x)
            batch_scores = model.get_anomaly_score(batch_x)
            
            scores.extend(batch_scores.cpu().numpy())
            labels.extend(batch_y[:, -1].cpu().numpy())
            total_loss += model.loss_function(batch_x, reconstructed, encoded).item()
    
    return total_loss / len(data_loader), scores, labels

def train_detector(model, train_loader, val_loader, test_loader, device, num_epochs=50):
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, min_lr=1e-6)
    
    best_val_loss = float('inf')
    best_model = None
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # 학습
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        # 검증
        model.eval()
        val_loss, val_scores, val_labels = evaluate(model, val_loader, device)
        
        # 현재 임계값으로 검증 성능 계산
        threshold = find_threshold(model, val_loader, device)
        val_predictions = (np.array(val_scores) > threshold).astype(float)
        val_precision = precision_score(val_labels, val_predictions)
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        print(f'Val Precision: {val_precision:.4f}')
        
        scheduler.step(val_loss)
        
        # 검증 손실과 precision을 모두 고려하여 모델 저장
        if val_loss < best_val_loss and val_precision > 0.5:  # precision 임계값 설정
            best_val_loss = val_loss
            best_model = model.state_dict()
            torch.save(best_model, 'best_anomaly_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping triggered after epoch {epoch+1}')
            break
    
    model.load_state_dict(best_model)
    return model

def find_threshold(model, val_loader, device):
    """최적의 anomaly detection 임계값을 찾습니다."""
    scores = []
    labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_scores = model.get_anomaly_score(batch_x)
            scores.extend(batch_scores.cpu().numpy())
            labels.extend(batch_y[:, -1].cpu().numpy())
    
    scores = np.array(scores)
    labels = np.array(labels)
    
    # 정상 데이터와 비정상 데이터의 reconstruction error 분포를 분석
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]
    
    # 정상 데이터의 평균과 표준편차를 계산
    mean_normal = np.mean(normal_scores)
    std_normal = np.std(normal_scores)
    
    # 임계값을 평균 + 2*표준편차로 설정
    threshold = mean_normal + 2 * std_normal
    
    return threshold

def inference(model, data_loader, threshold, device):
    """학습된 모델로 추론을 수행합니다."""
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            scores = model.get_anomaly_score(batch_x)
            pred = (scores > threshold).float()
            
            predictions.extend(pred.cpu().numpy())
            true_labels.extend(batch_y[:, -1].cpu().numpy())
    
    return predictions, true_labels

def create_sequences(data, seq_length=10):
    """
    시계열 데이터를 시퀀스로 변환합니다.
    
    Args:
        data (np.ndarray): 입력 데이터 (n_samples, n_features)
        seq_length (int): 시퀀스 길이
        
    Returns:
        np.ndarray: 시퀀스 데이터 (n_sequences, seq_length, n_features)
    """
    sequences = []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences)

class TensorDataset(Dataset):
    """
    이상 탐지를 위한 커스텀 Dataset 클래스
    """
    def __init__(self, X, y):
        """
        Args:
            X (np.ndarray): 입력 데이터 (n_sequences, seq_length, n_features)
            y (np.ndarray): 레이블 데이터 (n_sequences, seq_length)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def calculate_roc_auc(y_true, y_score):
    """
    ROC-AUC 점수를 계산합니다.
    
    Args:
        y_true (np.ndarray): 실제 레이블
        y_score (np.ndarray): 예측 점수
        
    Returns:
        float: ROC-AUC 점수
    """
    try:
        auc_score = roc_auc_score(y_true, y_score)
    except ValueError:
        # 모든 레이블이 같은 경우
        auc_score = 0.5
    return auc_score

def plot_roc_curve(y_true, y_score, save_path='roc_curve.png'):
    """
    ROC 곡선을 시각화합니다.
    
    Args:
        y_true (np.ndarray): 실제 레이블
        y_score (np.ndarray): 예측 점수
        save_path (str): 저장 경로
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_score = calculate_roc_auc(y_true, y_score)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_precision_recall_curve(y_true, y_score, save_path='pr_curve.png'):
    """
    Precision-Recall 곡선을 시각화합니다.
    
    Args:
        y_true (np.ndarray): 실제 레이블
        y_score (np.ndarray): 예측 점수
        save_path (str): 저장 경로
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    average_precision = average_precision_score(y_true, y_score)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR curve (AP = {average_precision:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_anomaly_scores(scores, labels, save_path='anomaly_scores.png'):
    """
    이상 점수를 시각화합니다.
    
    Args:
        scores (np.ndarray): 이상 점수
        labels (np.ndarray): 실제 레이블
        save_path (str): 저장 경로
    """
    plt.figure(figsize=(12, 6))
    
    # 이상 점수 플롯
    plt.subplot(2, 1, 1)
    plt.plot(scores, label='Anomaly Score')
    plt.title('Anomaly Scores')
    plt.xlabel('Time')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    # 실제 레이블 플롯
    plt.subplot(2, 1, 2)
    plt.plot(labels, label='True Label', color='red')
    plt.title('True Labels')
    plt.xlabel('Time')
    plt.ylabel('Label')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def evaluate_anomaly_detection(y_true, y_pred):
    """
    이상 탐지 성능을 평가합니다.
    
    Args:
        y_true (np.ndarray): 실제 레이블
        y_pred (np.ndarray): 예측 레이블
        
    Returns:
        dict: 평가 지표
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': calculate_roc_auc(y_true, y_pred)
    }
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm
    
    # Classification Report
    report = classification_report(y_true, y_pred, output_dict=True)
    metrics['classification_report'] = report
    
    return metrics

def print_evaluation_metrics(metrics, model, input_size, seq_len, save_path=None):
    """
    평가 지표를 출력하고 파일로 저장합니다.
    
    Args:
        metrics (dict): 평가 지표
        model: 모델 객체
        input_size: 입력 크기
        seq_len: 시퀀스 길이
        save_path: 저장 경로 (기본값: None)
    """
    # 현재 시간을 파일명에 포함
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 결과를 저장할 문자열 생성
    result_str = "=== Model Information ===\n"
    result_str += f"Model Type: {model.__class__.__name__}\n"
    result_str += f"Input Size: {input_size}\n"
    result_str += f"Sequence Length: {seq_len}\n"
    result_str += f"Number of Parameters: {sum(p.numel() for p in model.parameters())}\n"
    result_str += "\n=== Model Architecture ===\n"
    result_str += str(model) + "\n"
    
    result_str += "\n=== Evaluation Metrics ===\n"
    result_str += f"Accuracy: {metrics['accuracy']:.4f}\n"
    result_str += f"Precision: {metrics['precision']:.4f}\n"
    result_str += f"Recall: {metrics['recall']:.4f}\n"
    result_str += f"F1 Score: {metrics['f1']:.4f}\n"
    result_str += f"ROC-AUC: {metrics['roc_auc']:.4f}\n"
    
    result_str += "\n=== Confusion Matrix ===\n"
    result_str += str(metrics['confusion_matrix']) + "\n"
    
    result_str += "\n=== Classification Report ===\n"
    result_str += str(metrics['classification_report'])
    
    # 콘솔에 출력
    print(result_str)
    
    # 파일로 저장
    if save_path is None:
        save_path = str(project_root)
    
    file_name = f"evaluation_results_{current_time}.txt"
    file_path = os.path.join(save_path, "results", file_name)
    
    # results 디렉토리가 없으면 생성
    os.makedirs(os.path.join(save_path, "results"), exist_ok=True)
    
    with open(file_path, 'w') as f:
        f.write(result_str)
    
    print(f"\nResults saved to: {file_path}")

class AttentionLayer(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Linear(in_features // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: [batch, channels, seq_len, 1]
        b, c, s, _ = x.size()
        x_flat = x.squeeze(-1).transpose(1, 2)  # [batch, seq_len, channels]
        weights = self.attention(x_flat)  # [batch, seq_len, 1]
        weighted = x_flat * weights  # [batch, seq_len, channels]
        return weighted.transpose(1, 2).unsqueeze(-1)

class AttentionConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.attention = AttentionLayer(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), padding=(1, 0))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.attention(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class CNNAutoencoder(nn.Module):
    def __init__(self, input_size, seq_len=10):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_size, 128, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Conv2d(128, 64, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Conv2d(64, 32, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2)
        )
        
        # Global Attention
        self.attention = AttentionLayer(32)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.ConvTranspose2d(64, 128, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.ConvTranspose2d(128, input_size, kernel_size=(3, 1), padding=(1, 0))
        )
        
    def forward(self, x):
        # Encoder
        encoded = self.encoder(x)
        
        # Global Attention
        attended = self.attention(encoded)
        
        # Decoder
        decoded = self.decoder(attended)
        
        return decoded, encoded

    def get_anomaly_score(self, x):
        reconstructed, encoded = self(x)
        # MSE for each sample
        reconstruction_error = torch.mean((x - reconstructed) ** 2, dim=(1, 2))
        return reconstruction_error

    def loss_function(self, x, decoded, encoded, beta=0.2, lambda_reg=0.001):
        # Reconstruction Loss (MSE)
        recon_loss = F.mse_loss(decoded, x)
        
        # Feature Compactness Loss
        feature_loss = torch.mean(torch.norm(encoded, dim=1))
        
        # L2 Regularization
        l2_reg = 0
        for param in self.parameters():
            l2_reg += torch.norm(param)
        
        # Combined Loss with adjusted weights
        total_loss = recon_loss + beta * feature_loss + lambda_reg * l2_reg
        return total_loss

def main():
    """메인 실행 함수"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 데이터 준비
    train_loader, val_loader, test_loader, sensor_cols = prepare_data(df, seq_length=10, anomaly_context=30)
    
    # 모델 초기화
    input_size = len(sensor_cols)
    seq_len = 10
    model = CNNAutoencoder(input_size=input_size, seq_len=seq_len).to(device)
    
    # 학습
    model = train_detector(model, train_loader, val_loader, test_loader, device)
    
    # 임계값 찾기
    threshold = find_threshold(model, val_loader, device)
    print(threshold)
    # 최종 테스트
    predictions, true_labels = inference(model, test_loader, threshold, device)
    
    # 성능 평가
    metrics = evaluate_anomaly_detection(true_labels, predictions)
    print_evaluation_metrics(metrics, model, input_size, seq_len)
    
    # 시각화
    plot_roc_curve(true_labels, predictions)
    plot_precision_recall_curve(true_labels, predictions)
    plot_anomaly_scores(predictions, true_labels)

if __name__ == "__main__":
    main()