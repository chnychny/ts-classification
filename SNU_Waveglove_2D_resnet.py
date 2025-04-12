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
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import itertools

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
df = metadata_df.copy()
unit_data = {
    'unit': df['f_list'] + '-' + df['label'].astype(str) + '-' + df['seg_idx'].astype(str)
}
# 3. 한 번에 컬럼 추가
df = pd.concat([df, pd.DataFrame(unit_data)], axis=1)

# 각 레이블별로 20%를 테스트 세트로 분할
train_units = []
test_units = []

np.random.seed(42)  # 재현성을 위한 시드 설정
# 레이블별로 분할
for label in df['label'].unique():
    label_units = df[df['label'] == label]['unit'].unique()
    n_test = int(len(label_units) * 0.2)  # 20%를 테스트 세트로
    
    # 무작위로 테스트 세트 선택
    
    test_units_for_label = np.random.choice(label_units, size=n_test, replace=False)
    train_units_for_label = np.array([u for u in label_units if u not in test_units_for_label])
    
    train_units.extend(train_units_for_label)
    test_units.extend(test_units_for_label)
# train 데이터 준비
train_units_by_label = {}
# 레이블별로 unit 분류
for unit in train_units:
    label = df[df['unit'] == unit]['label'].unique()[0]
    if label not in train_units_by_label:
        train_units_by_label[label] = []
    train_units_by_label[label].append(unit)

# 각 레이블별 최소 샘플 수 찾기
min_samples = min(len(units) for units in train_units_by_label.values())
print(f"\nTrain set - Samples per label before balancing:")
for label, units in train_units_by_label.items():
    print(f"Label {label}: {len(units)} samples")
print(f"Minimum samples per label: {min_samples}")

# 균형잡힌 데이터셋 생성
balanced_train_data = []
balanced_train_labels = []

for label, units in train_units_by_label.items():
    # 각 레이블에서 무작위로 min_samples만큼 선택
    selected_units = np.random.choice(units, min_samples, replace=False)
    
    for unit in selected_units:
        df_unit = df[df['unit'] == unit].copy()
        df_unit = df_unit.sort_values('idx')
        x_temp = df_unit[[col for col in df_unit.columns if col.startswith(('raw_acc', 'raw_gyr', 'joint_pos', 'joint_angles'))]].to_numpy()
        balanced_train_data.append(x_temp)
        balanced_train_labels.append(label)

# test 데이터 준비
test_units_by_label = {}
for unit in test_units:
    label = df[df['unit'] == unit]['label'].unique()[0]
    if label not in test_units_by_label:
        test_units_by_label[label] = []
    test_units_by_label[label].append(unit)

min_test_samples = min(len(units) for units in test_units_by_label.values())
print(f"\nTest set - Samples per label before balancing:")
for label, units in test_units_by_label.items():
    print(f"Label {label}: {len(units)} samples")
print(f"Minimum test samples per label: {min_test_samples}")

balanced_test_data = []
balanced_test_labels = []

for label, units in test_units_by_label.items():
    selected_units = np.random.choice(units, min_test_samples, replace=False)
    
    for unit in selected_units:
        df_unit = df[df['unit'] == unit].copy()
        df_unit = df_unit.sort_values('idx')
        x_temp = df_unit[[col for col in df_unit.columns if col.startswith(('raw_acc', 'raw_gyr', 'joint_pos', 'joint_angles'))]].to_numpy()
        balanced_test_data.append(x_temp)
        balanced_test_labels.append(label)

# 레이블 매핑 정의
label_mapping = {
    'y': 0,
    'u': 1,
    'i': 2,
    'o': 3,
    'p': 4,
    ';': 5,
    'l': 6,
    'k': 7,
    'j': 8,
    'h': 9,
    'n': 10,
    'm': 11,
    ',': 12,
    '.': 13,
    '/': 14
}

# numpy array로 변환 및 레이블 매핑 적용
X_train = np.stack(balanced_train_data, axis=0)
y_train = np.array([label_mapping[label] for label in balanced_train_labels])
X_test = np.stack(balanced_test_data, axis=0)
y_test = np.array([label_mapping[label] for label in balanced_test_labels])

print("\n최종 데이터셋 형태:")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# 레이블 분포 확인
print("\n학습 데이터 레이블 분포:")
for label in np.unique(y_train):
    print(f"Label {label}: {np.sum(y_train == label)}")

print("\n테스트 데이터 레이블 분포:")
for label in np.unique(y_test):
    print(f"Label {label}: {np.sum(y_test == label)}")

# 데이터 reshape (시간 축 평균)
X_train_mean = X_train.mean(axis=1)
X_test_mean = X_test.mean(axis=1)
# data 선택
sensor_types = {
    'Accelerometer': (0, 33),    # 0-32 (33개 채널)
    'Gyroscope': (33, 66),      # 33-65 (33개 채널)
    'Joint Position': (66, 114), # 66-113 (48개 채널)
    'Joint Angles': (114, 178)   # 114-177 (64개 채널)
}

# # Accelerometer 데이터만 선택 (시간 축 평균)
# acc_start, acc_end = sensor_types['Accelerometer']
# X_train_mean = X_train.mean(axis=1)[:, acc_start:acc_end]  # Accelerometer 채널만 선택
# X_test_mean = X_test.mean(axis=1)[:, acc_start:acc_end]    # Accelerometer 채널만 선택

# Random Forest 모델 학습
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_mean, y_train)

# 모델 평가
train_score = rf.score(X_train_mean, y_train)
test_score = rf.score(X_test_mean, y_test)

print("\n모델 성능:")
print(f"Train accuracy: {train_score:.4f}")
print(f"Test accuracy: {test_score:.4f}")

class ResNet2D(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(ResNet2D, self).__init__()
        
        # 첫 번째 컨볼루션 레이어
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        
        # ResNet 블록들
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=(2, 1))
        self.layer3 = self._make_layer(128, 256, 2, stride=(2, 1))
        self.layer4 = self._make_layer(256, 512, 2, stride=(2, 1))
        
        # 평균 풀링과 완전 연결 레이어
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=(2, 1)):
        layers = []
        layers.append(ResBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(2, 1)):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), stride=stride, padding=(1, 0))
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != (1, 1) or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(identity)
        out = self.relu(out)
        
        return out

class WaveGloveDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f"{base_path}/combined_figure/confusion_matrix_{title}.png")
    plt.close()

def train_model(model, train_loader, test_loader, device, num_epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 입력 데이터 형태 확인
    for inputs, _ in train_loader:
        input_shape = inputs.shape
        break    
    # 학습 과정 기록
    train_losses = []
    train_accs = []
    test_accs = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss/len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        
        # 테스트 정확도 계산
        model.eval()
        test_correct = 0
        test_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        test_acc = 100. * test_correct / test_total
        test_accs.append(test_acc)
        print(f'Test Accuracy: {test_acc:.2f}%')
        
    # Confusion Matrix 계산 및 저장
    cm = confusion_matrix(all_labels, all_preds)
    mapped_classes = sorted(list(set(label_mapping.values())))  # 중복 제거 및 정렬
    plot_confusion_matrix(cm, classes=mapped_classes, 
                        title=f'confusion_matrix_epoch_{epoch+1}',
                        normalize=False)  # normalize 추가
    
    # 학습 과정 시각화
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{base_path}/combined_figure/training_process_{model_type}.png")
    plt.close()
    
    # 최종 분류 보고서 출력
    print("\nClassification Report:")
    # classification_report 사용 시
    report = classification_report(all_labels, all_preds,
                                 labels=mapped_classes,
                                 zero_division=0)
    print(report)
    report_path = f"{base_path}/combined_figure/classification_report_{model_type}_epoch_{num_epochs}.txt"
    with open(report_path, 'w') as f:
        f.write(f"Classification Report (Epochs: {num_epochs})\n")
        f.write("=" * 50 + "\n\n")
        f.write(report)
        f.write("\n\nModel Configuration:\n")
        f.write(f"Model Type: {model_type}\n")
        f.write(f"Number of Classes: {len(mapped_classes)}\n")
        f.write(f"Input Shape: {input_shape}\n")       
    return model

def train_and_evaluate(model_type, X_train, y_train, X_test, y_test, num_epochs=10):
    if model_type == 'random_forest':
        # Random Forest 모델 학습
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train.mean(axis=1), y_train)
        
        # 모델 평가
        train_score = rf.score(X_train.mean(axis=1), y_train)
        test_score = rf.score(X_test.mean(axis=1), y_test)
        
        print("\nRandom Forest 모델 성능:")
        print(f"Train accuracy: {train_score:.4f}")
        print(f"Test accuracy: {test_score:.4f}")
        return rf
    
    elif model_type == 'resnet':
        # 데이터 전처리 및 로더 생성
                # (batch, seg_cut*2+1, 178) -> (batch, 1, seg_cut*2+1, 178)
        # X_train_reshaped = X_train.reshape(-1, 1, X_train.shape[1], X_train.shape[2])
        # X_test_reshaped = X_test.reshape(-1, 1, X_test.shape[1], X_test.shape[2])
        # (batch, seg_cut*2+1, 178) -> (batch, 178, seg_cut*2+1, 1)
        X_train_reshaped = X_train.transpose(0, 2, 1).reshape(-1, 178, X_train.shape[1], 1)
        X_test_reshaped = X_test.transpose(0, 2, 1).reshape(-1, 178, X_test.shape[1], 1)
        
        print(f"\nResNet 입력 데이터 형태:")
        print(f"X_train_reshaped shape: {X_train_reshaped.shape}")
        print(f"X_test_reshaped shape: {X_test_reshaped.shape}")
        
        train_dataset = WaveGloveDataset(X_train_reshaped, y_train)
        test_dataset = WaveGloveDataset(X_test_reshaped, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
        
        # 모델 초기화 및 학습
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = ResNet2D(input_channels=1, num_classes=len(np.unique(y_train))).to(device)
        model = ResNet2D(input_channels=178, num_classes=len(np.unique(y_train))).to(device)
        model = train_model(model, train_loader, test_loader, device, num_epochs=num_epochs)
        return model
    
    else:
        raise ValueError("지원하지 않는 모델 타입입니다. 'random_forest' 또는 'resnet'을 선택해주세요.")


# 모델 선택 및 실행
model_type = 'resnet'  # 'random_forest' 또는 'resnet' 선택
num_epochs = 101  # 원하는 epoch 수 설정
model = train_and_evaluate(model_type, X_train, y_train, X_test, y_test, num_epochs=num_epochs)
# random_forest면 X_train_mean, y_train, X_test_mean, y_test 그대로 사용가능
# resnet면 X_train_mean대신 X_train