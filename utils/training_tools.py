import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
from tqdm import tqdm

class TrainingHistory:
    def __init__(self):
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.confusion_matrices = []
        self.class_names = None
    
    def add_confusion_matrix(self, y_true, y_pred, class_names=None):
        cm = confusion_matrix(y_true, y_pred)
        self.confusion_matrices.append(cm)
        if class_names is not None:
            self.class_names = class_names
    
    def plot_metrics(self, exp_name,save_path=None):
        """Loss와 Accuracy 그래프 그리기"""
        plt.figure(figsize=(12, 4))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.title('Loss History',wrap=True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accs, label='Train Acc')
        plt.plot(self.val_accs, label='Val Acc')
        plt.title('Accuracy History')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.suptitle(exp_name,wrap=True)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()
    def plot_confusion_matrix(self, exp_name, save_path=None):
        """Confusion Matrix 그리기"""
        if not self.confusion_matrices:
            print("No confusion matrix data available")
            return
        
        plt.figure(figsize=(10, 8))
        cm = self.confusion_matrices[-1]  # 마지막 confusion matrix
        
        # confusion matrix 정규화
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # class_names가 없는 경우 기본값 사용
        if self.class_names is None:
            self.class_names = [str(i) for i in range(cm.shape[0])]
            
        # heatmap 생성
        sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues',
                xticklabels=self.class_names,
                yticklabels=self.class_names)
        plt.title('Confusion Matrix'+exp_name,wrap=True)
        
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()
   
def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, accumulation_steps=4, grad_step_update=False):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    optimizer.zero_grad()  # epoch 시작시 한번만 zero_grad
    progress_bar = tqdm(train_loader, desc='Training')
    
    for i, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        outputs, _ = model(inputs)
        loss = criterion(outputs, targets)
        if grad_step_update:
            loss = loss/accumulation_steps
        loss.backward()
        
        # 그래디언트 누적 후 업데이트
        if (i + 1) % accumulation_steps == 0:
            if grad_step_update:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step() # batch 스텝 안나누고 누적한거 한번데
            optimizer.zero_grad()
        
        # Statistics
        running_loss += loss.item() * accumulation_steps  # Scale loss back
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        progress_bar.set_postfix({
            'Loss': f'{running_loss/(i+1):.3f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    # Handle remaining gradients
    if (i + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    
    # if not grad_step_update:
    #     scheduler.step()  # epoch 단위로 scheduler step
        
    return epoch_loss, epoch_acc

@torch.no_grad()  
def evaluate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    
    for inputs, labels in test_loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        outputs, _ = model(inputs)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(test_loader), correct / total


def get_model_size(model):
    """모델의 파라미터 수와 크기 계산"""
    # 총 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 모델 크기 계산 (MB 단위)
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': size_all_mb
    }

