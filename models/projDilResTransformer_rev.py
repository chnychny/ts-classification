import torch
import torch.nn as nn
import torch.nn.functional as F

class CompactMultiHeadAttention(nn.Module):
    def __init__(self, channels, num_heads=2, dropout=0.1):
        super().__init__()
        assert channels % num_heads == 0
        
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 차원을 1/4로 줄여서 효율적인 attention 계산
        self.reduced_dim = channels // 4
        self.to_reduced = nn.Conv1d(channels, self.reduced_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(self.reduced_dim, channels, 1, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, C, L = x.shape
        H = self.num_heads
        
        # 원본 feature 저장
        residual = x
        
        # Reduced dimension에서 QKV 계산
        qkv = self.to_reduced(x)
        qkv = qkv.reshape(B, 3, H, self.reduced_dim // H, L)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Attention 결과를 원래 차원으로 복원
        out = (attn @ v).reshape(B, self.reduced_dim, L)
        out = self.to_out(out)
        
        # Skip connection으로 원본 feature 보존
        return out + residual
class MultiDilatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2):
        super(MultiDilatedConv, self).__init__()
        
        self.num_dilations = 6
        channels_per_dilation = out_channels // self.num_dilations
        
        def get_padding(kernel_size, dilation):
            return ((kernel_size - 1) * dilation) // 2
        
        self.dilated_convs = nn.ModuleList([
            nn.Conv1d(
                in_channels,
                channels_per_dilation,
                kernel_size=kernel_size,
                stride=stride,
                padding=get_padding(kernel_size, d),
                dilation=d,
                bias=False
            ) for d in [1, 3, 5, 7, 9, 11]
        ])
        
        self.final_conv = nn.Conv1d(channels_per_dilation * self.num_dilations, 
                                  out_channels, 
                                  kernel_size=1,
                                  bias=False)
    
    def forward(self, x):
        dilated_outputs = [conv(x) for conv in self.dilated_convs]
        out = torch.cat(dilated_outputs, dim=1)
        out = self.final_conv(out)
        return out

class ImprovedDilResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        
        # Main branch
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = MultiDilatedConv(in_channels, out_channels, kernel_size=5, 
                             stride=1, padding=2)
        
        # Compact attention module
        self.attention = CompactMultiHeadAttention(out_channels, num_heads=2)
        
        # Second convolution with skip connection
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, 
                             stride=2, padding=1)
        
        # Shortcut connection
        if stride == 2:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x):
        # Main path
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        
        # Attention with skip connection
        out = self.attention(out)
        
        # Second convolution
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        
        # Add shortcut
        return out + self.shortcut(x)
class ImprovedProjDilResTransformer2(nn.Module):
    """
    제안모델 
    proj-conv1_bn1_relu_maxpool-Dilconv1_bn1_Conv1-
    ImprovedResidualBlock1(bn-relu-Dil-CMTHA-bn-conv1)-2-Adaptive pooling-Linear
    """
    def __init__(self, in_channels, output_channel, num_classes, stride=2):
        super().__init__()
        
        # Initial projection
        self.proj_conv = nn.Conv1d(in_channels, (output_channel//2), kernel_size=5, stride=1, padding=2, bias=False)
        self.conv1 = nn.Conv1d((output_channel//2), (output_channel//2), kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d((output_channel//2))
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        # DilResidualBlock1 직접적용 self.res_block1=DilResidualBlock((output_channel//2),(output_channel//2), stride=2)
        self.dil_conv1 = MultiDilatedConv((output_channel//2), (output_channel//2), kernel_size=5, 
                             stride=1, padding=2)
        self.dil_bn1 = nn.BatchNorm1d((output_channel//2))
        self.dil_conv2 = nn.Conv1d((output_channel//2), (output_channel//2), kernel_size=3, stride=2, padding=1)     
        self.down_sample = nn.Sequential(nn.Conv1d((output_channel//2), (output_channel//2), kernel_size=1, stride=2),
        nn.BatchNorm1d((output_channel//2)))

        # Improved residual blocks
        self.res_block1 = ImprovedDilResidualBlock((output_channel//2), output_channel, stride=2)
        self.res_block2 = ImprovedDilResidualBlock(output_channel, output_channel, stride=2)

        # Simple classifier
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(output_channel, num_classes)
        
    def forward(self, x):
        # Initial projection
        out=self.proj_conv(x)
        out = self.conv1(out)
        out=self.bn1(out)
        out=self.relu(out)
        out=self.maxpool(out)
        identity = out
        out=self.dil_conv1(out)
        out=self.dil_bn1(out)
        out=self.relu(out)
        out=self.dil_conv2(out)
        shortcut = self.down_sample(identity) # stride2니까 사용
        out += shortcut
        
        # Residual blocks
        out = self.res_block1(out)
        out = self.res_block2(out)

        
        # Global pooling and classification
        features = self.gap(out).squeeze(-1)
        out = self.classifier(features)
        
        return out, features
    
class DilResTransformer(nn.Module):
    def __init__(self, in_channels, output_channel,num_classes,stride=2):
        super(DilResTransformer, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, (output_channel//2), kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d((output_channel//2))
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        # DilResidualBlock1 직접적용 self.res_block1=DilResidualBlock((output_channel//2),(output_channel//2), stride=2)
        self.dil_conv1 = MultiDilatedConv((output_channel//2), (output_channel//2), kernel_size=5, 
                             stride=1, padding=2)
        self.dil_bn1 = nn.BatchNorm1d((output_channel//2))
        self.dil_conv2 = nn.Conv1d((output_channel//2), (output_channel//2), kernel_size=3, stride=2, padding=1)     
        self.down_sample = nn.Sequential(nn.Conv1d((output_channel//2), (output_channel//2), kernel_size=1, stride=2),
        nn.BatchNorm1d((output_channel//2)))

        # Improved residual blocks
        self.res_block1 = ImprovedDilResidualBlock((output_channel//2), output_channel, stride=2)
        self.res_block2 = ImprovedDilResidualBlock(output_channel, output_channel, stride=2)

        # Simple classifier
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(output_channel, num_classes)
        
    def forward(self, x):
        out = self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)
        out=self.maxpool(out)
        identity = out
        out=self.dil_conv1(out)
        out=self.dil_bn1(out)
        out=self.relu(out)
        out=self.dil_conv2(out)
        shortcut = self.down_sample(identity) # stride2니까 사용
        out += shortcut
        
        # Residual blocks
        out = self.res_block1(out)
        out = self.res_block2(out)        
        # Global pooling and classification
        features = self.gap(out).squeeze(-1)
        out = self.classifier(features)
        
        return out, features        