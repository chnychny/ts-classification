import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels,out_channels, stride=2):
        super(ResidualBlock, self).__init__()

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5, 
                             stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.stride = stride
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, 
                             stride=2, padding=1) 
        if stride == 2:
            self.down_sample = nn.Conv1d(in_channels, out_channels, kernel_size=1,  stride=2)

        else:
            self.down_sample = None
            

    def forward(self, x):
 

        shortcut = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.down_sample is not None:
            shortcut = self.down_sample(x)        
        
        
        out += shortcut
        return out
class projResnet(nn.Module):
    def __init__(self, in_channels, output_channel,num_classes,stride=2):
        super(projResnet, self).__init__()
    
        self.proj_conv = nn.Conv1d(in_channels, (output_channel//2), kernel_size=5, stride=1, padding=2, bias=False)
        self.conv1 = nn.Conv1d((output_channel//2), (output_channel//2), kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d((output_channel//2))
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
                
        self.res_block1=TypicalResidualBlock((output_channel//2),(output_channel//2), stride=2) # Residual Block대신 TypicalResidualBlock 사용
        self.res_block2=TypicalResidualBlock((output_channel//2),output_channel, stride=2)
        self.res_block3=TypicalResidualBlock(output_channel,output_channel, stride=2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.linear=nn.Linear(output_channel,num_classes)
    def forward(self,x):
        out=self.proj_conv(x)

        out =self.conv1(out)
        out=self.bn1(out)
        out=self.relu(out)
        out=self.maxpool(out)

        out=self.res_block1(out)
        out=self.res_block2(out)
        out=self.res_block3(out)
        
        x4=self.gap(out)
        x4=x4.squeeze(-1)
        out=self.linear(x4)
        return out,x4
    
class TypicalResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(TypicalResidualBlock, self).__init__()
        # First convolution block
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                            stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Second convolution block
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, 
                            stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Shortcut connection (identity mapping)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, 
                        stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
            
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out

class baseResnet(nn.Module):
    def __init__(self, in_channels, output_channel, num_classes):
        super(baseResnet, self).__init__()
        # Initial convolution layer
        self.conv1 = nn.Conv1d(in_channels, (output_channel//2), kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d((output_channel//2))
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # First ResNet module 
        self.layer1_block1 = TypicalResidualBlock((output_channel//2),(output_channel//2), stride=2)
        
        # Second ResNet module 
        self.layer2_block1 = TypicalResidualBlock((output_channel//2),output_channel, stride=2)
        
        # Third ResNet module (128 -> 256 channels)
        self.layer3_block1 = TypicalResidualBlock(output_channel,output_channel, stride=2)

        
        # Global Average Pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(output_channel, num_classes)
    
    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # First ResNet module
        x = self.layer1_block1(x)
        
        # Second ResNet module
        x = self.layer2_block1(x)
        
        # Third ResNet module
        x = self.layer3_block1(x)
        
        # Global average pooling and classification
        x_features = self.avgpool(x)
        x_features = x_features.squeeze(-1)
        out = self.fc(x_features)
        
        return out, x_features