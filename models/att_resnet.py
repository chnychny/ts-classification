import torch.nn as nn
import torch

class MultiDilatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2):
        super(MultiDilatedConv, self).__init__()
        
        # out_channels를 dilation의 개수로 나누어 각 dilated conv의 출력 채널 수 결정
        self.num_dilations = 6  # dilation rates: 1, 3, 5, 7, 9, 11
        channels_per_dilation = out_channels // self.num_dilations
        
        # 각 dilation rate에 대한 padding 계산 함수
        def get_padding(kernel_size, dilation):
            return ((kernel_size - 1) * dilation) // 2
        
        # 여러 dilation rate를 가진 컨볼루션 레이어들
        self.dilated_convs = nn.ModuleList([
            nn.Conv1d(
                in_channels,
                channels_per_dilation,
                kernel_size=kernel_size,
                stride=stride,
                padding=get_padding(kernel_size, d),  # 수정된 padding 계산
                dilation=d,
                bias=False
            ) for d in [1, 3, 5, 7, 9, 11]
        ])
        
        # 마지막 채널 수를 맞추기 위한 1x1 convolution
        self.final_conv = nn.Conv1d(channels_per_dilation * self.num_dilations, 
                                  out_channels, 
                                  kernel_size=1,
                                  bias=False)  # bias False로 설정
    
    def forward(self, x):
        # 각 dilated convolution 적용
        dilated_outputs = [conv(x) for conv in self.dilated_convs]
        
        # 채널 방향으로 concatenate
        out = torch.cat(dilated_outputs, dim=1)
        
        # 최종 채널 수 조정
        out = self.final_conv(out)
        return out
class Dil3ResidualBlock(nn.Module): 
    """
    bn1-relu-Dil-Dil-bn-Dil
    """
    def __init__(self, in_channels,out_channels, stride=2):
        super(Dil3ResidualBlock, self).__init__()

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = MultiDilatedConv(in_channels, out_channels, kernel_size=5, 
                             stride=1, padding=2)
        self.conv2 = MultiDilatedConv(out_channels, out_channels, kernel_size=5, 
                             stride=1, padding=2)             
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.stride = stride
        self.conv3 = MultiDilatedConv(out_channels, out_channels, kernel_size=3, 
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
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.down_sample is not None:
            shortcut = self.down_sample(x)        
        
        out += shortcut
        return out        
class Dil2ResidualBlock(nn.Module): 
    """
    (MPARN)conv1대신 mdc(multi-dilated conv)를 쓰고 dilation = 1,3,5,7,9,11
    bn2전에 PAM을 추가한게 MPARN (PAM: 채널별, 피처별 로 avgpool, maxpool 두번씩 해서 더하고 activation함수 써서 0~1로 값 맞춰줌)
    -> dilated conv 로 변경
    """
    def __init__(self, in_channels,out_channels, stride=2):
        super(Dil2ResidualBlock, self).__init__()

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = MultiDilatedConv(in_channels, out_channels, kernel_size=5, 
                             stride=1, padding=2)
             
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.stride = stride
        self.conv2 = MultiDilatedConv(out_channels, out_channels, kernel_size=3, 
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
class Dil2ConvResidualBlock(nn.Module): 
    """
    (bn1_relu_Dil-Dil-bn-conv1)
    """
    def __init__(self, in_channels,out_channels, stride=2):
        super(Dil2ConvResidualBlock, self).__init__()

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = MultiDilatedConv(in_channels, out_channels, kernel_size=5, 
                             stride=1, padding=2)
        self.conv2 = MultiDilatedConv(out_channels, out_channels, kernel_size=5, 
                             stride=1, padding=2)        
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.stride = stride

        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=3, 
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
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.down_sample is not None:
            shortcut = self.down_sample(x)        
        
        
        out += shortcut
        return out     
class DilConv2ResidualBlock(nn.Module): 
    """
    (bn1_relu_Dil-conv1-bn-conv1)
    """
    def __init__(self, in_channels,out_channels, stride=2):
        super(DilConv2ResidualBlock, self).__init__()

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = MultiDilatedConv(in_channels, out_channels, kernel_size=5, 
                             stride=1, padding=2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, 
                             stride=1, padding=2)                
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.stride = stride

        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=3, 
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
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.down_sample is not None:
            shortcut = self.down_sample(x)        
        
        
        out += shortcut
        return out     
class DilResidualBlock(nn.Module): 
    """
    (MPARN)conv1대신 mdc(multi-dilated conv)를 쓰고 dilation = 1,3,5,7,9,11
    bn2전에 PAM을 추가한게 MPARN (PAM: 채널별, 피처별 로 avgpool, maxpool 두번씩 해서 더하고 activation함수 써서 0~1로 값 맞춰줌)
    -> dilated conv 로 변경
    """
    def __init__(self, in_channels,out_channels, stride=2):
        super(DilResidualBlock, self).__init__()

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = MultiDilatedConv(in_channels, out_channels, kernel_size=5, 
                             stride=1, padding=2)
        
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
class projDil3Resnet3(nn.Module):
    """
    모듈 3개, (dil-dil-dil) 
    [proj-conv1_bn1_relu_maxpool-Dilconv1_bn1_Dilconv2-Dil2Resblock1(bn1_relu_Dil-Dil-bn-Dil)-2-GAP-Linear]
    """
    def __init__(self, in_channels, output_channel,num_classes,stride=2):
        super(projDil3Resnet3, self).__init__()
                        
        self.proj_conv = nn.Conv1d(in_channels, (output_channel//2), kernel_size=5, stride=1, padding=2, bias=False)
        self.conv1 = nn.Conv1d((output_channel//2), (output_channel//2), kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d((output_channel//2))
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        # DilResidualBlock1 직접적용 self.res_block1=DilResidualBlock((output_channel//2),(output_channel//2), stride=2)
        self.dil_conv1 = MultiDilatedConv((output_channel//2), (output_channel//2), kernel_size=5, 
                             stride=1, padding=2)
        self.dil_bn1 = nn.BatchNorm1d((output_channel//2))
        self.dil_conv2 = MultiDilatedConv((output_channel//2), (output_channel//2), kernel_size=3, 
                             stride=2, padding=1)        
        self.down_sample = nn.Sequential(nn.Conv1d((output_channel//2), (output_channel//2), kernel_size=1, stride=2),
        nn.BatchNorm1d((output_channel//2)))

        self.res_block2=Dil3ResidualBlock((output_channel//2),output_channel, stride=2)
        self.res_block3=Dil3ResidualBlock(output_channel,output_channel, stride=2)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.linear=nn.Linear(output_channel,num_classes)
    def forward(self,x):     
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

        out=self.res_block2(out)
        out=self.res_block3(out) 
        x4=self.gap(out)
        x4=x4.squeeze(-1)
        out=self.linear(x4)
        return out,x4  
    
class projDilConv2Resnet3(nn.Module):
    """
    모듈 3개, (dil-dil-conv)
    [proj-conv1_bn1_relu_maxpool-Dilconv1_bn1_conv2-Dil2Resblock1(bn1_relu_Dil-Dil-bn-conv1)-2-GAP-Linear]
    """    
    def __init__(self, in_channels, output_channel,num_classes,stride=2):
        super(projDilConv2Resnet3, self).__init__()
                        
        self.proj_conv = nn.Conv1d(in_channels, (output_channel//2), kernel_size=5, stride=1, padding=2, bias=False)
        self.conv1 = nn.Conv1d((output_channel//2), (output_channel//2), kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d((output_channel//2))
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        # DilResidualBlock1 직접적용 self.res_block1=DilResidualBlock((output_channel//2),(output_channel//2), stride=2)
        self.dil_conv1 = MultiDilatedConv((output_channel//2), (output_channel//2), kernel_size=5, 
                             stride=1, padding=2)
        self.dil_bn1 = nn.BatchNorm1d((output_channel//2))
        self.conv2 = nn.Conv1d((output_channel//2), (output_channel//2), kernel_size=3, stride=2, padding=1)        
        self.down_sample = nn.Sequential(nn.Conv1d((output_channel//2), (output_channel//2), kernel_size=1, stride=2),
        nn.BatchNorm1d((output_channel//2)))

        self.res_block2=DilConv2ResidualBlock((output_channel//2),output_channel, stride=2)
        self.res_block3=DilConv2ResidualBlock(output_channel,output_channel, stride=2)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.linear=nn.Linear(output_channel,num_classes)
    def forward(self,x):     
        out=self.proj_conv(x)
        out = self.conv1(out)
        out=self.bn1(out)
        out=self.relu(out)
        out=self.maxpool(out)
        identity = out
        out=self.dil_conv1(out)
        out=self.dil_bn1(out)
        out=self.relu(out)
        out=self.conv2(out)
        shortcut = self.down_sample(identity) # stride2니까 사용
        out += shortcut

        out=self.res_block2(out)
        out=self.res_block3(out) 
        x4=self.gap(out)
        x4=x4.squeeze(-1)
        out=self.linear(x4)
        return out,x4


class projDil2ConvResnet3(nn.Module):
    """
    모듈 3개, (dil-dil-conv)
    [proj-conv1_bn1_relu_maxpool-Dilconv1_bn1_conv2-Dil2Resblock1(bn1_relu_Dil-Dil-bn-conv1)-2-GAP-Linear]
    """    
    def __init__(self, in_channels, output_channel,num_classes,stride=2):
        super(projDil2ConvResnet3, self).__init__()
                        
        self.proj_conv = nn.Conv1d(in_channels, (output_channel//2), kernel_size=5, stride=1, padding=2, bias=False)
        self.conv1 = nn.Conv1d((output_channel//2), (output_channel//2), kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d((output_channel//2))
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        # DilResidualBlock1 직접적용 self.res_block1=DilResidualBlock((output_channel//2),(output_channel//2), stride=2)
        self.dil_conv1 = MultiDilatedConv((output_channel//2), (output_channel//2), kernel_size=5, 
                             stride=1, padding=2)
        self.dil_bn1 = nn.BatchNorm1d((output_channel//2))
        self.conv2 = nn.Conv1d((output_channel//2), (output_channel//2), kernel_size=3, stride=2, padding=1)        
        self.down_sample = nn.Sequential(nn.Conv1d((output_channel//2), (output_channel//2), kernel_size=1, stride=2),
        nn.BatchNorm1d((output_channel//2)))

        self.res_block2=Dil2ConvResidualBlock((output_channel//2),output_channel, stride=2)
        self.res_block3=Dil2ConvResidualBlock(output_channel,output_channel, stride=2)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.linear=nn.Linear(output_channel,num_classes)
    def forward(self,x):     
        out=self.proj_conv(x)
        out = self.conv1(out)
        out=self.bn1(out)
        out=self.relu(out)
        out=self.maxpool(out)
        identity = out
        out=self.dil_conv1(out)
        out=self.dil_bn1(out)
        out=self.relu(out)
        out=self.conv2(out)
        shortcut = self.down_sample(identity) # stride2니까 사용
        out += shortcut

        out=self.res_block2(out)
        out=self.res_block3(out) 
        x4=self.gap(out)
        x4=x4.squeeze(-1)
        out=self.linear(x4)
        return out,x4         
class projDil22Resnet3(nn.Module):
    """
    모듈3개(dil-dil)x3 
    [proj-conv1_bn1_relu_maxpool-Dilconv1_bn1_Dilconv2-Dil2Resblock1(bn1_relu_Dil-bn-Dil)-2-GAP-Linear]
    """    
    def __init__(self, in_channels, output_channel,num_classes,stride=2):
        super(projDil22Resnet3, self).__init__()
                        
        self.proj_conv = nn.Conv1d(in_channels, (output_channel//2), kernel_size=5, stride=1, padding=2, bias=False)
        self.conv1 = nn.Conv1d((output_channel//2), (output_channel//2), kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d((output_channel//2))
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        # DilResidualBlock1 직접적용 self.res_block1=DilResidualBlock((output_channel//2),(output_channel//2), stride=2)
        self.dil_conv1 = MultiDilatedConv((output_channel//2), (output_channel//2), kernel_size=5, 
                             stride=1, padding=2)
        self.dil_bn1 = nn.BatchNorm1d((output_channel//2))
        self.dil_conv2 = MultiDilatedConv((output_channel//2), (output_channel//2), kernel_size=3, 
                             stride=2, padding=1)        
        self.down_sample = nn.Sequential(nn.Conv1d((output_channel//2), (output_channel//2), kernel_size=1, stride=2),
        nn.BatchNorm1d((output_channel//2)))

        self.res_block2=Dil2ResidualBlock((output_channel//2),output_channel, stride=2)
        self.res_block3=Dil2ResidualBlock(output_channel,output_channel, stride=2)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.linear=nn.Linear(output_channel,num_classes)
    def forward(self,x):     
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

        out=self.res_block2(out)
        out=self.res_block3(out) 
        x4=self.gap(out)
        x4=x4.squeeze(-1)
        out=self.linear(x4)
        return out,x4  
class projDilResnet3(nn.Module):
    """
    모듈 3개, (dil-conv)x3 
    [proj-conv1_bn1_relu_maxpool-Dilconv1_bn1_conv2-DilResblock1(bn1_relu_Dil-bn-conv)-2-GAP-Linear]
    """    
    def __init__(self, in_channels, output_channel,num_classes,stride=2):
        super(projDilResnet3, self).__init__()
                        
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

        self.res_block2=DilResidualBlock((output_channel//2),output_channel, stride=2)
        self.res_block3=DilResidualBlock(output_channel,output_channel, stride=2)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.linear=nn.Linear(output_channel,num_classes)
    def forward(self,x):     
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

        out=self.res_block2(out)
        out=self.res_block3(out) 
        x4=self.gap(out)
        x4=x4.squeeze(-1)
        out=self.linear(x4)
        return out,x4

class DilResnet(nn.Module):
    def __init__(self, in_channels, output_channel,num_classes,stride=2):
        super(DilResnet, self).__init__()
                        
        self.conv1 = nn.Conv1d(in_channels, (output_channel//2), kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d((output_channel//2))
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.res_block1=DilResidualBlock((output_channel//2),(output_channel//2), stride=2)
        self.res_block2=DilResidualBlock((output_channel//2),output_channel, stride=2)
        self.res_block3=DilResidualBlock(output_channel,output_channel, stride=2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.linear=nn.Linear(output_channel,num_classes)
    def forward(self,x):     
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        out = self.maxpool(x)
        out=self.res_block1(out)
        out=self.res_block2(out)
        out=self.res_block3(out)
        
        x4=self.gap(out)
        x4=x4.squeeze(-1)
        out=self.linear(x4)
        return out,x4

