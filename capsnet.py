import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import parameter
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU

def squash(x,dim=-1):
    '''
    参数：
        x:floatTensor
        dim:求指定维度上的范数
    '''
    norm=torch.norm(x,p=2,dim=dim,keepdim=True) #norm:指定维度的2范
    scale=norm**2/(1+norm**2)/(norm+1e-8)
    return scale*x  #squash公式
    pass
class PrimaryCaps(nn.Module):
    def __init__(self,in_channel=256,out_channel=256,caps_dim=8,kernel_size=5,stride=2,padding=2):
        '''
        参数：
            in_channel:输入的通道数
            out_channel:输出的通道数
            caps_dim:输出的胶囊维度（1个胶囊的长度）
            kernel_size:卷积核大小
            stride：卷积步长
            padding：填充
        '''
        super(PrimaryCaps,self).__init__()
        self.caps_dim=caps_dim
        self.conv=nn.Sequential(
            # x:[batch,256,6000]=>[batch,256,3000]
            nn.Conv1d(in_channel,out_channel,kernel_size,stride,padding),
            # x:[batch,256,3000]=>[batch,256,1000]
            nn.Conv1d(out_channel,out_channel,kernel_size,stride+1,padding),
            # x:[batch,256,1000]=>[batch,256,250]
            nn.Conv1d(out_channel,out_channel,kernel_size,stride+2,padding)
        )
        pass
    def forward(self,x):
        # x:[batch,256,6000]=>[batch,256,250]
        x=self.conv(x)
        # x:[batch,256,250]=>[batch,8000,8]
        x=x.view(x.size(0),-1,self.caps_dim)
        return squash(x)
        pass
    pass
class DenseCaps(nn.Module):
    # 类似于普通cnn的全连接层4
    '''
    参数：
    in_caps_num:输入的胶囊个数
    out_caps_num:输出的胶囊个数
    in_caps_dim:输入的胶囊长度
    out_caps_dim：输出的胶囊长度
    interation_num:动态路径迭代次数
    '''

    def __init__(self,in_caps_num,out_caps_num,in_caps_dim,out_caps_dim,interation_num):
        super(DenseCaps,self).__init__()
        self.in_caps_num=in_caps_num
        self.out_caps_num=out_caps_num
        self.in_caps_dim=in_caps_dim
        self.out_caps_dim=out_caps_dim
        self.interation_num=interation_num
        # 2分类
        # x:[batch,8000,8]
        # W:[2,8000,16,8]
        # x W不能直接矩阵相乘 需要做变换 在forward中做变换
        self.W=nn.Parameter(0.01*torch.randn(out_caps_num,in_caps_num,out_caps_dim,in_caps_dim))
        pass
    def forward(self,x):
        '''
        # 1.x:[batch,8000,8]=>[batch, 1,8000,8, 1]
        #   W:                [       2,8000,16,8]
        # 2.u'=W@x:[batch,2,8000,16,1]===压缩维度===>[batch,2,8000,16]
        '''

        # 1.x:[batch,8000,8]=>[batch, 1,8000,8, 1]
        x=x.unsqueeze(1).unsqueeze(4)
        # 2.u'=W@x:[batch,2,8000,16,1]=缩维度=>[batch,2,8000,16]
        u_hat=self.W@x
        u_hat=u_hat.squeeze(-1)
        # u_hat在动态路由时梯度不需要更新，使用detach截断
        temp_u_hat=u_hat.detach()

        '''
        动态路由算法：
        b=0
        循环：（迭代时u'不更新，最后最后一次迭代时才更新，所以要用temp_u_hat）
            权重c=softmax(b)
            不是最后一次迭代：
                计算s=sum(c*temp_u')
                计算v=squash(s)
                更新b=b+u'@v
            是最后一次迭代：
                计算s=sum(c*u') 此时使用u'以便向后传播求梯度
                计算v
        结束循环
        返回v
        ''' 
        # b:[batch,2,8000,1]
        b=torch.zeros(x.size(0),self.out_caps_num,self.in_caps_num,1)
        # v:[batch,2,16]
        v=torch.ones(x.size(0),self.out_caps_num,self.out_caps_dim)
        for i in range(self.interation_num):
            print('动态路由迭代：',i)
            # 对b在维度dim=1求softmax
            # 可以吧b看做[2,3]矩阵 对dim=0求softmax  每行对应位置的数相加和为1（列相加） 如下所示
            # 0.1 0.5 0.4
            # 0.9 0.5 0.6

            # c:[batch,2,8000,1]
            c=b.softmax(dim=1)
            if i<self.interation_num-1:
                # c*temp_u_hat [batch,2,8000,1]*[batch,2,8000,16]=>[batch,2,8000,16]
                # dim=2求和  ==>s:[batch,2,16]
                s=(c*temp_u_hat).sum(dim=2)
                # v:[batch,2,16]
                v=squash(s)
                # b:[batch,2,8000,1]
                #           u_hat:[batch,2,8000,16]
                # v:[batch,2,16]=>[batch,2,16,1]
                # u_hat@v:[batch,2,8000,1]才能和b相加
                # 这里是为了计算c 不需要u'向后传播  所以使用temp_u_hat
                b=b+temp_u_hat@v.unsqueeze(-1)
            else:
                # 最后一次迭代后 u'需向后传播 所以使用u_hat
                s=(c*u_hat).sum(dim=2)
                v=squash(s)
        # v:[batch,2,16]
        return v
        pass
class CapsNet(nn.Module):
    def __init__(self,in_channel=1,out_channel=256,kernel_size=5,stride=1,padding=2):
        super(CapsNet,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv1d(in_channel,out_channel,kernel_size,stride,padding),
            nn.ReLU()
        )
        
        self.primarycaps=PrimaryCaps(256,256,8,5,2,2)
        self.densecaps=DenseCaps(8000,2,8,16,3)
        pass
    def forward(self,x):
        # x:[batch,1,6000]=>[batch,256,6000]
        output=self.conv1(x)
        # output:[batch,256,250]
        output=self.primarycaps(output)
        # output:[batch,2,16]
        output=self.densecaps(output)
        
        # output:[batch,2]  得到2范
        output=torch.norm(output,dim=-1)
        # print(output.size())
        return output
        pass

def main():
    ecg=torch.randn(16,1,6000)
    model=CapsNet()
    y=model(ecg)
    print(y)
    pass

if __name__=='__main__':
    main()