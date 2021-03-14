from capsnet import CapsNet
from getdata import get_train_data,get_test_data
import torch
from torch.autograd import Variable
import torch.utils.data as Data

# train_data:[17023,6000,1],train_tag:[17023] numpy类型
train_data,train_tag=get_train_data()
# test_data,test_tag=get_test_data()
net=CapsNet()
# 数据集加载
# 数据转换成tensor 并[17023,6000,1]=>[17023,1,6000]
train_data=torch.FloatTensor(train_data).permute(0,2,1)
train_tag=torch.LongTensor(train_tag)

train_set=Data.TensorDataset(train_data,train_tag)
train_loader=Data.DataLoader(dataset=train_set,batch_size=32,shuffle=True)
# 优化器设置
optimizer=torch.optim.Adam(net.parameters(),lr=0.001)
loss_func=torch.nn.CrossEntropyLoss()

# train...
print('开始训练：')
accc=[]
epoch=50
maxnum=0
for i in range(epoch):
    for j,(x,y) in enumerate(train_loader):
        
        x,y=Variable(x),Variable(y)
        out=net(x)
        loss=loss_func(out,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('epoch:',i,'j:',j,'轮训练完成')
    # 准确度
    print('开始测试准确度')
    epochnum=0
    epochtotal=0
    for j,(x,y) in enumerate(train_loader):
        print('+++++++++++++')
        x=Variable(x)
        test_out=net(x)
        pred_y=torch.max(test_out,1)[1].data.numpy().squeeze()
        y=y.data.numpy()

        epoch_acc_num=sum(pred_y==y)
        epoch_total_num=len(pred_y)
        epochnum=epochnum+epoch_acc_num
        epochtotal=epochtotal+epoch_total_num

        print('current num:',epochnum,'max num:',maxnum)
    if maxnum<epochnum:
        maxnum=epochnum
    acc=epochnum/epochtotal
    maxacc=maxnum/epochtotal
    accc.append(acc)
    print('本轮准确度acc：',acc,'最高准确度:',maxacc)


    pass
print('acc_arr:',accc,'maxacc',maxacc)