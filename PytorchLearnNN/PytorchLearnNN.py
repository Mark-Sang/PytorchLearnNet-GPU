import torch
import torch.nn
import torch.optim

x=torch.tensor([[0.1, 0.8],[0.8, 0.2]]).cuda()
y=torch.tensor([[1.], [0.]]).cuda()

class MyNet(torch.nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.layer1=torch.nn.Linear(2,1)
        self.relu1=torch.nn.Tanh()
        self.layer2=torch.nn.Linear(1,2)
        self.relu2=torch.nn.Tanh()
        self.layer3=torch.nn.Linear(2,1)
    def forward(self, x):
        x=self.layer1(x)
        x=self.relu1(x)
        x=self.layer2(x)
        x=self.relu2(x)
        x=self.layer3(x)
        return x

net = MyNet().cuda()
mls = torch.nn.MSELoss().cuda()
opt = torch.optim.Adam(net.parameters(), lr=0.01)
for i in range(1000):
    out = net(x).cuda()
    loss = mls(out, y).cuda()
    print(loss)
    opt.zero_grad()
    loss.backward()
    opt.step()

print(net(x))