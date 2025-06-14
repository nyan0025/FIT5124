import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import random
import time

# attack model
class MLP_Attacker(nn.Module):
    def __init__(self):
        super(MLP_Attacker, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# class MLP_Attacker(nn.Module):
#     def __init__(self):
#         super(MLP_Attacker, self).__init__()
#         self.fc1 = nn.Linear(28 * 28, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, 10)

#     def forward(self, x):
#         x = x.view(-1, 28 * 28)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return F.log_softmax(x, dim=1)

# class MLP_Attacker(nn.Module):
#     def __init__(self):
#         super(MLP_Attacker, self).__init__()
#         self.fc1 = nn.Linear(28 * 28, 256)
#         self.fc2 = nn.Linear(256, 128)
#         self.fc_mid = nn.Linear(128, 96)    
#         self.dropout = nn.Dropout(0.2) 
#         self.fc3 = nn.Linear(96, 10)

#     def forward(self, x):
#         x = x.view(-1, 28 * 28)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc_mid(x)) 
#         x = self.dropout(x)
#         x = self.fc3(x)
#         return F.log_softmax(x, dim=1)

# class MLP_Attacker(nn.Module):

#     def __init__(self):

#         super(MLP_Attacker, self).__init__()

#         self.conv1 = nn.Conv2d(1, 6, 3, stride=1, padding=1)

#         self.conv2 = nn.Conv2d(6, 16, 5, stride=1, padding=0)

#         self.dropout1 = nn.Dropout(0.25)

#         self.dropout2 = nn.Dropout(0.5)

#         self.fc1 = nn.Linear(400, 120)

#         self.fc2 = nn.Linear(120, 84)

#         self.fc3 = nn.Linear(84, 10)



#     def forward(self, x):

#         x = self.conv1(x)

#         x = F.relu(x)

#         x = F.max_pool2d(x, 2)

#         x = self.conv2(x)

#         x = F.relu(x)

#         x = F.max_pool2d(x, 2)

#         x = self.dropout1(x)

#         x = torch.flatten(x, 1)

#         x = self.fc1(x)

#         x = F.relu(x)

#         x = self.dropout2(x)

#         x = self.fc2(x)

#         x = F.relu(x)

#         x = self.dropout2(x)

#         x = self.fc3(x)

#         output = F.log_softmax(x, dim=1)

#         return output

# class MLP_Attacker(nn.Module):
#     def __init__(self):
#         super(MLP_Attacker, self).__init__()
#         self.conv1 = nn.Conv2d(1, 4, 3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(4, 8, 5, stride=1, padding=0)
#         self.dropout1 = nn.Dropout(0.25)
#         self.dropout2 = nn.Dropout(0.5)
#         self.fc1 = nn.Linear(200, 60) 
#         self.fc2 = nn.Linear(60, 30)
#         self.fc3 = nn.Linear(30, 10)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, 2)
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, 2)
#         x = self.dropout1(x)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.dropout2(x)
#         x = self.fc2(x)
#         x = F.relu(x)
#         x = self.dropout2(x)
#         x = self.fc3(x)
#         return F.log_softmax(x, dim=1)

# class MLP_Attacker(nn.Module):
#     def __init__(self):
#         super(MLP_Attacker, self).__init__()
#         self.conv1 = nn.Conv2d(1, 6, 3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(6, 12, 5, stride=1, padding=0)
#         self.dropout1 = nn.Dropout(0.25)
#         self.dropout2 = nn.Dropout(0.5)
#         self.fc1 = nn.Linear(300, 100)  # 12 * 5 * 5 = 300
#         self.fc2 = nn.Linear(100, 50)
#         self.fc3 = nn.Linear(50, 10)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, 2)
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, 2)
#         x = self.dropout1(x)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.dropout2(x)
#         x = self.fc2(x)
#         x = F.relu(x)
#         x = self.dropout2(x)
#         x = self.fc3(x)
#         return F.log_softmax(x, dim=1)

# class MLP_Attacker(nn.Module):
#     def __init__(self):
#         super(MLP_Attacker, self).__init__()
#         self.conv1 = nn.Conv2d(1, 12, 3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(12, 32, 5, stride=1, padding=0)
#         self.dropout1 = nn.Dropout(0.25)
#         self.dropout2 = nn.Dropout(0.5)
#         self.fc1 = nn.Linear(800, 400)  # 32 * 4 * 4 = 512
#         self.fc2 = nn.Linear(400, 256)
#         self.fc3 = nn.Linear(256, 10)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, 2)
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, 2)
#         x = self.dropout1(x)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.dropout2(x)
#         x = self.fc2(x)
#         x = F.relu(x)
#         x = self.dropout2(x)
#         x = self.fc3(x)
#         return F.log_softmax(x, dim=1)

# class MLP_Attacker(nn.Module):
#     def __init__(self):
#         super(MLP_Attacker, self).__init__()
#         self.conv1 = nn.Conv2d(1, 6, 3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(6, 16, 5, stride=1, padding=0)
#         self.dropout1 = nn.Dropout(0.25)
#         self.dropout2 = nn.Dropout(0.5)
#         self.fc1 = nn.Linear(400, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, 2)
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, 2)
#         x = self.dropout1(x)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.dropout2(x)
#         x = self.fc2(x)
#         x = F.relu(x)
#         x = self.dropout2(x)
#         x = self.fc3(x)
#         return F.log_softmax(x, dim=1)


# Target model structure（Lenet）
class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(6, 16, 5, stride=1, padding=0)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)  

# Data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.1307,), (0.3081,))
])
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])

# transform = transforms.Compose([
#     transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)), 
#     transforms.ToTensor(),
#     # transforms.Normalize((0.1307,), (0.3081,))
# ])
# Attack data subset
full_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
random.seed(42)
subset_size = int(0.5 * len(full_dataset))
subset_indices = random.sample(range(len(full_dataset)), subset_size)
attack_dataset = Subset(full_dataset, subset_indices)
attack_loader = DataLoader(attack_dataset, batch_size=64, shuffle=True)

# Test set
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Model loading
target_model = Lenet()
target_model.load_state_dict(torch.load("target_model.pth"))
target_model.eval()

attacker_model = MLP_Attacker()
optimizer = optim.Adadelta(attacker_model.parameters(), lr=1)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.75)
kl_loss = nn.KLDivLoss(reduction='batchmean')

# White-box training process: Fit the softmax distribution of the target_model
start_time = time.time()
attacker_model.train()
for epoch in range(1, 4):
    for batch_idx, (data, _) in enumerate(attack_loader):
        with torch.no_grad():
            target_output = target_model(data)  # log_softmax output
            target_probs = target_output.exp()  # softmax probability distribution

        output = attacker_model(data)  # log_softmax output
        loss = kl_loss(output, target_probs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print('Epoch: {} [{}/{} ({:.0f}%)]\tKLDivLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(attack_loader.dataset),
                100. * batch_idx / len(attack_loader), loss.item()))
    scheduler.step()

# Test part
attacker_model.eval()
correct_mimic = 0
correct_true = 0
correct_target = 0

with torch.no_grad():
    for data, target in test_loader:
        target_logit = target_model(data)
        target_pred = target_logit.argmax(dim=1)

        attacker_logit = attacker_model(data)
        pred = attacker_logit.argmax(dim=1)

        correct_mimic += pred.eq(target_pred).sum().item()
        correct_true += pred.eq(target).sum().item()
        correct_target += target_pred.eq(target).sum().item()
end_time = time.time()
total_time = end_time - start_time
print("\nTotal attack execution time (training + testing) :{:.2f} 秒".format(total_time))
print("\n=== Black attack（KLDivLoss）result ===")
total = len(test_loader.dataset)
print("Accuracy vs Target Model: {}/{} ({:.0f}%)".format(correct_mimic, total, 100. * correct_mimic / total))
print("Accuracy vs True Labels (Attacker): {}/{} ({:.0f}%)".format(correct_true, total, 100. * correct_true / total))
print("Accuracy vs True Labels (Target): {}/{} ({:.0f}%)\n".format(correct_target, total, 100. * correct_target / total))