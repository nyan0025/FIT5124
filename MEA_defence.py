import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import random
import time

def noisy_log_softmax_preserve_label(log_probs, noise_std=2.0, max_attempts=10):
    """
    log_probs: Tensor [B, C] — original log_softmax output
    return: softmax distribution (target_probs), used for KLDivLoss
    """
    # initial part
    batch_size, num_classes = log_probs.size()
    original_labels = log_probs.argmax(dim=1)
    final_softmax = torch.zeros_like(log_probs)
    success_count = 0
    fail_count = 0
    # Traverse the entire batch
    for i in range(batch_size):
        original_log = log_probs[i].unsqueeze(0)
        original_label = original_labels[i].item()
        success = False

        for _ in range(max_attempts):
            noise = torch.randn_like(original_log) * noise_std
            noisy_log = original_log + noise        # add noise
            pred_label = noisy_log.argmax(dim=1).item()

            if pred_label == original_label:
                final_softmax[i] = noisy_log.exp()  # only exp AFTER noise
                success = True
                break

        if not success:
            final_softmax[i] = original_log.exp()  # fallback: original distribution
            fail_count += 1
        else:
            success_count += 1

    print(f"[NoisyLog] Success: {success_count}, Fallbacks: {fail_count}")
    return final_softmax


# Attacker model
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

# Target model structure
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
        return F.log_softmax(x, dim=1)  # 输出 log_softmax

# Data preprocessing
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     # transforms.Normalize((0.1307,), (0.3081,))
# ])
# transform = transforms.Compose([
#     transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)), 
#     transforms.ToTensor(),
#     # transforms.Normalize((0.1307,), (0.3081,))
# ])

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
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
# mse_loss = nn.MSELoss(reduction='mean')
# White-box training process: Fit the softmax distribution of the target_model

start_time = time.time()
attacker_model.train()
for epoch in range(1, 4):
    for batch_idx, (data, _) in enumerate(attack_loader):
        with torch.no_grad():
            target_output = target_model(data)  # log_softmax output
            # target_probs = target_output.exp() 
            target_probs = noisy_log_softmax_preserve_label(target_output, noise_std=10.0)

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

# Testing stage
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
print("\nTotal attack execution time (training + testing) :{:.2f} seconds".format(total_time))
print("\n=== Black box attack（KLDivLoss）result ===")
total = len(test_loader.dataset)
print("Accuracy vs Target Model: {}/{} ({:.0f}%)".format(correct_mimic, total, 100. * correct_mimic / total))
print("Accuracy vs True Labels (Attacker): {}/{} ({:.0f}%)".format(correct_true, total, 100. * correct_true / total))
print("Accuracy vs True Labels (Target): {}/{} ({:.0f}%)\n".format(correct_target, total, 100. * correct_target / total))
