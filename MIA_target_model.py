import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from torch.optim.lr_scheduler import StepLR

# ========== 1. Define LeNet (Adjusted for Higher Capacity and Overfitting) ==========
class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)  # filters: 6 -> 16
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1) # filters: 16 -> 32, kernel 5->3, padding 0->1
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)

        # 禁用所有 Dropout 层
        self.dropout1 = nn.Dropout(0.0) 
        self.dropout2 = nn.Dropout(0.0) 
        
        # Input: 32x32x3 (CIFAR-10)
        # conv1 (3x3, s=1, p=1): Output = 32x32x16
        # max_pool2d (k=2): Output = 16x16x16
        # conv2 (3x3, s=1, p=1): Output = 16x16x32
        # max_pool2d (k=2): Output = 8x8x32
        # conv3 (3x3, s=1, p=1): Output = 8x8x64
        # max_pool2d (k=2): Output = 4x4x64
        # Flatten: 4 * 4 * 64 = 1024 features

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256) 
        self.fc3 = nn.Linear(256, 128) 
        self.fc4 = nn.Linear(128, 10)     

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x)) 
        x = F.max_pool2d(x, 2) 
        
        x = torch.flatten(x, 1) 

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x)) 
        x = F.relu(self.fc3(x)) 
        x = self.fc4(x)         

        return F.log_softmax(x, dim=1)

# ========== 2. Load and split dataset (CIFAR-10) ==========
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

full_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)

train_indices = list(range(0, 300))
train_subset = Subset(full_dataset, train_indices)
test_subset = test_dataset 

batch_size = 32
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
eval_train_loader = DataLoader(train_subset, batch_size=len(train_subset), shuffle=False) 
test_loader = DataLoader(test_subset, batch_size=128, shuffle=False) 

model = Lenet()
optimizer = optim.Adam(model.parameters(), lr=0.001) 
scheduler = StepLR(optimizer, step_size=1, gamma=0.99) 

num_epochs = 300

model.train()
print(f"Starting training for {num_epochs} epochs with {len(train_subset)} training samples and batch size {train_loader.batch_size}.")
for epoch in range(1, num_epochs + 1):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
    scheduler.step()
    if epoch % 50 == 0 or epoch == num_epochs: 
        print(f"[INFO] Epoch {epoch}/{num_epochs} completed, Loss: {loss.item():.4f}")

# ========== 4. Evaluate ==========
def evaluate(loader, name):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            total_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    avg_loss = total_loss / len(loader.dataset)
    acc = 100. * correct / len(loader.dataset)
    print(f"[{name}] Loss: {avg_loss:.4f}, Accuracy: {correct}/{len(loader.dataset)} ({acc:.2f}%)")
    return acc

# ========== 5. Run evaluation ==========
print("\n--- Evaluation Results ---")
train_acc = evaluate(eval_train_loader, "Train")
test_acc = evaluate(test_loader, "Test")

# ========== 6. Save model ==========
torch.save(model.state_dict(), "cifar10_overfit_target_model_fixed_v6.pth")
print("[INFO] Model saved to cifar10_overfit_target_model_fixed_v6.pth")
