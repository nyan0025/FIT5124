import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score
import random # For setting random seeds
import time   # For generating truly random seeds for each run

start_time = time.time()
# --- Global Settings ---
NUM_SHADOW_MODELS = 5 # The number of trained shadow models
SHADOW_MODEL_EPOCHS = 50 # The number of training Epochs for each shadow model (matching the Epochs of the target model to ensure deep overfitting)
ATTACK_MODEL_EPOCHS = 200 # Increased training Epochs for the attack model to allow more learning
NOISE_FLIP_RATIO = 0.1 # Pixel flip noise ratio (keep consistent with the description in the paper)
# Key adjustment: The total number of samples obtained by each shadow model from the auxiliary dataset.
SHADOW_DATA_SIZE_PER_MODEL = 60 # For example: 30 are used for training and 30 are used for extracting non-trained features.
                                # Ensure that the data volume of each shadow model is similar to the training data volume of the target model to simulate deep overfitting.

# The prediction threshold in the evaluation of attack models (used to adjust the trade-off between precision and recall rates)
PREDICTION_THRESHOLD = 0.5 # The default is 0.5. You can try to increase this value to improve Precision and reduce Recall.

# Class weights during the training of the attack model (used to solve the imbalance of training data)
ATTACK_CLASS_WEIGHTS = None # It is not set initially. It can be dynamically calculated in train_attack_model_func or passed in manually


# --- Set random seeds for within-run diversity of shadow models ---
# Global script seed is removed to ensure different results across different script runs.
def set_individual_shadow_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True # Uncomment if CUDA used and strictly deterministic behavior is needed
    # torch.backends.cudnn.benchmark = False


# ==== Pixel Flip Noise Function (Precise Match to Paper's Description) ====
def add_pixel_flip_noise(img_tensor, flip_ratio=0.1):
    noisy_img_tensor = img_tensor.clone()
    
    # Process each image in the batch
    for i in range(img_tensor.shape[0]):
        flat_img = noisy_img_tensor[i].flatten()
        n_flips = int(flat_img.shape[0] * flip_ratio)
        
        if n_flips == 0 and flip_ratio > 0 and flat_img.shape[0] > 0:
            n_flips = 1 # Ensure at least one pixel is flipped if ratio > 0

        if n_flips > 0:
            # Use device for torch.randperm
            idx_to_flip = torch.randperm(flat_img.shape[0], device=img_tensor.device)[:n_flips]
            flat_img[idx_to_flip] = 1.0 - flat_img[idx_to_flip] # Flip the pixel value

        noisy_img_tensor[i] = flat_img.view(img_tensor[i].shape)
        
    return noisy_img_tensor


# ========== LeNet (for Target Model) ==========
class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)  # filters: 6 -> 16
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1) # filters: 16 -> 32, kernel 5->3, padding 0->1
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1) 

        self.dropout1 = nn.Dropout(0.0) 
        self.dropout2 = nn.Dropout(0.0) 

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


# ========== Attacker's Shadow Model Definition (My_Shadow_model - now similar but not identical to LeNet) ==========
# This model will be trained by the attacker on their auxiliary data.
class My_Shadow_model(nn.Module):
    def __init__(self):
        super(My_Shadow_model, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 60, 3, stride=1, padding=1)

        self.dropout1 = nn.Dropout(0.0) 
        self.dropout2 = nn.Dropout(0.0) 
        self.fc1 = nn.Linear(960, 512) # Adjusted input from 1024 to 960
        self.fc2 = nn.Linear(512, 240) # Modified: 256 -> 240 neurons
        self.fc3 = nn.Linear(240, 128) # Adjusted input from 256 to 240
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

# class My_Shadow_model_VerySimple_Unknown(nn.Module):
#     def __init__(self):
#         super(My_Shadow_model_VerySimple_Unknown, self).__init__()
#         self.conv = nn.Conv2d(3, 8, 5, stride=1, padding=2)               #Simple + Unknown
#         self.fc = nn.Linear(8 * 16 * 16, 10)

#     def forward(self, x):
#         x = F.relu(self.conv(x))
#         x = F.max_pool2d(x, 2)  # 32 -> 16
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#         return F.log_softmax(x, dim=1)


# class My_Shadow_model_Complex_Unknown(nn.Module):
#     def __init__(self):
#         super(My_Shadow_model_Complex, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
#         self.conv3 = nn.Conv2d(64, 128, 3, padding=1)                     #Complex + Unknown
#         self.conv4 = nn.Conv2d(128, 128, 3, padding=1)

#         self.fc1 = nn.Linear(128 * 4 * 4, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, 128)
#         self.fc4 = nn.Linear(128, 10)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))     # [32x32]
#         x = F.max_pool2d(x, 2)        # [16x16]
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, 2)        # [8x8]
#         x = F.relu(self.conv3(x))
#         x = F.max_pool2d(x, 2)        # [4x4]
#         x = F.relu(self.conv4(x))     # [4x4]
#         x = torch.flatten(x, 1)       # 128*4*4 = 2048
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = self.fc4(x)
#         return F.log_softmax(x, dim=1)

# class Lenet_Simple(nn.Module):
#     def __init__(self):
#         super(Lenet_Simple, self).__init__()
#         self.conv1 = nn.Conv2d(3, 8, 3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(8, 16, 3, stride=1, padding=1)             #Simple + Known

#         self.fc1 = nn.Linear(16 * 8 * 8, 128)  # [B,16,8,8] -> flatten -> 1024
#         self.fc2 = nn.Linear(128, 10)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))        # -> [8,32,32]
#         x = F.max_pool2d(x, 2)           # -> [8,16,16]
#         x = F.relu(self.conv2(x))        # -> [16,16,16]
#         x = F.max_pool2d(x, 2)           # -> [16,8,8]

#         x = torch.flatten(x, 1)          # -> [B,1024]
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)

# class Lenet_Complex(nn.Module):
#     def __init__(self):
#         super(Lenet_Complex, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
#         self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
#         self.conv4 = nn.Conv2d(128, 128, 3, stride=1, padding=1)          #Complex + Known

#         self.fc1 = nn.Linear(128 * 4 * 4, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, 128)
#         self.fc4 = nn.Linear(128, 64)
#         self.fc5 = nn.Linear(64, 10)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))       # -> [32,32,32]
#         x = F.max_pool2d(x, 2)          # -> [32,16,16]
#         x = F.relu(self.conv2(x))       # -> [64,16,16]
#         x = F.max_pool2d(x, 2)          # -> [64,8,8]
#         x = F.relu(self.conv3(x))       # -> [128,8,8]
#         x = F.max_pool2d(x, 2)          # -> [128,4,4]
#         x = F.relu(self.conv4(x))       # -> [128,4,4]

#         x = torch.flatten(x, 1)         # -> [2048]
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = F.relu(self.fc4(x))
#         x = self.fc5(x)
#         return F.log_softmax(x, dim=1)


# ========== Attack Model Definition (Binary Classifier for Membership) ==========
class AttackModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Input features: 10 logits + 1 (max_prob) + 1 (entropy) + 10 one-hot labels + 1 loss value = 23 features

        self.fc1 = nn.Linear(23, 256) # Increased hidden layer size
        self.fc2 = nn.Linear(256, 128) # Added another hidden layer
        self.fc3 = nn.Linear(128, 64) # Added another hidden layer
        self.fc4 = nn.Linear(64, 2) # Final output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x) # Return raw logits for CrossEntropyLoss


# === One-hot encoder ===
def one_hot(y, num_classes=10):
    return F.one_hot(y, num_classes).float()


# ==== Feature Collection Function (Used for both Shadow and Target Model) ====
def collect_features_for_attack(model, loader, is_member_flag, add_noise_to_input=False, noise_flip_ratio=0.1):
    X_features, Y_labels = [], []
    model.eval() # Ensure the model is in evaluation mode
    with torch.no_grad():
        for x_batch, y_batch in loader:
            if add_noise_to_input: # If this is True, it means we are adding noise to *clean* input
                x_batch = add_pixel_flip_noise(x_batch, flip_ratio=noise_flip_ratio)
            out_log_probs = model(x_batch) # Model's log_softmax output
            
            # --- (max probability) ---
            probabilities = torch.exp(out_log_probs) # Convert log-probabilities to probabilities
            max_probs, _ = torch.max(probabilities, dim=1)
            max_probs = max_probs.unsqueeze(1) # Make it 1 dimension

            # --- (Entropy) ---
            epsilon = 1e-10 # Add a small epsilon to avoid log(0)
            entropy = -torch.sum(probabilities * torch.log(probabilities + epsilon), dim=1).unsqueeze(1) # 1 dimension

            # Feature 1: Logits (log_probabilities) - 10 dimensions
            # This is directly out_log_probs

            # Feature 2: NLL Loss per sample (strong membership signal)
            losses = F.nll_loss(out_log_probs, y_batch, reduction='none').unsqueeze(1) # 1 dimension

            # Feature 3: One-Hot Encoding of True Labels
            one_hot_labels = one_hot(y_batch, num_classes=10) # 10 dimensions

            # Concatenate all features: 10 (logits) + 1 (max_prob) + 1 (entropy) + 10 (one-hot) + 1 (loss) = 23 dimensions
            features_batch = torch.cat([out_log_probs, max_probs, entropy, one_hot_labels, losses], dim=1)

            X_features.append(features_batch)
            Y_labels.extend([is_member_flag] * x_batch.size(0)) # Assign 1 for member, 0 for non-member

    return torch.cat(X_features), torch.tensor(Y_labels)


# ========== Training Function for Attack Model ==========
def train_attack_model_func(x, y, class_weights=None):
    model = AttackModel()
    loader = DataLoader(TensorDataset(x, y), batch_size=64, shuffle=True) # Adjusted batch size

    if class_weights is not None:
        weights_tensor = torch.tensor(class_weights, dtype=torch.float)
        loss_fn = nn.CrossEntropyLoss(weight=weights_tensor)
        print(f"[INFO] Using CrossEntropyLoss with class weights: {class_weights}")
    else:
        loss_fn = nn.CrossEntropyLoss()

    opt = optim.Adam(model.parameters(), lr=0.001) # Increased learning rate slightly for attack model
    model.train()
    print("[INFO] Training Attack Model...")
    for epoch in range(ATTACK_MODEL_EPOCHS):
        for batch_x, batch_y in loader:
            opt.zero_grad()
            out = model(batch_x)
            loss = loss_fn(out, batch_y)
            loss.backward()
            opt.step()
        if (epoch + 1) % 20 == 0:
            print(f"Attack Model Epoch {epoch+1}/{ATTACK_MODEL_EPOCHS}, Loss: {loss.item():.4f}")
    print("[INFO] Attack Model training completed.")
    return model


# ========== Evaluation Function for Attack Model ==========
def evaluate_attack_func(model, x, y, prediction_threshold=0.5):
    model.eval()
    y_true_eval, y_pred_hard_eval, y_probs_eval = [], [], []
    with torch.no_grad():
        eval_loader = DataLoader(TensorDataset(x, y), batch_size=64, shuffle=False)
        for x_batch, y_batch in eval_loader:
            output_raw_logits = model(x_batch)
            probabilities = F.softmax(output_raw_logits, dim=1) # Convert raw logits to probabilities

            preds_hard = (probabilities[:, 1] >= prediction_threshold).long() # Predict 1 if prob >= threshold
            
            positive_class_probs = probabilities[:, 1] # Probability of being 'member' (class 1)

            y_true_eval.extend(y_batch.tolist())
            y_pred_hard_eval.extend(preds_hard.tolist())
            y_probs_eval.extend(positive_class_probs.tolist())
            
    print(f"========== MIA Attack Evaluation (Threshold: {prediction_threshold:.2f}) ==========")
    print("Accuracy :", accuracy_score(y_true_eval, y_pred_hard_eval))
    print("Precision:", precision_score(y_true_eval, y_pred_hard_eval, zero_division=0))
    print("Recall   :", recall_score(y_true_eval, y_pred_hard_eval, zero_division=0))
    print("AUC      :", roc_auc_score(y_true_eval, y_probs_eval))


# ========== Main Pipeline ==========

# --- Generate a truly random base seed for this script run ---
# This ensures that each time you run the entire script, you get different results.
base_random_seed = int(time.time())
print(f"[INFO] Using base random seed: {base_random_seed} for this run.")
random.seed(base_random_seed) # Set the base seed for general random operations


# --- Load Datasets ---
print("[INFO] Loading CIFAR-10 datasets...") # Changed to CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
full_cifar10_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform) # Changed to CIFAR-10
test_cifar10_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform) # Changed to CIFAR-10


# --- Load Pretrained Target Model ---
print("[INFO] Loading pre-trained Target Model (cifar10_overfit_target_model_fixed_v6.pth)...") # Changed filename
target_model = Lenet()
target_model.load_state_dict(torch.load("cifar10_overfit_target_model_fixed_v6.pth", map_location=torch.device("cpu")))
target_model.eval() # Set target model to evaluation mode


# --- STEP 1 & 2: Build and Train Ensemble Shadow Models & Extract Behavioral Fingerprints ---
# This section combines Step 1 and Step 2 as they are interdependent in an ensemble setup.
print("\n--- STEP 1 & 2: Building and Training Ensemble Shadow Models & Extracting Features ---")

all_shadow_features_for_attack = []
all_shadow_labels_for_attack = []

# Determine available indices for shadow datasets
train_target_model_size = 300 # The target model was trained on 300 samples
available_indices_start = train_target_model_size # Start after target model's training data
total_aux_data_per_model = SHADOW_DATA_SIZE_PER_MODEL # Total data (member + non-member) for one shadow model
total_aux_data_needed = NUM_SHADOW_MODELS * total_aux_data_per_model

if available_indices_start + total_aux_data_needed > len(full_cifar10_dataset): # Changed dataset name
    print(f"[WARNING] Not enough auxiliary data for {NUM_SHADOW_MODELS} shadow models with {SHADOW_DATA_SIZE_PER_MODEL} samples each.")
    print(f"  Required: {available_indices_start + total_aux_data_needed}, Available: {len(full_cifar10_dataset)}") # Changed dataset name
    print("  Adjusting number of shadow models based on available data.")
    NUM_SHADOW_MODELS = (len(full_cifar10_dataset) - available_indices_start) // total_aux_data_per_model # Changed dataset name
    if NUM_SHADOW_MODELS == 0:
        raise ValueError("Not enough auxiliary data to train even one shadow model. Increase full_cifar10_dataset size or decrease SHADOW_DATA_SIZE_PER_MODEL.") # Changed dataset name
    print(f"  New NUM_SHADOW_MODELS: {NUM_SHADOW_MODELS}")


print(f"Training {NUM_SHADOW_MODELS} shadow models.")

for i in range(NUM_SHADOW_MODELS):
    # Set a new seed for each shadow model to ensure different random initializations and data shuffling
    # We use base_random_seed + i to ensure diversity between shadow models *within* a run,
    # and base_random_seed ensures diversity *between* runs.
    set_individual_shadow_seed(base_random_seed + i) 
    
    print(f"\nTraining Shadow Model {i+1}/{NUM_SHADOW_MODELS}...")
    
    current_aux_data_start_idx = available_indices_start + i * total_aux_data_per_model
    current_aux_data_end_idx = current_aux_data_start_idx + total_aux_data_per_model
    
    # Get the clean subset from full_cifar10_dataset for this specific shadow model
    # Ensure not to go out of bounds
    if current_aux_data_end_idx > len(full_cifar10_dataset):
        print(f"[WARNING] Adjusting shadow model {i+1} aux data end index due to dataset size limit.")
        current_aux_data_end_idx = len(full_cifar10_dataset)
    
    clean_current_shadow_aux_subset_data = torch.stack([full_cifar10_dataset[j][0] for j in range(current_aux_data_start_idx, current_aux_data_end_idx)])
    clean_current_shadow_aux_subset_labels = torch.tensor([full_cifar10_dataset[j][1] for j in range(current_aux_data_start_idx, current_aux_data_end_idx)])

    # CRITICAL CHANGE: Apply noise ONCE to the entire auxiliary data chunk for THIS shadow model
    noisy_current_shadow_aux_data = add_pixel_flip_noise(clean_current_shadow_aux_subset_data, flip_ratio=NOISE_FLIP_RATIO)

    # Split the *already noisy* data into training and non-training parts for the shadow model
    member_split_size = SHADOW_DATA_SIZE_PER_MODEL // 2 # e.g., 30 members for shadow training
    
    # Ensure split size doesn't exceed available data
    if member_split_size > len(noisy_current_shadow_aux_data):
        member_split_size = len(noisy_current_shadow_aux_data) // 2
    
    shadow_train_x = noisy_current_shadow_aux_data[:member_split_size]
    shadow_train_y = clean_current_shadow_aux_subset_labels[:member_split_size] # Labels are not noisy
    
    shadow_non_train_x = noisy_current_shadow_aux_data[member_split_size:]
    shadow_non_train_y = clean_current_shadow_aux_subset_labels[member_split_size:]

    # Create DataLoaders from the *already noisy tensors*
    shadow_train_loader = DataLoader(TensorDataset(shadow_train_x, shadow_train_y), batch_size=32, shuffle=True) # Mimic target batch size (32)
    shadow_non_train_loader = DataLoader(TensorDataset(shadow_non_train_x, shadow_non_train_y), batch_size=32, shuffle=False) # For collecting non-member features


    # Initialize and train this single shadow model instance
    shadow_model = My_Shadow_model() # Using My_Shadow_model (which inherits LeNet)
    # Mimic target optimizer and scheduler from our_model.py for similar overfitting behavior
    shadow_opt = optim.Adam(shadow_model.parameters(), lr=0.001) # Changed to Adam and 0.001
    shadow_scheduler = StepLR(shadow_opt, step_size=1, gamma=0.99) # Matched target model scheduler gamma

    for epoch in range(1, SHADOW_MODEL_EPOCHS + 1):
        shadow_model.train()
        for batch_idx, (data, target) in enumerate(shadow_train_loader):
            # CRITICAL CHANGE: NOISE IS ALREADY APPLIED. Just use 'data' directly.
            shadow_opt.zero_grad()
            output = shadow_model(data) # 'data' (batch_x) is already noisy
            loss = F.nll_loss(output, target)
            loss.backward()
            shadow_opt.step()
        shadow_scheduler.step()
        if epoch % 20 == 0:
            print(f"  Shadow Model {i+1} Epoch {epoch}/{SHADOW_MODEL_EPOCHS}, Loss: {loss.item():.4f}")
    print(f"Shadow Model {i+1} training completed.")

    # Collect features from this trained shadow model
    # add_noise_to_input must be FALSE because the loaders already provide noisy data.
    current_shadow_member_features, current_shadow_member_attack_labels = \
        collect_features_for_attack(shadow_model, shadow_train_loader, 1, add_noise_to_input=False) 
    current_shadow_non_member_features, current_shadow_non_member_attack_labels = \
        collect_features_for_attack(shadow_model, shadow_non_train_loader, 0, add_noise_to_input=False) 
    
    all_shadow_features_for_attack.append(current_shadow_member_features)
    all_shadow_features_for_attack.append(current_shadow_non_member_features)
    all_shadow_labels_for_attack.append(current_shadow_member_attack_labels)
    all_shadow_labels_for_attack.append(current_shadow_non_member_attack_labels)

# Concatenate all collected features from all shadow models to form the Attack Model's training dataset
attack_train_X = torch.cat(all_shadow_features_for_attack)
attack_train_Y = torch.cat(all_shadow_labels_for_attack)

attack_train_dataset = TensorDataset(attack_train_X, attack_train_Y)
attack_train_loader = DataLoader(attack_train_dataset, batch_size=64, shuffle=True)
print(f"\nTotal attack model training dataset size: {len(attack_train_X)} samples (from {NUM_SHADOW_MODELS} shadow models).")


# --- STEP 3: Train an Attack Model ---
print("\n--- STEP 3: Training Attack Model ---")
# Reset seed before training the attack model to ensure its initialization is consistent for this part
set_individual_shadow_seed(base_random_seed + NUM_SHADOW_MODELS) # Use a new deterministic seed for attack model
# Calculate class weights for the attack model training to handle potential imbalance
member_count = attack_train_Y.sum().item()
non_member_count = len(attack_train_Y) - member_count
if member_count > 0 and non_member_count > 0: # Ensure both classes exist
    # Calculate weights based on inverse frequency
    weight_for_non_member = len(attack_train_Y) / (2.0 * non_member_count)
    weight_for_member = len(attack_train_Y) / (2.0 * member_count)
    attack_class_weights_tensor = torch.tensor([weight_for_non_member, weight_for_member], dtype=torch.float)
    print(f"[INFO] Attack model class weights: {attack_class_weights_tensor.tolist()}")
else:
    attack_class_weights_tensor = None
    print("[INFO] Cannot apply class weights due to imbalanced or empty classes in attack training data.")


attack_model = train_attack_model_func(attack_train_X, attack_train_Y, class_weights=attack_class_weights_tensor)


# --- STEP 4: Evaluate Attack Effect (Against the TRUE Target Model) ---
print("\n--- STEP 4: Evaluating Attack Effect Against TRUE Target Model ---")

# Prepare actual members and non-members for the target model
# Real target members (the 300 images the target model was trained on, from original train_dataset)
true_target_members_indices = list(range(0, train_target_model_size)) # Changed to 300
true_target_members_subset = Subset(full_cifar10_dataset, true_target_members_indices) # Changed dataset name
true_target_members_loader = DataLoader(true_target_members_subset, batch_size=32, shuffle=False) # Changed batch size

# Real target non-members (samples NOT seen by the target model during its training).
# These should also be disjoint from any data used to train shadow models.
eval_non_member_multiplier = 1 # Multiplier for non-member count relative to true members (300 * 1 = 300 non-members for eval)
eval_non_member_count = len(true_target_members_indices) * eval_non_member_multiplier 

# Start non-member indices after all data used by target model (0-299) and shadow models.
# Make sure this range is entirely separate from data used for shadow model training.
# A buffer of 1000 is good to ensure disjointness.
eval_non_member_start_idx = available_indices_start + total_aux_data_needed + 1000 

true_target_non_members_indices = list(range(eval_non_member_start_idx, eval_non_member_start_idx + eval_non_member_count)) 

# Ensure we don't go out of bounds of full_cifar10_dataset
if true_target_non_members_indices and true_target_non_members_indices[-1] >= len(full_cifar10_dataset): # Changed dataset name
    print(f"[WARNING] Adjusting true target non-members count due to dataset size limit. Max index: {len(full_cifar10_dataset)-1}") # Changed dataset name
    true_target_non_members_indices = list(range(eval_non_member_start_idx, len(full_cifar10_dataset))) # Changed dataset name
    if len(true_target_non_members_indices) == 0:
        raise ValueError("Not enough data for true target non-members evaluation.")
elif not true_target_non_members_indices: # If list is empty due to start_idx being too high
    raise ValueError(f"Could not find valid range for true target non-members starting from {eval_non_member_start_idx}. Increase dataset size or adjust ranges.")


true_target_non_members_subset = Subset(full_cifar10_dataset, true_target_non_members_indices) # Changed dataset name
true_target_non_members_loader = DataLoader(true_target_non_members_subset, batch_size=32, shuffle=False) # Changed batch size


# Collect features from the TRUE TARGET MODEL for evaluation
# IMPORTANT: Since target model was trained on CLEAN data, we query it with CLEAN data here.
# Removed add_noise_to_input=True here.
eval_member_features, eval_member_labels = \
    collect_features_for_attack(target_model, true_target_members_loader, 1, add_noise_to_input=False) 
print(f"Collected {len(eval_member_features)} true target member features.")

eval_non_member_features, eval_non_member_labels = \
    collect_features_for_attack(target_model, true_target_non_members_loader, 0, add_noise_to_input=False) 
print(f"Collected {len(eval_non_member_features)} true target non-member features.")


# Combine features for final attack evaluation
final_attack_eval_X = torch.cat([eval_member_features, eval_non_member_features])
final_attack_eval_Y_true = torch.cat([eval_member_labels, eval_non_member_labels])

final_attack_eval_loader = DataLoader(TensorDataset(final_attack_eval_X, final_attack_eval_Y_true), batch_size=64, shuffle=False)

# Perform final inference with the trained Attack Model
# Test with default threshold (0.5) first
print("\n--- Evaluation with default threshold (0.5) ---")
evaluate_attack_func(attack_model, final_attack_eval_X, final_attack_eval_Y_true, prediction_threshold=0.5)

# Then test with a higher threshold to improve Precision
print("\n--- Evaluation with higher threshold (e.6, 0.7, 0.8) for better Precision ---")
evaluate_attack_func(attack_model, final_attack_eval_X, final_attack_eval_Y_true, prediction_threshold=0.6)
evaluate_attack_func(attack_model, final_attack_eval_X, final_attack_eval_Y_true, prediction_threshold=0.7)
evaluate_attack_func(attack_model, final_attack_eval_X, final_attack_eval_Y_true, prediction_threshold=0.8)

print("\n--- Evaluation with even higher threshold (e.g., 0.95) for even better Precision ---")
evaluate_attack_func(attack_model, final_attack_eval_X, final_attack_eval_Y_true, prediction_threshold=0.95)

# Save the parameters of the attack model to the.pth file
attack_model_save_path = "MIA_attack_model.pth"
torch.save(attack_model.state_dict(), attack_model_save_path)
print(f"\n[Information] The trained attack model has been saved to: {attack_model_save_path}")

end_time = time.time()
total_time = end_time - start_time
print("\nTotal attack execution time (training + testing) :{:.2f} seconds".format(total_time))