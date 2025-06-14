import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.optim.lr_scheduler import StepLR 
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score
import random 
import time   


# --- Global Settings ---
NUM_SHADOW_MODELS = 5 # Number of shadow models
SHADOW_MODEL_EPOCHS = 50 # Training epochs for shadow models
ATTACK_MODEL_EPOCHS = 200 # Training epochs for attack models
NOISE_FLIP_RATIO = 0.1 # Pixel flip noise ratio for input images
SHADOW_DATA_SIZE_PER_MODEL = 60 # Size of auxiliary data per shadow model (e.g., 30 members for shadow training, 30 for non-members)

PREDICTION_THRESHOLD = 0.5 # Prediction threshold for the attack model

# --- MemGuard Defense Setting (for internal optimization) ---
MEMGUARD_EPSILON = 0.5 # L2 distortion budget for the added noise 'n'.
MEMGUARD_INNER_ITERS = 1 # Number of internal optimization steps for finding 'n' per batch.
MEMGUARD_LEARNING_RATE = 0.01 # Learning rate for optimizing 'n'.


def set_individual_shadow_seed(seed):
    """Sets random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def add_pixel_flip_noise(img_tensor, flip_ratio=0.1):
    """Applies pixel flip noise to an image tensor."""
    noisy_img_tensor = img_tensor.clone()
    for i in range(img_tensor.shape[0]):
        flat_img = noisy_img_tensor[i].flatten()
        n_flips = int(flat_img.shape[0] * flip_ratio)
        if n_flips == 0 and flip_ratio > 0 and flat_img.shape[0] > 0:
            n_flips = 1
        if n_flips > 0:
            idx_to_flip = torch.randperm(flat_img.shape[0], device=img_tensor.device)[:n_flips]
            flat_img[idx_to_flip] = 1.0 - flat_img[idx_to_flip]
        noisy_img_tensor[i] = flat_img.view(img_tensor[i].shape)
    return noisy_img_tensor


# ======== MemGuard Perturbation Function (PGD-like Optimization for Noise 'n') ========
def memguard_perturb_logits(target_model_logits, true_labels, attack_model_g_prime, 
                            epsilon_mg, inner_iterations, learning_rate):
    """
    Implements MemGuard's perturbation by performing an internal optimization
    to find noise 'n' that pushes the attack model's output towards 0.5,
    while adhering to predicted label preservation, non-negativity, and L2 distortion budget.
    This simulates the defender's internal iterative process using its knowledge (g').
    """
    # Detach original_logits: We optimize 'n', not gradients through the target model.
    original_logits = target_model_logits.clone().detach() 
    original_pred_labels = torch.argmax(original_logits, dim=1)

    # Initialize noise 'n' (requires gradients for optimization)
    n = torch.zeros_like(original_logits, requires_grad=True)

    optimizer_n = optim.Adam([n], lr=learning_rate)
    attack_model_g_prime.eval() # Ensure attack model is in eval mode for inference

    for i in range(inner_iterations):
        optimizer_n.zero_grad()
        
        # Calculate perturbed_logits. This tensor and its dependents will track gradients to 'n'.
        perturbed_logits = original_logits + n

        # Prepare features for the attack model using the current perturbed_logits
        # perturbed_log_probs will retain gradient dependency on 'n'
        perturbed_log_probs = F.log_softmax(perturbed_logits, dim=1) 

        # losses_for_attack_feature also retains gradient dependency on 'n' via perturbed_log_probs
        losses_for_attack_feature = F.nll_loss(perturbed_log_probs, true_labels, reduction='none').unsqueeze(1)
        
        # one_hot_labels is a constant feature, explicitly detach to avoid any potential issues
        one_hot_labels = one_hot(true_labels, num_classes=10).detach() 
        
        # attack_features will contain tensors that depend on 'n' (perturbed_log_probs, losses_for_attack_feature)
        attack_features = torch.cat([perturbed_log_probs, one_hot_labels, losses_for_attack_feature], dim=1)

        # Attack model's forward pass. Its output will depend on 'n'
        attack_output_logits = attack_model_g_prime(attack_features)
        
        # attack_probs will also depend on 'n'
        attack_probs = F.softmax(attack_output_logits, dim=1)[:, 1]
        
        # MemGuard objective: push attack_probs towards 0.5. Minimize the absolute difference.
        # This loss directly depends on 'n', allowing backward() to compute gradients for 'n'.
        loss_memguard = torch.mean(torch.abs(attack_probs - 0.5)) 

        loss_memguard.backward() # Backpropagate to 'n'
        optimizer_n.step() # Update 'n'

        # --- Project 'n' back onto constraints (these steps should NOT record gradients for 'n') ---
        with torch.no_grad(): 
            # 1. Enforce Non-negativity of (original_logits + n)
            # Apply clamp to the current state (original_logits + n.data)
            clamped_perturbed_logits_candidate = torch.clamp(original_logits + n.data, min=0)
            # Update n.data such that original_logits + n.data equals the clamped value
            n.data = clamped_perturbed_logits_candidate - original_logits

            # 2. Enforce L2 Distortion Budget on 'n'
            l2_norm = torch.norm(n.data, p=2, dim=1, keepdim=True) 
            factor = torch.min(torch.ones_like(l2_norm), epsilon_mg / (l2_norm + 1e-9)) 
            n.data = n.data * factor 

            # 3. Enforce Predicted Label Preservation (argmax)
            # Check current argmax based on (original_logits + n.data)
            current_pred_labels = torch.argmax(original_logits + n.data, dim=1) 
            changed_indices = (current_pred_labels != original_pred_labels).nonzero(as_tuple=True)[0]
            
            if len(changed_indices) > 0:
                for idx in changed_indices:
                    # Find the maximum logit among incorrect classes from the current perturbed logits
                    temp_logits_with_n = (original_logits + n.data)[idx].clone() 
                    # Temporarily set original class logit to very low to find max among others
                    temp_logits_with_n[original_pred_labels[idx]] = -float('inf') 
                    max_other_logit_val = torch.max(temp_logits_with_n) if temp_logits_with_n.numel() > 1 else -float('inf')
                    
                    # Calculate required value for the original class logit to make it the max again
                    required_val = max_other_logit_val + 0.01 
                    current_val = (original_logits + n.data)[idx, original_pred_labels[idx]]
                    
                    # Adjust n.data to achieve the required value
                    n.data[idx, original_pred_labels[idx]] += (required_val - current_val)
    
    # Final perturbed logits after optimization and all projections. Detach 'n' for final output.
    final_perturbed_logits = original_logits + n.detach() 

    # One final clamp and argmax check after the loop for robustness, in case any subtle numerical issues
    # or projection orders caused slight violations in the very last step.
    final_perturbed_logits = torch.clamp(final_perturbed_logits, min=0)
    final_pred_labels_check = torch.argmax(final_perturbed_logits, dim=1)
    final_changed_indices_check = (final_pred_labels_check != original_pred_labels).nonzero(as_tuple=True)[0]
    if len(final_changed_indices_check) > 0:
        with torch.no_grad():
            for idx in final_changed_indices_check:
                temp_logits_final = final_perturbed_logits[idx].clone()
                temp_logits_final[original_pred_labels[idx]] = -float('inf') 
                max_other_logit_val_final = torch.max(temp_logits_final) if temp_logits_final.numel() > 1 else -float('inf')
                final_perturbed_logits[idx, original_pred_labels[idx]] = max_other_logit_val_final + 0.01

    return final_perturbed_logits


# ========== LeNet (for Target Model) - UNIFIED to return RAW LOGITS ==========
class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)  
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1) 
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


# ========== Attacker's Shadow Model Definition ==========
class My_Shadow_model(nn.Module):
    def __init__(self):
        super(My_Shadow_model, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 60, 3, stride=1, padding=1) # Modified: 64 -> 60 filters

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
# ========== Attack Model Definition ==========
class AttackModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(21, 256) # Increased hidden layer size
        self.fc2 = nn.Linear(256, 128) # Added another hidden layer
        self.fc3 = nn.Linear(128, 64) # Added another hidden layer
        self.fc4 = nn.Linear(64, 2) # Final output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


def one_hot(y, num_classes=10):
    """Converts integer labels to one-hot encoded vectors."""
    return F.one_hot(y, num_classes).float()


# ==== Feature Collection Function (Used for both Shadow and Target Model) ====
def collect_features_for_attack(model, loader, is_member_flag, add_noise_to_input=False, noise_flip_ratio=0.1, apply_memguard=False, memguard_epsilon=0.1, attack_model_for_memguard=None):
    """
    Collects behavioral features (logits, one-hot labels, loss) from a given model.
    Model is expected to return RAW LOGITS.
    
    Args:
        model (nn.Module): The classification model (shadow or target) to extract features from.
        loader (DataLoader): DataLoader for the input data.
        is_member_flag (int): Label for the attack model (1 for member, 0 for non-member).
        add_noise_to_input (bool): If True, adds pixel flip noise to inputs before feeding to `model`.
        noise_flip_ratio (float): Ratio for pixel flip noise.
        apply_memguard (bool): If True, applies MemGuard-like perturbation to the `model`'s output logits.
        memguard_epsilon (float): L2 distortion budget for MemGuard perturbation.
        attack_model_for_memguard (nn.Module, optional): The defender's trained internal attack model (g').
                                                         Required if `apply_memguard` is True.
    """
    X_features, Y_labels = [], []
    model.eval() 
    
    for x_batch, y_batch in loader: # y_batch are the true labels
        if add_noise_to_input: 
            x_batch = add_pixel_flip_noise(x_batch, flip_ratio=noise_flip_ratio)
        
        # IMPORTANT: Manage no_grad context based on whether MemGuard is applied
        if apply_memguard:
            # MemGuard needs to compute gradients for 'n' internally.
            # raw_logits itself does not need gradients flowing back to 'model' parameters for feature collection,
            # but it needs to be available for 'n' optimization.
            # memguard_perturb_logits handles its own gradient context for 'n'.
            raw_logits = model(x_batch) 
            if attack_model_for_memguard is None:
                raise ValueError("attack_model_for_memguard must be provided when apply_memguard is True.")
            
            processed_logits = memguard_perturb_logits(raw_logits, y_batch, attack_model_for_memguard, 
                                                       MEMGUARD_EPSILON, MEMGUARD_INNER_ITERS, MEMGUARD_LEARNING_RATE)
        else:
            # For baseline scenarios (no MemGuard), we explicitly disable gradients for efficiency
            with torch.no_grad():
                raw_logits = model(x_batch) 
                processed_logits = raw_logits 

        # These calculations should be able to run whether gradients are on/off,
        # but if they depend on 'n', they need gradients to flow back to 'n'.
        # If no MemGuard, they simply don't have requires_grad=True.
        out_log_probs = F.log_softmax(processed_logits, dim=1)

        losses = F.nll_loss(out_log_probs, y_batch, reduction='none').unsqueeze(1) 
        one_hot_labels = one_hot(y_batch, num_classes=10) 

        features_batch = torch.cat([out_log_probs, one_hot_labels, losses], dim=1)

        X_features.append(features_batch)
        Y_labels.extend([is_member_flag] * x_batch.size(0)) 

    return torch.cat(X_features), torch.tensor(Y_labels)


def train_attack_model_func(x, y, class_weights=None):
    """Trains the membership inference attack model."""
    model = AttackModel() 
    loader = DataLoader(TensorDataset(x, y), batch_size=64, shuffle=True) 

    if class_weights is not None:
        weights_tensor = torch.tensor(class_weights, dtype=torch.float)
        loss_fn = nn.CrossEntropyLoss(weight=weights_tensor)
        print(f"[INFO] Using CrossEntropyLoss with class weights: {weights_tensor.tolist()}")
    else:
        loss_fn = nn.CrossEntropyLoss()

    opt = optim.Adam(model.parameters(), lr=0.001) 
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


def evaluate_attack_func(model, x, y, prediction_threshold=0.5):
    """Evaluates the performance of the membership inference attack model (Accuracy, Precision, Recall, AUC)."""
    model.eval() 
    y_true_eval, y_pred_hard_eval, y_probs_eval = [], [], []
    with torch.no_grad(): # For evaluation, always disable gradients
        eval_loader = DataLoader(TensorDataset(x, y), batch_size=64, shuffle=False)
        for x_batch, y_batch in eval_loader:
            output_raw_logits = model(x_batch) 
            probabilities = F.softmax(output_raw_logits, dim=1) 

            preds_hard = (probabilities[:, 1] >= prediction_threshold).long() 
            
            positive_class_probs = probabilities[:, 1] 

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
base_random_seed = int(time.time())
print(f"[INFO] Using base random seed: {base_random_seed} for this run.")
set_individual_shadow_seed(base_random_seed) 


# --- Load Datasets ---
print("[INFO] Loading cifar10 datasets...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
full_cifar_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
test_cifar_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)


# --- Load Pretrained Target Model ---
print("[INFO] Loading pre-trained Target Model...")

# Model: Original overfitted model (for baseline and MemGuard comparison)
target_model_original = Lenet() 
target_model_original.load_state_dict(torch.load("cifar10_overfit_target_model_fixed_v6.pth", map_location=torch.device("cpu")))
target_model_original.eval() 
print("[INFO] Loaded cifar10_overfit_target_model_fixed_v6.pth (Original Baseline)")

# --- Define evaluation related variables early ---
# Indices for target model members
true_target_members_indices = list(range(0, 30))
# Number of members for evaluation, used to calculate required dataset size
eval_member_count = len(true_target_members_indices)


# --- STEP 1: Build and Train Ensemble Shadow Models ---
# This block trains ONE set of shadow models that both attacker and defender will use for training their attack models.
print("\n--- STEP 1: Building and Training Ensemble Shadow Models ---")

# Determine available indices for shadow datasets, ensuring disjointness from target model's 0-29.
available_indices_start = 30 
total_aux_data_per_model = SHADOW_DATA_SIZE_PER_MODEL 
total_aux_data_needed = NUM_SHADOW_MODELS * total_aux_data_per_model

# Ensure we have enough data for all shadow models + a buffer for final evaluation
# We need enough data for target members (30), shadow members (NUM_SHADOW_MODELS * SHADOW_DATA_SIZE_PER_MODEL / 2),
# shadow non-members (NUM_SHADOW_MODELS * SHADOW_DATA_SIZE_PER_MODEL / 2), and eval non-members (30).
# A simpler way: total used data is 30 + NUM_SHADOW_MODELS * SHADOW_DATA_SIZE_PER_MODEL.
# Need to make sure there's enough for final eval non-members which are disjoint from all training data.
min_required_dataset_size = available_indices_start + total_aux_data_needed + eval_member_count 
if len(full_cifar_dataset) < min_required_dataset_size:
    print(f"[ERROR] Not enough auxiliary data in CIFAR10. Required at least {min_required_dataset_size} samples. Current: {len(full_cifar_dataset)}")
    print("Please consider reducing SHADOW_DATA_SIZE_PER_MODEL or NUM_SHADOW_MODELS if this is for a smaller dataset.")
    raise ValueError("Insufficient auxiliary data for shadow model training and evaluation.")

print(f"Training {NUM_SHADOW_MODELS} shadow models.")

# Stores all trained shadow models and their respective train/non-train loaders
trained_shadow_models = []
shadow_data_loaders = [] # List of (train_loader, non_train_loader, train_y, non_train_y)
start_time = time.time()
for i in range(NUM_SHADOW_MODELS):
    set_individual_shadow_seed(base_random_seed + i) 
    
    print(f"\nTraining Shadow Model {i+1}/{NUM_SHADOW_MODELS}...")
    
    # Calculate index range for this shadow model's auxiliary data
    current_aux_data_start_idx = available_indices_start + i * total_aux_data_per_model
    current_aux_data_end_idx = current_aux_data_start_idx + total_aux_data_per_model
    
    clean_current_shadow_aux_subset_data = torch.stack([full_cifar_dataset[j][0] for j in range(current_aux_data_start_idx, current_aux_data_end_idx)])
    clean_current_shadow_aux_subset_labels = torch.tensor([full_cifar_dataset[j][1] for j in range(current_aux_data_start_idx, current_aux_data_end_idx)])

    noisy_current_shadow_aux_data = add_pixel_flip_noise(clean_current_shadow_aux_subset_data, flip_ratio=NOISE_FLIP_RATIO)

    member_split_size = SHADOW_DATA_SIZE_PER_MODEL // 2 
    
    shadow_train_x = noisy_current_shadow_aux_data[:member_split_size]
    shadow_train_y = clean_current_shadow_aux_subset_labels[:member_split_size] 
    
    shadow_non_train_x = noisy_current_shadow_aux_data[member_split_size:]
    shadow_non_train_y = clean_current_shadow_aux_subset_labels[member_split_size:]

    shadow_train_loader = DataLoader(TensorDataset(shadow_train_x, shadow_train_y), batch_size=2, shuffle=True) 
    shadow_non_train_loader = DataLoader(TensorDataset(shadow_non_train_x, shadow_non_train_y), batch_size=2, shuffle=False) 

    shadow_model = My_Shadow_model() 
    shadow_opt = optim.Adadelta(shadow_model.parameters(), lr=1) 
    shadow_scheduler = StepLR(shadow_opt, step_size=1, gamma=0.95) 

    for epoch in range(1, SHADOW_MODEL_EPOCHS + 1):
        shadow_model.train()
        for batch_idx, (data, target) in enumerate(shadow_train_loader):
            shadow_opt.zero_grad()
            output_raw_logits = shadow_model(data) 
            loss = F.nll_loss(F.log_softmax(output_raw_logits, dim=1), target) 
            loss.backward()
            shadow_opt.step()
        shadow_scheduler.step()
        if epoch % 20 == 0:
            print(f"  Shadow Model {i+1} Epoch {epoch}/{SHADOW_MODEL_EPOCHS}, Loss: {loss.item():.4f}")
    print(f"Shadow Model {i+1} training completed.")
    
    trained_shadow_models.append(shadow_model)
    shadow_data_loaders.append((shadow_train_loader, shadow_non_train_loader, shadow_train_y, shadow_non_train_y))
train_end_time = time.time()
total_trained_time = train_end_time - start_time
# --- STEP 2: Extract Behavioral Fingerprints & Train TWO Attack Models ---
# Both attacker and defender use the SAME set of trained shadow models and their outputs.
print("\n--- STEP 2: Extracting Behavioral Fingerprints & Training TWO Attack Models ---")

# --- 2a. Collect features for the ATTACKER'S attack model ---
# The attacker trains its attack model on UNPROTECTED shadow model outputs.
print("\n--- 2a. Collecting features for ATTACKER'S attack model (No MemGuard applied) ---")
attacker_attack_train_X_list = []
attacker_attack_train_Y_list = []

for i, (shadow_model, (train_loader, non_train_loader, _, _)) in enumerate(zip(trained_shadow_models, shadow_data_loaders)):
    current_shadow_member_features, current_shadow_member_attack_labels = \
        collect_features_for_attack(shadow_model, train_loader, 1, 
                                    add_noise_to_input=False, apply_memguard=False) 
    current_shadow_non_member_features, current_shadow_non_member_attack_labels = \
        collect_features_for_attack(shadow_model, non_train_loader, 0, 
                                    add_noise_to_input=False, apply_memguard=False) 
    
    attacker_attack_train_X_list.append(current_shadow_member_features)
    attacker_attack_train_X_list.append(current_shadow_non_member_features)
    attacker_attack_train_Y_list.append(current_shadow_member_attack_labels)
    attacker_attack_train_Y_list.append(current_shadow_non_member_attack_labels)

attacker_attack_train_X = torch.cat(attacker_attack_train_X_list)
attacker_attack_train_Y = torch.cat(attacker_attack_train_Y_list)
print(f"Total features collected for attacker's attack model: {len(attacker_attack_train_X)} samples.")

# --- 2b. Train the ATTACKER'S attack model ---
set_individual_shadow_seed(base_random_seed + NUM_SHADOW_MODELS + 1) # Unique seed for attacker's attack model
member_count_attacker = attacker_attack_train_Y.sum().item()
non_member_count_attacker = len(attacker_attack_train_Y) - member_count_attacker
attacker_attack_class_weights_tensor = torch.tensor([1.0, non_member_count_attacker / member_count_attacker], dtype=torch.float) if member_count_attacker > 0 else None

print("\n--- 2b. Training ATTACKER'S Attack Model ---")
attacker_attack_model = train_attack_model_func(attacker_attack_train_X, attacker_attack_train_Y, class_weights=attacker_attack_class_weights_tensor)
print("[INFO] Attacker's attack model training completed.")


# --- 2c. Collect features for the DEFENDER'S INTERNAL attack model (g') ---
# Defender also uses the same shadow models, and trains its own attack model (g').
# This model will be used by MemGuard's internal optimization. It needs to learn from UNPROTECTED outputs.
print("\n--- 2c. Collecting features for DEFENDER'S INTERNAL attack model (g') (No MemGuard applied) ---")
defender_attack_train_X_list = []
defender_attack_train_Y_list = []

for i, (shadow_model, (train_loader, non_train_loader, _, _)) in enumerate(zip(trained_shadow_models, shadow_data_loaders)):
    current_shadow_member_features, current_shadow_member_attack_labels = \
        collect_features_for_attack(shadow_model, train_loader, 1, 
                                    add_noise_to_input=False, apply_memguard=False) 
    current_shadow_non_member_features, current_shadow_non_member_attack_labels = \
        collect_features_for_attack(shadow_model, non_train_loader, 0, 
                                    add_noise_to_input=False, apply_memguard=False) 
    
    defender_attack_train_X_list.append(current_shadow_member_features)
    defender_attack_train_X_list.append(current_shadow_non_member_features)
    defender_attack_train_Y_list.append(current_shadow_member_attack_labels)
    defender_attack_train_Y_list.append(current_shadow_non_member_attack_labels)

defender_attack_train_X = torch.cat(defender_attack_train_X_list)
defender_attack_train_Y = torch.cat(defender_attack_train_Y_list)
print(f"Total features collected for defender's internal attack model (g'): {len(defender_attack_train_X)} samples.")


# --- 2d. Train the DEFENDER'S INTERNAL attack model (g') ---
set_individual_shadow_seed(base_random_seed + NUM_SHADOW_MODELS + 2) # Unique seed for defender's attack model
member_count_defender = defender_attack_train_Y.sum().item()
non_member_count_defender = len(defender_attack_train_Y) - member_count_defender
defender_attack_class_weights_tensor = torch.tensor([1.0, non_member_count_defender / member_count_defender], dtype=torch.float) if member_count_defender > 0 else None
defender_start_time = time.time()
print("\n--- 2d. Training DEFENDER'S INTERNAL Attack Model (g') ---")
defender_internal_attack_model = train_attack_model_func(defender_attack_train_X, defender_attack_train_Y, class_weights=defender_attack_class_weights_tensor)
print("[INFO] Defender's internal attack model (g') training completed.")
defender_end_time = time.time()
train_defender_time = defender_end_time - defender_start_time
# --- STEP 3: Prepare Evaluation Data for True Target Model ---
print("\n--- STEP 3: Preparing Evaluation Data for True Target Model ---")

true_target_members_subset = Subset(full_cifar_dataset, true_target_members_indices)
true_target_members_loader = DataLoader(true_target_members_subset, batch_size=32, shuffle=False)

eval_non_member_count = eval_member_count 

# Gather all used indices from target training and shadow model training
used_indices = set(range(0, 30)) 
for i in range(NUM_SHADOW_MODELS):
    current_aux_data_start_idx = available_indices_start + i * total_aux_data_per_model
    current_aux_data_end_idx = current_aux_data_start_idx + total_aux_data_per_model
    used_indices.update(range(current_aux_data_start_idx, current_aux_data_end_idx))

all_available_indices = set(range(len(full_cifar_dataset)))
available_non_member_pool_indices = list(all_available_indices - used_indices)

if len(available_non_member_pool_indices) < eval_non_member_count:
    print(f"[WARNING] Not enough available disjoint non-member data for 1:1 evaluation ratio.")
    print(f"  Required: {eval_non_member_count}, Available pool: {len(available_non_member_pool_indices)}")
    eval_non_member_count = len(available_non_member_pool_indices)
    if eval_non_member_count == 0:
        raise ValueError("Not enough data for true target non-members evaluation even after adjusting. Increase full_cifar_dataset size or reduce member count.")

set_individual_shadow_seed(base_random_seed + NUM_SHADOW_MODELS + 3) 
true_target_non_members_indices = random.sample(available_non_member_pool_indices, eval_non_member_count)

# It seems there was an issue with creating DataLoader directly from Subset in previous turn,
# let's extract data and targets explicitly to be safe and consistent.
true_target_non_members_data = torch.stack([full_cifar_dataset[j][0] for j in true_target_non_members_indices])
true_target_non_members_labels = torch.tensor([full_cifar_dataset[j][1] for j in true_target_non_members_indices])
true_target_non_members_loader = DataLoader(TensorDataset(true_target_non_members_data, true_target_non_members_labels),
                                         batch_size=32, shuffle=False)

print(f"Evaluation setup: {len(true_target_members_indices)} members and {len(true_target_non_members_indices)} non-members for 1:1 ratio.")


# =========================================================================
# === Evaluation Scenario 0: ATTACK AGAINST ORIGINAL OVERFITTED MODEL (NO DEFENSE) ===
# =========================================================================
print("\n========== SCENARIO 0: ATTACK AGAINST ORIGINAL OVERFITTED MODEL (NO DEFENSE) ==========")

eval_member_features_original, eval_member_labels_original = \
    collect_features_for_attack(target_model_original, true_target_members_loader, 1, 
                                add_noise_to_input=True, 
                                apply_memguard=False, # No MemGuard
                                attack_model_for_memguard=None) 
print(f"Collected {len(eval_member_features_original)} true target member features (Original Model).")

eval_non_member_features_original, eval_non_member_labels_original = \
    collect_features_for_attack(target_model_original, true_target_non_members_loader, 0, 
                                add_noise_to_input=True, 
                                apply_memguard=False, # No MemGuard
                                attack_model_for_memguard=None) 
print(f"Collected {len(eval_non_member_features_original)} true target non-member features (Original Model).")

final_attack_eval_X_original = torch.cat([eval_member_features_original, eval_non_member_features_original])
final_attack_eval_Y_true_original = torch.cat([eval_member_labels_original, eval_non_member_labels_original])

print("\n--- Evaluation (Original Model, NO Defense) with default threshold (0.5) ---")
evaluate_attack_func(attacker_attack_model, final_attack_eval_X_original, final_attack_eval_Y_true_original, prediction_threshold=0.5)

print("\n--- Evaluation (Original Model, NO Defense) with higher threshold (e.g., 0.8) for better Precision ---")
evaluate_attack_func(attacker_attack_model, final_attack_eval_X_original, final_attack_eval_Y_true_original, prediction_threshold=0.8)

print("\n--- Evaluation (Original Model, NO Defense) with even higher threshold (e.g., 0.95) for even better Precision ---")
evaluate_attack_func(attacker_attack_model, final_attack_eval_X_original, final_attack_eval_Y_true_original, prediction_threshold=0.95)


# =========================================================================
# === Evaluation Scenario 1: ATTACK AGAINST ORIGINAL OVERFITTED MODEL (WITH MEMGUARD DEFENSE) ===
# =========================================================================
# MemGuard will use defender_internal_attack_model (g') for its internal optimization.
print(f"\n\n========== SCENARIO 1: ATTACK AGAINST ORIGINAL OVERFITTED MODEL (WITH MEMGUARD, Epsilon={MEMGUARD_EPSILON:.2f}, Iters={MEMGUARD_INNER_ITERS}) ==========")
evaluate_start_time = time.time()
eval_member_features_original_mg, eval_member_labels_original_mg = \
    collect_features_for_attack(target_model_original, true_target_members_loader, 1, 
                                add_noise_to_input=True, 
                                apply_memguard=True, memguard_epsilon=MEMGUARD_EPSILON,
                                attack_model_for_memguard=defender_internal_attack_model) 
print(f"Collected {len(eval_member_features_original_mg)} true target member features (Original Model + MemGuard).")

eval_non_member_features_original_mg, eval_non_member_labels_original_mg = \
    collect_features_for_attack(target_model_original, true_target_non_members_loader, 0, 
                                add_noise_to_input=True, 
                                apply_memguard=True, memguard_epsilon=MEMGUARD_EPSILON,
                                attack_model_for_memguard=defender_internal_attack_model) 
print(f"Collected {len(eval_non_member_features_original_mg)} true target non-member features (Original Model + MemGuard).")

final_attack_eval_X_original_mg = torch.cat([eval_member_features_original_mg, eval_non_member_features_original_mg])
final_attack_eval_Y_true_original_mg = torch.cat([eval_member_labels_original_mg, eval_non_member_labels_original_mg])

print("\n--- Evaluation (Original Model, WITH MemGuard) with default threshold (0.5) ---")
evaluate_attack_func(attacker_attack_model, final_attack_eval_X_original_mg, final_attack_eval_Y_true_original_mg, prediction_threshold=0.5)

print("\n--- Evaluation (Original Model, WITH MemGuard) with higher threshold (e.g., 0.8) for better Precision ---")
evaluate_attack_func(attacker_attack_model, final_attack_eval_X_original_mg, final_attack_eval_Y_true_original_mg, prediction_threshold=0.8)

print("\n--- Evaluation (Original Model, WITH MemGuard) with even higher threshold (e.g., 0.95) for even better Precision ---")
evaluate_attack_func(attacker_attack_model, final_attack_eval_X_original_mg, final_attack_eval_Y_true_original_mg, prediction_threshold=0.95)
evaluate_end_time = time.time()
evaluate_time = evaluate_start_time - evaluate_end_time
total_time = evaluate_time + total_trained_time + train_defender_time
print("\nTotal defender execution time (training + testing) :{:.2f} seconds".format(total_time))