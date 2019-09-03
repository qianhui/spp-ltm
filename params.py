"""Params for ADDA."""
import torch
# params for dataset and data loader
data_root = "data"
dataset_mean_value = 0.5
dataset_std_value = 0.5
dataset_mean = (dataset_mean_value, )
dataset_std = (dataset_std_value, )
batch_size = 50
image_size = 64
test_batch_size = 1000

# params for source dataset
src_dataset = "SVHN"
src_model_trained = True

# params for target dataset
tgt_dataset = "MNIST"
tgt_model_trained = True

# params for setting up models
model_root = "snapshots"
d_input_dims = 500
d_hidden_dims = 500
d_output_dims = 2

# params for training network
num_gpu = 1
cuda = True
device = torch.device("cuda" if cuda else "cpu")
num_epochs_pre = 30
log_step_pre = 20
eval_step_pre = 20
save_step_pre = 100
num_epochs = 100
log_step = 100
save_step = 100
manual_seed = 1
log_interval = 100

# params for optimizing models
d_learning_rate = 1e-4
# c_learning_rate = 1e-4
c_learning_rate = 1e-4
beta1 = 0.5
beta2 = 0.9
lr_gamma = 0.97
clamp_lower = -0.001
clamp_upper = 0.001
