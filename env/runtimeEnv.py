from utils.pathHandler import HRankPathGather

# get from json
# get from arg-parser

### 
# Static Env
###

gpu = [0, 1]
train_limit = 5
valid_limit = 3
limit = 5 
arch = "vgg16"

# path
datasets = 'res/datasets'
ranks = 'res/milestone/ranks'
vgg_model = 'res/checkpoint/vgg'

# dataloader
num_slices = 100
data_per_client_epoch = 100
client_per_round = 10
workers = 10

# compress rate
compress_rate = [0.]*100
candidate_rate = [0.45]*7 + [0.78]*5

# PruningFL

### 
# Dynamic Env
###

# Path
file_repo = HRankPathGather(vgg_model, datasets, ranks)
