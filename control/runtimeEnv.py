from control.pathHandler import HRankPathGather

# get from json
# get from arg-parser

### 
# Static Env
###

gpu = [0]
limit = 5 
arch = "vgg16"

# path
datasets = 'localSet'
ranks = 'ranks'
vgg_model = 'results/vgg'

# dataloader
num_slices = 100
data_per_client_epoch = 100
client_per_round = 10
workers = 10

# pruning rate
compress_rate=[0.]*100

# PruningFL

### 
# Dynamic Env
###

# Path
file_repo = HRankPathGather(vgg_model, datasets, ranks)