from utils.pathHandler import HRankPathGather

# get from json
# get from arg-parser

### 
# Static Env
###

gpu = [0, 1]
train_limit = 1000
union_train_limit = 5
batch_size = 32

valid_limit = 3
limit = 5 
arch = "vgg16"

# path
datasets = 'res/datasets'
ranks = 'res/milestone/ranks'
vgg_model = 'res/checkpoint/vgg'
resnet_model = 'res/checkpoint/resnet'
images = 'res/images'

# dataloader
num_slices = 100
data_per_client_epoch = union_train_limit * batch_size
client_per_round = 10
workers = 10

# compress rate
compress_rate = [0.]*100
candidate_rate = [0.45]*7 + [0.78]*5

# vgg16_bn
vgg16_candidate_rate = [0.45]*7 + [0.78]*5

# resnet50
resnet50_candidate_rate = [0.1] + [0.2]*3 + [0.5]*16

# resnet56
resnet56_candidate_rate = [0.] + [0.18]*29

# mobilenetv2
mobile_candidate_rate = [0.] + [0.3]*7

# PruningFL

### 
# Dynamic Env
###

# Path
file_repo = HRankPathGather(resnet_model, datasets, images)

