seed = 42
batch_size = 50
end_epoch = 200
init_lr = 0.001
lr_milestones = [15, 30, 45, 60]
lr_decay_rate = 0.1
weight_decay = 1e-5
input_size = 512
    
# root = r'W:\breast\Classification\dataset'
# model_path = r'W:\breast\Classification\checkpoint'
# dataset_path = r'W:\breast\data\original_crop_dataset'

root = r'W:\breast\NgocToan\Cls\dataset'
# model_path = r'W:\breast\NgocToan\Cls\checkpoint'
model_path = r'W:\breast\NgocToan\Cls\checkpoint_CMMD'
# model_path = ''

#VinDr
# dataset_path = r'W:\breast\data\crop800_1245'
#CMMD
dataset_path = r'W:\breast\data\CMMD'
