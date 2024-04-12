_base_ = [
    '_base_/models/fcn_unet_s5-d16.py', '_base_/datasets/gg_hack_dataset.py',
    '_base_/default_runtime.py', '_base_/schedules/schedule_20k.py'
]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=10000, val_interval=1000)
crop_size = (128, 128)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    test_cfg=dict(crop_size=(128, 128), stride=(85, 85)))
