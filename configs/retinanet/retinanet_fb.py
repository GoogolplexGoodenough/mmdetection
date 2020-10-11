_base_ =[
    '../_base_/models/retinanet_fb.py',
    '../_base_/datasets/voc0712.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(pretrained='./Imagenet-pretrained/searched_backbone.pth')
