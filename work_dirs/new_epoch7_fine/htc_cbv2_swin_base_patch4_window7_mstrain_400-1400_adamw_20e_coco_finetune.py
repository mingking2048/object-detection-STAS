_base_ = 'htc_cbv2_swin_base_patch4_window7_mstrain_400-1400_adamw_20e_coco.py'
dataset_type = 'CustomDataset'


model = dict(
    roi_head=dict(
        bbox_head=[
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0))
        ]
    )
)


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

mutli_scale_image_size = [(686, 376), (950, 520)]
test_mutli_scale_image_size = [(858, 471), (943, 518), (1000, 565), (1115, 612), (1200, 660)]

'''
train_cfg = dict(  # rpn 和 rcnn 训练超参数的配置
        rpn=dict(  # rpn 的训练配置
            assigner=dict(  # 分配器(assigner)的配置
                type='MaxIoUAssigner',  # 分配器的类型，MaxIoUAssigner 用于许多常见的检测器，更多细节请参考 https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/assigners/max_iou_assigner.py#L10。
                pos_iou_thr=0.7,  # IoU >= 0.7(阈值) 被视为正样本。
                neg_iou_thr=0.3,  # IoU < 0.3(阈值) 被视为负样本。
                min_pos_iou=0.3,  # 将框作为正样本的最小 IoU 阈值。
                match_low_quality=True,  # 是否匹配低质量的框(更多细节见 API 文档).
                ignore_iof_thr=-1),  # 忽略 bbox 的 IoF 阈值。
            sampler=dict(  # 正/负采样器(sampler)的配置
                type='RandomSampler',  # 采样器类型，还支持 PseudoSampler 和其他采样器，更多细节请参考 https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/samplers/random_sampler.py#L8。
                num=256,  # 样本数量。
                pos_fraction=0.5,  # 正样本占总样本的比例。
                neg_pos_ub=-1,  # 基于正样本数量的负样本上限。
                add_gt_as_proposals=False),  # 采样后是否添加 GT 作为 proposal。
            allowed_border=-1,  # 填充有效锚点后允许的边框。
            pos_weight=-1,  # 训练期间正样本的权重。
            debug=False),  # 是否设置调试(debug)模式
        rpn_proposal=dict(  # 在训练期间生成 proposals 的配置
            nms_across_levels=False,  # 是否对跨层的 box 做 NMS。仅适用于 `GARPNHead` ，naive rpn 不支持 nms cross levels。
            nms_pre=2000,  # NMS 前的 box 数
            nms_post=1000,  # NMS 要保留的 box 的数量，只在 GARPNHHead 中起作用。
            max_per_img=1000,  # NMS 后要保留的 box 数量。
            nms=dict( # NMS 的配置
                type='nms',  # NMS 的类别
                iou_threshold=0.7 # NMS 的阈值
                ),
            min_bbox_size=0),  # 允许的最小 box 尺寸
        rcnn=dict(  # roi head 的配置。
            assigner=dict(  # 第二阶段分配器的配置，这与 rpn 中的不同
                type='MaxIoUAssigner',  # 分配器的类型，MaxIoUAssigner 目前用于所有 roi_heads。更多细节请参考 https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/assigners/max_iou_assigner.py#L10。
                pos_iou_thr=0.5,  # IoU >= 0.5(阈值)被认为是正样本。
                neg_iou_thr=0.5,  # IoU < 0.5(阈值)被认为是负样本。
                min_pos_iou=0.5,  # 将 box 作为正样本的最小 IoU 阈值
                match_low_quality=False,  # 是否匹配低质量下的 box(有关更多详细信息，请参阅 API 文档)。
                ignore_iof_thr=-1),  # 忽略 bbox 的 IoF 阈值
            sampler=dict(
                type='RandomSampler',  #采样器的类型，还支持 PseudoSampler 和其他采样器，更多细节请参考 https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/samplers/random_sampler.py#L8。
                num=512,  # 样本数量
                pos_fraction=0.25,  # 正样本占总样本的比例。.
                neg_pos_ub=-1,  # 基于正样本数量的负样本上限。.
                add_gt_as_proposals=True
            ),  # 采样后是否添加 GT 作为 proposal。
            mask_size=28,  # mask 的大小
            pos_weight=-1,  # 训练期间正样本的权重。
            debug=False))  # 是否设置调试模式。
    test_cfg = dict(  # 用于测试 rpn 和 rcnn 超参数的配置
        rpn=dict(  # 测试阶段生成 proposals 的配置
            nms_across_levels=False,  # 是否对跨层的 box 做 NMS。仅适用于`GARPNHead`，naive rpn 不支持做 NMS cross levels。
            nms_pre=1000,  # NMS 前的 box 数
            nms_post=1000,  # NMS 要保留的 box 的数量，只在`GARPNHHead`中起作用。
            max_per_img=1000,  # NMS 后要保留的 box 数量
            nms=dict( # NMS 的配置
                type='nms',  # NMS 的类型
                iou_threshold=0.7 # NMS 阈值
                ),
            min_bbox_size=0),  # box 允许的最小尺寸
        rcnn=dict(  # roi heads 的配置
            score_thr=0.05,  # bbox 的分数阈值
            nms=dict(  # 第二步的 NMS 配置
                type='nms',  # NMS 的类型
                iou_thr=0.5),  # NMS 的阈值
            max_per_img=100,  # 每张图像的最大检测次数
            mask_thr_binary=0.5))  # mask 预处的阈值
'''
albu_train_transforms = [
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0),
            dict(type='FancyPCA', alpha=0.1, always_apply=False, p=1.0), #trick
        ],
        p=0.1),
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=mutli_scale_image_size,
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=test_mutli_scale_image_size,
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]


samples_per_gpu=1
workers_per_gpu=1
optimizer = dict(_delete_=True, type='AdamW', lr=2e-5*(samples_per_gpu/2), betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
runner = dict(type='EpochBasedRunnerAmp', max_epochs=2)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[3, 5])
'''
data = dict(
    samples_per_gpu=2,  # 单个 GPU 的 Batch size
    workers_per_gpu=2,  # 单个 GPU 分配的数据加载线程数
    train=dict(  # 训练数据集配置
        type='CocoDataset',  # 数据集的类别, 更多细节请参考 https://github.com/open-mmlab/mmdetection/blob/master/mmdet/datasets/coco.py#L19。
        ann_file='data/coco/annotations/instances_train2017.json',  # 注释文件路径
        img_prefix='data/coco/train2017/',  # 图片路径前缀
        pipeline=[  # 流程, 这是由之前创建的 train_pipeline 传递的。
            dict(type='LoadImageFromFile'),
            dict(
                type='LoadAnnotations',
                with_bbox=True,
                with_mask=True,
                poly2mask=False),
            dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
        ]),
    val=dict(  # 验证数据集的配置
        type='CocoDataset',
        ann_file='data/coco/annotations/instances_val2017.json',
        img_prefix='data/coco/val2017/',
        pipeline=[  # 由之前创建的 test_pipeline 传递的流程。
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(  # 测试数据集配置，修改测试开发/测试(test-dev/test)提交的 ann_file
        type='CocoDataset',
        ann_file='data/coco/annotations/instances_val2017.json',
        img_prefix='data/coco/val2017/',
        pipeline=[  # 由之前创建的 test_pipeline 传递的流程。
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        samples_per_gpu=2  # 单个 GPU 测试时的 Batch size
        ))
'''
classes = ('stas',)
workflow = [('train', 1), ('val', 1)]
data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=workers_per_gpu,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file='./data/OBJ_Train_Datasets/pkl/train.pkl',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file='./data/OBJ_Train_Datasets/pkl/train.pkl',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file='./data/OBJ_Train_Datasets/pkl/test.pkl',
        pipeline=test_pipeline))

evaluation = dict(metric=['mAP'])

load_from = './work_dirs/new_epoch7/latest.pth'
