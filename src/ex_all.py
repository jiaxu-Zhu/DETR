"""check all api"""
import pickle
import numpy as np
from numpy import dtype

from models import detr
from models import matcher
from mindspore.common.tensor import Tensor
from mindspore import Model
from mindvision.engine.callback import LossMonitor
import mindspore.nn as nn
from mindspore import context, ops
from mindspore import load_checkpoint, load_param_into_net

from datasets import coco, cocopanoptic
from models.position_encoding import PositionEmbeddingSine
from models.backbone import build_backbone
from models.resnet import resnet50, resnet101
from models.transformer import MultiHeadAttention, TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, TransformerDecoder, build_transformer
from models.detr import bulid_detr
from models.segmentation import DETRsegm
from models.matcher import build_matcher, build_criterion, box_cxcywh_to_xyxy, box_iou, generalized_box_iou

context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")  # 设置为动态图，静态图有比较多约束

def ex_coco():
    """check coco api"""
    print(__doc__)
    dataset = coco.build(img_set='val', batch=1, shuffle=False)
    n = 0
    for d in dataset.create_dict_iterator():
        for k, v in d.items():
            print(n, k, v.shape)
            if 'size' in k:
                print(v)
        n += 1
        if n > 1:
            break


def ex_coco_pano():
    """check coco panoptic api"""
    print(__doc__)
    dataset = cocopanoptic.build(img_set='val', batch=2, shuffle=False)
    n = 0
    for d in dataset.create_dict_iterator():
        for k, v in d.items():
            print(n, k, v.shape)
        n += 1
        if n > 10:
            break


def ex_postion_encoding():
    """check postion_encoding"""
    p = PositionEmbeddingSine()
    print(p)
    m = np.ones((2, 3, 4)).astype(np.float32)
    m = Tensor(m)
    print(m)
    print(type(m))
    print(p(m).shape)  # (2, 256, 3, 4)


def ex_backbone():
    # 加载数据
    with open('./sample_np.pkl', 'rb') as f:
        s = pickle.load(f)
    x = Tensor(s['img'].astype(np.float32))  # 图片输入，shape：B*3*H*W (2, 3, 800, 994)
    mask = Tensor(s['mask'].astype(np.float32))  # mask输入 shape：B*H*W (2, 800, 994)
    net = build_backbone(resnet='resnet50', return_interm_layers=True, is_dilation=False)
    x, mask, pos = net(x, mask)  # 输出为特征图shape：B*256*H/32*W/32，mask shape：B*H/32*W/32，位置编码shape：B*256*H/32*W/32
    print('x:', x[-1][0][0][0])


def ex_resnet():
    """check resnet"""
    # 加载数据
    with open('./sample_np.pkl', 'rb') as f:
        s = pickle.load(f)
    x = Tensor(s['img'].astype(np.float32))  # 图片输入，shape：B*3*H*W (2, 3, 800, 994)
    print('x:', x.shape)
    # mask = Tensor(s['mask'].astype(np.float32)) # mask输入 shape：B*H*W (2, 800, 994)
    net = resnet50(return_interm_layers=True, is_dilation=True)
    x = net(x)  # 输出为特征图shape：B*256*H/32*W/32，mask shape：B*H/32*W/32，位置编码shape：B*256*H/32*W/32
    print('x:', x[-1].shape, x[-1][0][0][0])


def ex_multhead():
    k = v = Tensor(np.ones((20, 2, 256)).astype(np.float32))  # shape: L*B*C
    q = Tensor(np.ones((30, 2, 256)).astype(np.float32))  # shape: L*B*C
    mha = MultiHeadAttention(n_head=8, d_model=256)
    mask = Tensor(np.ones((2, k.shape[0])).astype(np.float32))  # shape: B*L 这里的L和k、v的长度相同
    output, attn = mha(q, k, v, mask)
    # output shape: (30, 2, 256) attn shape: (2, 8, 30, 20)
    print('output shape:', output.shape, 'attn shape:', attn.shape)


def ex_encoder_layer():
    # 验证编码器层类
    tel = TransformerEncoderLayer()  # 编码器的输入有三个：src=(WH,N,256) src_mask= (N,WH) pos_embed= (WH,N,256) 注释：W=W/32,H=H/32
    src = Tensor(np.ones((10, 2, 256)).astype(np.float32))
    src_mask = Tensor(np.ones((2, 10)).astype(np.float32))
    pos = Tensor(np.ones((10, 2, 256)).astype(np.float32))
    out = tel(src=src, src_padding_mask=src_mask, pos=pos)
    print('out shape:', out.shape)  # out shape: (10, 2, 256)


def ex_encoder():
    # 验证编码器类
    tel = TransformerEncoderLayer()  # 编码器的输入有三个：src=(WH,N,256) src_mask= (N,WH) pos_embed= (WH,N,256) 注释：W=W/32,H=H/32
    src = Tensor(np.ones((10, 2, 256)).astype(np.float32))
    src_mask = Tensor(np.ones((2, 10)).astype(np.float32))
    pos = Tensor(np.ones((10, 2, 256)).astype(np.float32))
    encoder = TransformerEncoder(encoder_layer=tel, num_layers=2, norm=None)
    out = encoder(src=src, src_padding_mask=src_mask, pos=pos)
    print('out shape:', out.shape)  # out shape: (10, 2, 256)


def ex_decoder_layer():
    # 解码器层类的输入 有五个参数 decoder tgt=(100,N,256) memory=(WH,N,256),mask=(N,WH) pos_embed=(WH,N,256) query_pos=(100,N,256)
    src = Tensor(np.ones((10, 2, 256)).astype(np.float32))
    src_mask = Tensor(np.ones((2, 10)).astype(np.float32))
    pos = Tensor(np.ones((10, 2, 256)).astype(np.float32))
    tgt = Tensor(np.ones((100, 2, 256)).astype(np.float32))
    query_pos = Tensor(np.ones((100, 2, 256)).astype(np.float32))
    tdl = TransformerDecoderLayer(d_model=256, nhead=8, dim_feedforward=2048,
                                  dropout=0.1, activation="relu")
    out = tdl(tgt, src, memory_padding_mask=src_mask, pos=pos, query_pos=query_pos)
    print('out shape:', out.shape)  # out shape: (100, 2, 256)


def ex_decoder():
    # 验证解码器
    src = Tensor(np.ones((10, 2, 256)).astype(np.float32))
    src_mask = Tensor(np.ones((2, 10)).astype(np.float32))
    pos = Tensor(np.ones((10, 2, 256)).astype(np.float32))
    tgt = Tensor(np.ones((100, 2, 256)).astype(np.float32))
    query_pos = Tensor(np.ones((100, 2, 256)).astype(np.float32))
    tdl = TransformerDecoderLayer(d_model=256, nhead=8, dim_feedforward=2048,
                                  dropout=0.1, activation="relu")
    decoder = TransformerDecoder(decoder_layer=tdl, num_layers=2, norm=None, return_intermediate=True)
    hs = decoder(tgt, src, memory_padding_mask=src_mask, pos=pos,
                 query_pos=query_pos)  # 输出 hs=(decoder_layers, 100, N, 256)
    print('hs shape:', hs.shape)  # hs shape: (1, 100, 2, 256)


def ex_transformer():
    # transormer src=(N,256,W/32,H/32)-> (WH,N,256) pos_embed=(N,256,W,H)-> (WH,N,256) query_embed=(100,256) -> (100,N,256) mask=(N,W,H) -> (N,WH)
    # tf =  Transformer(d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=True,return_intermediate_dec=True)
    tf = build_transformer()
    src = Tensor(np.ones((2, 256, 2, 5)).astype(np.float32))
    mask = Tensor(np.zeros((2, 2, 5)).astype(np.float32))
    pos = Tensor(np.ones((2, 256, 2, 5)).astype(np.float32))
    query_embed = Tensor(np.ones((100, 256)).astype(np.float32))
    hs, memory = tf(src=src, mask=mask, query_embed=query_embed, pos_embed=pos)
    # hs shape: (6, 2, 100, 256) memory shape: (2, 256, 2, 5)
    print('hs shape:', hs.shape, 'memory shape:', memory.shape)


def ex_detr():
    #反序列化，利用pickle.load直接反序列化一个file-like object
    with open('sample_np.pkl', 'rb') as f:
        s = pickle.load(f)
    x = Tensor(s['img'].astype(np.float32)) # 图片输入，shape：B*3*H*W
    mask = Tensor(s['mask'].astype(np.float32)) # mask输入 shape：B*H*W
    # print('x:', x.shape, 'mask:', mask.shape) # x: (2, 3, 800, 994) mask: (2, 800, 994)
    # backbone = build_backbone()  # 生成第一部分模型backbone
    # tf = build_transformer() # 生成第二部分模型transformer
    # net = DETR(backbone, tf, num_classes=90, num_queries=100, aux_loss=True,)  # 类别数 # 边框数
    net = bulid_detr(return_interm_layers=True)
    out = net(x, mask)
    for k, v in out.items():
        print(k, v.shape) # pred_logits (2, 100, 92) pred_boxes (2, 100, 4)
    print(out.keys()) # dict_keys(['pred_logits', 'pred_boxes'])


def ex_segmentation():
    """
    outputs:
        pred_logits (2, 100, 251)
        pred_boxes (2, 100, 4)
        pred_masks (2, 100, 267, 301)
    """
    x = Tensor(np.ones((2, 3, 1066, 1201)).astype(np.float32)) # 模拟图片输入，shape：B*3*H*W
    mask = Tensor(np.zeros((2, 1066, 1201)).astype(np.float32)) # 模拟mask输入 shape：B*H*W
    net = bulid_detr(num_classes=250, return_interm_layers=True)
    model = DETRsegm(net, freeze_detr=True)
    out = model(x, mask)
    for k, v in out.items():
        print(k, v.shape)


def pkl_read():
    # 反序列化，利用pickle.load直接反序列化一个file-like object
    with open('sample_coco_np.pkl', 'rb') as f:  # 获取样本里的标签数据
        s = pickle.load(f)
    target = dict()
    print('========= target ==========')
    for k, v in s['target'][-1].items():
        target[k] = Tensor(v)
        print(k, v.shape)
    # boxes (8, 4)
    # labels (8,)
    # image_id (1,)
    # area (8,)
    # iscrowd (8,)
    # orig_size (2,)
    # size (2,)
    # 反序列化，利用pickle.load直接反序列化一个file-like object
    with open('output_np.pkl', 'rb') as f:  # 获取模型输出的结果
        output = pickle.load(f)
    print('========= output ==========')
    print(output.keys())  # dict_keys(['pred_logits', 'pred_boxes', 'aux_outputs'])
    # pred_logits: torch.Size([2, 100, 92])])
    print('pred_logits:', output['pred_logits'].shape, type(output['pred_logits']))
    print('pred_boxes:', output['pred_boxes'].shape)  # pred_boxes: torch.Size([2, 100, 4])
    print('aux_outputs len:', len(output['aux_outputs']))  # aux_outputs len: 5; 长度为5的列表，解码器中间输出结果
    output['pred_logits'] = Tensor(output['pred_logits'])
    output['pred_boxes'] = Tensor(output['pred_boxes'])
    # for i in output['aux_outputs']:
    #     for k, v in i.items():
    #         print(k, v.shape, type(v))

    print('size:', target['size'], 'orig_size:', target['orig_size'])
    print('boxes:', target['boxes'], type(target['boxes']))

    return output, [target]


def ex_iou():
    box1 = Tensor(np.array([[0.5205, 0.6888, 0.9556, 0.5955],
        [0.2635, 0.2472, 0.4989, 0.4764],
        [0.3629, 0.7329, 0.4941, 0.5106],
        [0.6606, 0.4189, 0.6789, 0.7815],
        [0.3532, 0.1326, 0.1180, 0.0969],
        [0.2269, 0.1298, 0.0907, 0.0972],
        [0.3317, 0.2269, 0.1313, 0.1469]]).astype(np.float32))
    box2 = Tensor(np.array([[0.3532, 0.1326, 0.1180, 0.0969],
        [0.2269, 0.1298, 0.0907, 0.0972],
        [0.3317, 0.2269, 0.1313, 0.1469]]).astype(np.float32))
    b1 = box_cxcywh_to_xyxy(box1)
    b2 = box_cxcywh_to_xyxy(box2)
    print('b1:', b1.shape, 'b2:', b2.shape)
    iou, union = box_iou(b1, b2)
    print('iou:', iou.shape, 'union:', union.shape)
    giou = generalized_box_iou(b1, b2)
    print('giou:', giou.shape)


def ex_matcher():
    hm = build_matcher()
    print(hm)
    output, target = pkl_read()
    indices = hm(output, target)
    print(indices)  # [(Tensor(shape=[8], dtype=Int64, value= [ 4, 43, 61, 63, 71, 78, 81, 98]), Tensor(shape=[8], dtype=Int64, value= [7, 2, 1, 5, 0, 4, 6, 3]))]
    for i, (src, _) in enumerate(indices):
        print(i, src, _)


def ex_criterion():
    with open('loss_input.pkl', 'rb') as f:
        s = pickle.load(f)
    outputs = {k:Tensor(v) for k, v in s['out'].items()}
    targets = s['tgts']
    # 损失函数
    criterion = build_criterion(is_segmentation=True)
    outs = criterion(outputs, targets)
    for k, v in outs.items():
        print(k, v)


def ex_one_sample():
    is_segmentation = False
    # 反序列化，利用pickle.load直接反序列化一个file-like object
    with open('sample_coco_np.pkl', 'rb') as f:
        s = pickle.load(f)
    x = Tensor(s['img'].astype(np.float32))  # 图片输入，shape：B*3*H*W
    mask = Tensor(s['mask'].astype(np.float32))  # mask输入 shape：B*H*W
    print('x:', x.shape, 'mask:', mask.shape)  # x: (2, 3, 800, 994) mask: (2, 800, 994)
    # 标签
    target = [{k: Tensor(v) for k, v in t.items()} for t in s['target']]
    for t in target:
        for k, v in t.items():
            print(k, v.shape)
        break

    # 构建模型
    print('构建detr网络......')
    if is_segmentation:  # 判断是否为实例分割
        net = detr.bulid_detr(resnet='resnet50', return_interm_layers=is_segmentation,
                              num_classes=250, is_dilation=False)
        net = DETRsegm(net, freeze_detr=False)
    else:
        net = detr.bulid_detr(resnet='resnet50', return_interm_layers=False, num_classes=91, is_dilation=False)
    param_dict = load_checkpoint('./detr/resume/resnet50.ckpt')
    load_param_into_net(net, param_dict)
    net.set_train(False)
    # 数据输入网络
    out = net(x, mask)
    for k, v in out.items():
        print(k, v.shape, v[0][0])  # pred_logits (2, 100, 92) pred_boxes (2, 100, 4)
    print(out.keys())  # dict_keys(['pred_logits', 'pred_boxes'])


if __name__ == '__main__':
    ex_coco()
