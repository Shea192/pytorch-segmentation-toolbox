# encoding: utf-8
import os
import sys
import time
import argparse
from collections import OrderedDict, defaultdict

import torch
import torch.utils.model_zoo as model_zoo
import torch.distributed as dist
import numpy as np
from .logger import get_logger
from PIL import Image
import wandb 

logger = get_logger()

# colour map
label_colours = [(0,0,0)
                # 0=background
                ,(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128)
                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                ,(0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0)
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
                # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
                ,(0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)]
                # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor


def reduce_tensor(tensor, dst=0, op=dist.ReduceOp.SUM, world_size=1):
    tensor = tensor.clone()
    dist.reduce(tensor, dst, op)
    if dist.get_rank() == dst:
        tensor.div_(world_size)

    return tensor


def all_reduce_tensor(tensor, op=dist.ReduceOp.SUM, world_size=1, norm=True):
    tensor = tensor.clone()
    dist.all_reduce(tensor, op)
    if norm:
        tensor.div_(world_size)

    return tensor


def load_model(model, model_file, is_restore=False):
    t_start = time.time()
    if isinstance(model_file, str):
        device = torch.device('cpu')
        state_dict = torch.load(model_file, map_location=device)
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']
    else:
        state_dict = model_file
    t_ioend = time.time()

    if is_restore:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = 'module.' + k
            new_state_dict[name] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict, strict=False)
    ckpt_keys = set(state_dict.keys())
    own_keys = set(model.state_dict().keys())
    missing_keys = own_keys - ckpt_keys
    unexpected_keys = ckpt_keys - own_keys

    if len(missing_keys) > 0:
        logger.warning('Missing key(s) in state_dict: {}'.format(
            ', '.join('{}'.format(k) for k in missing_keys)))

    if len(unexpected_keys) > 0:
        logger.warning('Unexpected key(s) in state_dict: {}'.format(
            ', '.join('{}'.format(k) for k in unexpected_keys)))

    del state_dict
    t_end = time.time()
    logger.info(
        "Load model, Time usage:\n\tIO: {}, initialize parameters: {}".format(
            t_ioend - t_start, t_end - t_ioend))

    return model


def parse_devices(input_devices):
    if input_devices.endswith('*'):
        devices = list(range(torch.cuda.device_count()))
        return devices

    devices = []
    for d in input_devices.split(','):
        if '-' in d:
            start_device, end_device = d.split('-')[0], d.split('-')[1]
            assert start_device != ''
            assert end_device != ''
            start_device, end_device = int(start_device), int(end_device)
            assert start_device < end_device
            assert end_device < torch.cuda.device_count()
            for sd in range(start_device, end_device + 1):
                devices.append(sd)
        else:
            device = int(d)
            assert device < torch.cuda.device_count()
            devices.append(device)

    logger.info('using devices {}'.format(
        ', '.join([str(d) for d in devices])))

    return devices


def extant_file(x):
    """
    'Type' for argparse - checks that file exists but does not open.
    """
    if not os.path.exists(x):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist".format(x))
    return x


def link_file(src, target):
    if os.path.isdir(target) or os.path.isfile(target):
        os.remove(target)
    os.system('ln -s {} {}'.format(src, target))


def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def _dbg_interactive(var, value):
    from IPython import embed
    embed()

def decode_labels(mask, num_images=1, num_classes=21):
    """Decode batch of segmentation masks.
    
    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).
    
    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """
    mask = mask.data.cpu().numpy()
    n, h, w = mask.shape
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
      img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
      pixels = img.load()
      for j_, j in enumerate(mask[i, :, :]):
          for k_, k in enumerate(j):
              if k < num_classes:
                  pixels[k_,j_] = label_colours[k]
      outputs[i] = np.array(img)
    return outputs
def decode_edge_logits(preds,num_images=1):

    if isinstance(preds,list):
        vis=[]
        for pred in preds:
            vis=vis+decode_predictions((pred[:num_images].sigmoid()>0.5).squeeze(1).long(),num_images=num_images,num_classes=1)
        return vis 
    else:
        preds=(preds[:num_images].sigmoid()>0.5).squeeze(1).long()
        return decode_predictions(preds,num_images=num_images,num_classes=1)

def decode_edge_labels(edge,num_images=1):
    return decode_labels(edge.squeeze(1),num_images,1)

def decode_logits(preds,num_images=1,num_classes=21):
    if isinstance(preds,list):
        vis=[]
        for pred in preds:
            vis=vis+decode_predictions(torch.max(pred[:num_images],1)[1],num_images=num_images,num_classes=num_classes) #n,h,w
        return vis 
    else:
        preds=torch.max(preds[:num_images],1)[1] #n,h,w
        return decode_predictions(preds,num_images=num_images,num_classes=num_classes)

def decode_predictions(preds, num_images=1, num_classes=21):
    """Decode batch of segmentation masks.
    
    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).
    
    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """
    if isinstance(preds, list):
        preds_list = []
        for pred in preds:
            preds_list.append(pred[-1].data.cpu().numpy())
        preds = np.concatenate(preds_list, axis=0)
    else:
        preds = preds.data.cpu().numpy()

    preds = np.argmax(preds, axis=1)
    n, h, w = preds.shape
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
      img = Image.new('RGB', (len(preds[i, 0]), len(preds[i])))
      pixels = img.load()
      for j_, j in enumerate(preds[i, :, :]):
          for k_, k in enumerate(j):
              if k < num_classes:
                  pixels[k_,j_] = label_colours[k]
      outputs[i] = np.array(img)
    return outputs

def inv_preprocess(imgs, num_images, img_mean):
    """Inverse preprocessing of the batch of images.
       Add the mean vector and convert from BGR to RGB.
       
    Args:
      imgs: batch of input images.
      num_images: number of images to apply the inverse transformations on.
      img_mean: vector of mean colour values.
  
    Returns:
      The batch of the size num_images with the same spatial dimensions as the input.
    """
    imgs = imgs.data.cpu().numpy()
    n, c, h, w = imgs.shape
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((num_images, h, w, c), dtype=np.uint8)
    for i in range(num_images):
        outputs[i] = (np.transpose(imgs[i], (1,2,0)) + img_mean).astype(np.uint8)
    return outputs

def resize(tensor,shape,mode='bilinear'):
    size=[s for s in tensor.size()]
    if len(size)==2:
        size=[1,1]+size
    elif len(size)==3:
        size=[1]+size
    kwargs=dict()
    if mode=='bilinear':
        kwargs['align_corners']=True
    return torch.nn.functional.interpolate(tensor.view(*size),size=shape,mode=mode,**kwargs).squeeze()

labels={
    'citys':{k:v for k,v in zip(range(20),['raod','sidewalk','building','wall','fence','pole','traffic light','traffic sign','vegetation','terrain','sky','person','rider','car','truck','bus','train','motorcycle','bicycle','ignore'])}
}

class Logger:
    def __init__(self,img_mean,img_std=[1,1,1],ignore_index=255,category_set='citys'):
        self.img_std=np.reshape(np.asarray(img_std),(3,1,1))
        self.img_mean=np.reshape(np.asarray(img_mean),(3,1,1))
        self.ignore_index=ignore_index
        self.labels=labels[category_set]
        self.global_step=-1
    def inv_process(self,imgs):
        '''
        imgs: Tensor [c,h,w]
        '''
        return np.transpose((imgs.data.cpu().numpy()*self.img_std + self.img_mean).astype(np.uint8),(1,2,0))

    def visualize(self,imgs,seg_preds,seg_gts=None,edge_preds=None,edge_gts=None,index=0,global_step=-1):

        assert isinstance(imgs,torch.Tensor)
        assert isinstance(seg_preds,(list,torch.Tensor))
        assert (seg_gts is None) or isinstance(seg_gts,torch.Tensor)
        
        if global_step<=self.global_step:
            self.global_step+=1
        else:
            self.global_step=global_step

        imgs=self.inv_process(imgs[index].detach())
        h,w,_=imgs.shape
        
        seg_gts=resize(seg_gts[index].detach().float(),(h,w),'nearest')

        seg_gts[seg_gts==self.ignore_index]=len(labels)-1
        if isinstance(seg_preds,list):
            seg_preds=[resize(pred[index].detach(),(h,w),'bilinear').max(dim=0)[1] for pred in seg_preds]
        elif isinstance(seg_preds,torch.Tensor):
            seg_preds=[resize(seg_preds[index].detach(),(h,w),'bilinear').max(dim=0)[1]]
        else:
            raise TypeError('Unknown type for segmentation prediction : ', type(seg_preds))

        masks=dict()

        for i,pred in enumerate(seg_preds):
            masks['prediction_'+str(i)]={
                'mask_data':seg_preds[i].cpu().numpy(),
                'class_labels':self.labels
            }
        
        masks['ground_truth']={
            'mask_data':seg_gts.long().cpu().numpy(),
            'class_labels':self.labels
        }

        seg_img=wandb.Image(imgs,masks=masks,caption='seg')
        vis=[seg_img]
        if edge_preds is not None:
            assert isinstance(edge_preds,(list,torch.Tensor))
            if isinstance(edge_preds,list):
                edge_preds=[(255*resize(edge_pred[index].detach(),(h,w)).squeeze(0).sigmoid().cpu().numpy()).astype(np.uint8) for edge_pred in edge_preds]
            else:
                edge_preds=[(255*resize(edge_preds[index].detach(),(h,w)).squeeze(0).sigmoid().cpu().numpy()).astype(np.uint8)]
             
            if edge_gts is not None:
                assert isinstance(edge_gts,torch.Tensor) 
                edge_preds.append((resize(edge_gts[index:(index+1)],(h,w)).cpu().numpy()*255).astype(np.uint8)) 
            
            edge_img=wandb.Image(np.concatenate(edge_preds,axis=1)[:,:,None],caption='edge')
            vis.append(edge_img)
        #segmentation
        wandb.log({'image':vis},step=self.global_step) 
