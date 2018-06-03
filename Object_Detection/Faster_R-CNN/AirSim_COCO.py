# Faster R-CNN
# Original implemented by Ross Girshick

# Modified to use COCO dataset model and suit the project
# Jiahuan He

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

## COCO dataset classes
CLASSES = ('background',
            'person', 'bicycle', 'car','motorcycle','airplane','bus','train', 'truck', 'boat','traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench','bird','cat','dog','horse','sheep',
            'cow', 'elephant','bear','zebra','giraffe','hat','umbrella', 'handbag','tie','suitcase',
            'frisbee','skis','snowboard','sports ball','kite', 'baseball bat','baseball glove','skateboard','surfboard','tennis racket',
            'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich',
            'orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant',
            'bed','dining table','window','tv','laptop','mouse','remote','keyboard','cell phone','microwave',
            'oven', 'sink','refrigerator','blender','book','clock','vase','scissors','teddy bear','hair drier','tooth brush')


## PASCAL VOC 2007 dataset classes
#CLASSES = ('__background__',
#           'aeroplane', 'bicycle', 'bird', 'boat',
#           'bottle', 'bus', 'car', 'cat', 'chair',
#           'cow', 'diningtable', 'dog', 'horse',
#           'motorbike', 'person', 'pottedplant',
#           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}


def vis_detections(im, class_name, dets, ax, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=2.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    im_file = os.path.join(cfg.DATA_DIR, 'AirSim_sample_input', image_name)
    im = cv2.imread(im_file)

    out_file = os.path.join(cfg.DATA_DIR, 'output', image_name)
    cv2.imwrite(out_file,im)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.5 # threshold of confidence
    NMS_THRESH = 0.3


    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(500.0/96, 375.0/96))
    ax.imshow(im, aspect='equal')
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]

        vis_detections(im, cls, dets, ax,thresh=CONF_THRESH)
        

    plt.axis('off')
    #plt.tight_layout()
    plt.draw()
    outfile = os.path.join(cfg.DATA_DIR, 'output', image_name)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())  
    plt.gca().yaxis.set_major_locator(plt.NullLocator())  
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)  
    plt.margins(0,0)  
    fig.savefig(outfile, transparent=True, dpi=96, pad_inches = 0)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    ## COCO dataset setting
    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0], 'faster_rcnn_alt_opt', 'coco_test.pt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models/coco_vgg16_faster_rcnn_final.caffemodel')


    ## PASCAL VOC 2007 dataset setting
    #prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
    #                        'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    #caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
    #                          NETS[args.demo_net][1])
    

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    im_names = []
    for i in range(0,105):
        image = str(i) + ".jpg"
        im_names.append(image)


    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Object Detection for data/AirSim_sample_input/{}'.format(im_name)
        demo(net, im_name)
