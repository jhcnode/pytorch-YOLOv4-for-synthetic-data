from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
from gen_anchor import read_anchor
from models import Yolov4
import argparse
import numpy as np
use_cuda = True


def detect_cv2(namesfile,anchors,weightfile,imgfile):
    import cv2
    class_names = load_class_names(namesfile)
    n_classes=len(class_names)	
    m = Yolov4(anchors=anchors,yolov4conv137weight=None, n_classes=n_classes, inference=True)
    pretrained_dict = torch.load(weightfile, map_location=torch.device('cuda'))
    m.load_state_dict(pretrained_dict)

    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()


    img = cv2.imread(imgfile)
    sized = cv2.resize(img, (416, 416))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

    plot_boxes_cv2(img, boxes[0], savename='predictions.jpg', class_names=class_names)


def detect_cv2_camera(namesfile,anchors,weightfile,size=416):

    import cv2
    anchors=np.array(anchors)*size
    anchors=anchors.tolist()    
    class_names = load_class_names(namesfile)
    n_classes=len(class_names)	
    m = Yolov4(anchors=anchors,yolov4conv137weight=None, n_classes=n_classes, inference=True)

    pretrained_dict = torch.load(weightfile, map_location=torch.device('cuda'))
    m.load_state_dict(pretrained_dict)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("../dataset/sample1.mov")
    cap.set(3, 1280)
    cap.set(4, 720)
    print("Starting the YOLO loop...")

    while True:
        ret, img = cap.read()
        if(ret==False): continue
        sized = cv2.resize(img, (size,size))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        start = time.time()
        boxes = do_detect(m, sized, 0.2, 0.6, use_cuda)
        finish = time.time()
        print('Predicted in %f seconds.' % (finish - start))

        result_img = plot_boxes_cv2(img, boxes[0], savename=None, class_names=class_names)

        cv2.imshow('Yolo demo', result_img)
        cv2.waitKey(1)

    cap.release()


def detect_skimage(namesfile,anchors,weightfile, imgfile):
    from skimage import io
    from skimage.transform import resize
    class_names = load_class_names(namesfile)
    n_classes=len(class_names)
    m = Yolov4(anchors=anchors,yolov4conv137weight=None, n_classes=n_classes, inference=True)
    pretrained_dict = torch.load(weightfile, map_location=torch.device('cuda'))
    m.load_state_dict(pretrained_dict)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    img = io.imread(imgfile)
    sized = resize(img, (m.width, m.height)) * 255

    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.4, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

    plot_boxes_cv2(img, boxes, savename='predictions.jpg', class_names=class_names)


def get_args():
    parser = argparse.ArgumentParser('Test your image or video by trained model.')
    parser.add_argument('-weightfile', type=str,
                        default='./checkpoints/Yolov4_epoch199.pth',
                        help='path of trained model.', dest='weightfile')
    parser.add_argument('-imgfile', type=str,
                        default=None,
                        help='path of your image file.', dest='imgfile')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    anchors=read_anchor("./data/anchor.txt")
    namesfile="./data/x.names"

    if args.imgfile:
        detect_cv2(namesfile,anchors,args.weightfile, args.imgfile)
        # detect_cv2(namesfile,anchors,args.weightfile, args.imgfile)
        # detect_skimage(namesfile,anchors,args.weightfile, args.imgfile)
    else:
        detect_cv2_camera(namesfile,anchors,args.weightfile)
