import numpy as np
import os
import cv2
from operator import itemgetter, attrgetter

def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: average IoU as a single float
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])
    
def iou(box, clusters):
    """
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    :param box: tuple or array, shifted to the origin (i. e. width and height)
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: numpy array of shape (k, 0) where k is the number of clusters
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    # if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        # raise ValueError("Box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection)

    return iou_
    
def kmeans(boxes, k, dist=np.median):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param k: number of clusters
    :param dist: distance function
    :return: numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed()

    # the Forgy method will fail if the whole array contains the same rows
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)

        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters

def read_anchor(path):
    anchor_dir = os.path.join(path, "anchor.txt")
    anchors=[]
    if os.path.exists(anchor_dir)==True:
        with open(anchor_dir) as f:
            contents = f.readlines()
            for c in contents:
                c=c.split('\n')[0]
                c=c.replace("[","")
                c=c.replace("]","")
                c=c.split(',')
                c=[float(x) for x in c]
                anchors.append(c)
    return anchors
def write_anchor(path,anchor):
    f = open(os.path.join(path,"anchor.txt"),"w")
    for an in anchor:
        contents="{}\n".format(an)
        f.write(contents)
    f.close()
	
def calculate_anchor_image(labels,num_ab,num_classes):
	labels=labels[:1]
	boxes=[]
	max_w=0
	max_h=0
	min_w=0
	min_h=0	
	isinit=False
	for category in labels:
		for c in category.keys():
			for i,im_dir in enumerate(category[c]):
				image = cv2.imread(im_dir)
				if(image is None):
					continue  	
				h,w,_=image.shape
				if(isinit==False):
					min_w=w
					min_h=h	
					isinit=True
				max_w=max_w if max_w>w else w
				max_h=max_h if max_h>h else h
				min_w=min_w if min_w<w else w
				min_h=min_h if min_h<h else h		
				
				
	for category in labels:
		for c in category.keys():
			for im_dir in category[c]:
				image = cv2.imread(im_dir)
				if(image is None):
					continue  
				h,w,_=image.shape					
				w=(w-min_w)/(max_w-min_w)
				h=(h-min_h)/(max_h-min_h) 
				if np.count_nonzero(w == 0) > 0 or np.count_nonzero(h == 0) > 0:
					continue
				boxes.append([w,h]) 

	anchors=kmeans(np.array(boxes), k=num_ab)
	sort_anchor=[]
	for a in anchors:
		sort_anchor.append((a,a[0]*a[1]))

	sort_anchor = sorted(sort_anchor, key=itemgetter(1), reverse=False)
	anchors=[]
	for a in sort_anchor:
		anchors.append(a[0])
	anchors=np.array(anchors)
	print("Accuracy: {:.2f}%".format(avg_iou(np.array(boxes), anchors) * 100))
	
	anchors=anchors.tolist()
	return anchors
	
	
def calculate_anchor_boxes(labels,num_ab,num_classes,image_size):
	boxes=[]
	for category in labels:
		for c in category.keys():
			for im_dir in category[c]:
				image = cv2.imread(im_dir)
				if(image is None):
					continue
				h_ratio = 1.0 * image_size[0] / image.shape[0]
				w_ratio = 1.0 * image_size[1] / image.shape[1]
				image = cv2.resize(image, (image_size[1], image_size[0]))
				if len(image.shape) == 2:
					im = np.expand_dims(im, 2)
					im = np.concatenate([im, im, im], -1)    
				w=float(data['width'])*w_ratio
				h=float(data['height'])*h_ratio
				x_min=float(data['xmin'])*w_ratio
				x_max=float(data['xmax'])*w_ratio
				y_min=float(data['ymin'])*h_ratio
				y_max=float(data['ymax'])*h_ratio
				# normalize object coordinates and clip the values
				x_min, y_min, x_max, y_max = x_min / image_size[0], y_min / image_size[1], x_max / image_size[0], y_max / image_size[1]
				x_min, y_min, x_max, y_max = np.clip([x_min, y_min, x_max, y_max], 0, 1)        
				w,h=[x_max - x_min, y_max - y_min]
				if np.count_nonzero(w == 0) > 0 or np.count_nonzero(h == 0) > 0:
					continue
				boxes.append([w,h])                
	anchors=kmeans(np.array(boxes), k=num_ab)
	print("Accuracy: {:.2f}%".format(avg_iou(np.array(boxes), anchors) * 100))
	return anchors