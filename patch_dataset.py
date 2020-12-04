import os
import random
import sys

import cv2
import numpy as np

import torch
from torch.utils.data.dataset import Dataset
from operator import itemgetter, attrgetter



def rand_uniform_strong(min, max):
	if min > max:
		swap = min
		min = max
		max = swap
	return random.random() * (max - min) + min


def rand_scale(s):
	scale = rand_uniform_strong(1, s)
	if random.randint(0, 1) % 2:
		return scale
	return 1. / scale


def rand_precalc_random(min, max, random_part):
	if max < min:
		swap = min
		min = max
		max = swap
	return (random_part * (max - min)) + min


def fill_truth_detection(bboxes, num_boxes, classes, flip, dx, dy, sx, sy, net_w, net_h):
	if bboxes.shape[0] == 0:
		return bboxes, 10000
	np.random.shuffle(bboxes)
	bboxes[:, 0] -= dx
	bboxes[:, 2] -= dx
	bboxes[:, 1] -= dy
	bboxes[:, 3] -= dy

	bboxes[:, 0] = np.clip(bboxes[:, 0], 0, sx)
	bboxes[:, 2] = np.clip(bboxes[:, 2], 0, sx)

	bboxes[:, 1] = np.clip(bboxes[:, 1], 0, sy)
	bboxes[:, 3] = np.clip(bboxes[:, 3], 0, sy)

	out_box = list(np.where(((bboxes[:, 1] == sy) & (bboxes[:, 3] == sy)) |
							((bboxes[:, 0] == sx) & (bboxes[:, 2] == sx)) |
							((bboxes[:, 1] == 0) & (bboxes[:, 3] == 0)) |
							((bboxes[:, 0] == 0) & (bboxes[:, 2] == 0)))[0])
	list_box = list(range(bboxes.shape[0]))
	for i in out_box:
		list_box.remove(i)
	bboxes = bboxes[list_box]

	if bboxes.shape[0] == 0:
		return bboxes, 10000

	bboxes = bboxes[np.where((bboxes[:, 4] < classes) & (bboxes[:, 4] >= 0))[0]]

	if bboxes.shape[0] > num_boxes:
		bboxes = bboxes[:num_boxes]

	min_w_h = np.array([bboxes[:, 2] - bboxes[:, 0], bboxes[:, 3] - bboxes[:, 1]]).min()

	bboxes[:, 0] *= (net_w / sx)
	bboxes[:, 2] *= (net_w / sx)
	bboxes[:, 1] *= (net_h / sy)
	bboxes[:, 3] *= (net_h / sy)

	if flip:
		temp = net_w - bboxes[:, 0]
		bboxes[:, 0] = net_w - bboxes[:, 2]
		bboxes[:, 2] = temp

	return bboxes, min_w_h


def rect_intersection(a, b):
	minx = max(a[0], b[0])
	miny = max(a[1], b[1])

	maxx = min(a[2], b[2])
	maxy = min(a[3], b[3])
	return [minx, miny, maxx, maxy]
	
def get_rotated_info(w, h, center, M):
    abs_cos = abs(M[0, 0])
    abs_sin = abs(M[0, 1])
    bound_w = int(h * abs_sin + w * abs_cos)
    bound_h = int(h * abs_cos + w * abs_sin)
    M[0, 2] += bound_w / 2 - center[0]
    M[1, 2] += bound_h / 2 - center[1]
    return (bound_w, bound_h), M	
	
def color_aug(img):
	hue=0.1
	saturation=1.5
	exposure=1.5
	dhue = rand_uniform_strong(-hue, hue)
	dsat = rand_scale(saturation)
	dexp = rand_scale(exposure)
	if dsat != 1 or dexp != 1 or dhue != 0:
		if img.shape[2] >= 3:
			hsv_src = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2HSV)  # BGR to HSV
			hsv = cv2.split(hsv_src)
			hsv[1] *= dsat
			hsv[2] *= dexp
			hsv[0] += 179 * dhue
			hsv_src = cv2.merge(hsv)
			img = np.clip(cv2.cvtColor(hsv_src, cv2.COLOR_HSV2BGR), 0, 255)  # HSV to BGR (the same as previous)
		else:
			img *= dexp

def image_data_augmentation(mat, w, h, pleft, ptop, swidth, sheight, flip, dhue, dsat, dexp, gaussian_noise, blur,
							truth):
	try:
		img = mat
		oh, ow, _ = img.shape
		pleft, ptop, swidth, sheight = int(pleft), int(ptop), int(swidth), int(sheight)
		# crop
		src_rect = [pleft, ptop, swidth + pleft, sheight + ptop]  # x1,y1,x2,y2
		img_rect = [0, 0, ow, oh]
		new_src_rect = rect_intersection(src_rect, img_rect)  # 交集

		dst_rect = [max(0, -pleft), max(0, -ptop), max(0, -pleft) + new_src_rect[2] - new_src_rect[0],
					max(0, -ptop) + new_src_rect[3] - new_src_rect[1]]
		# cv2.Mat sized

		if (src_rect[0] == 0 and src_rect[1] == 0 and src_rect[2] == img.shape[0] and src_rect[3] == img.shape[1]):
			sized = cv2.resize(img, (w, h), cv2.INTER_LINEAR)
		else:
			cropped = np.zeros([sheight, swidth, 3])
			cropped[:, :, ] = np.mean(img, axis=(0, 1))

			cropped[dst_rect[1]:dst_rect[3], dst_rect[0]:dst_rect[2]] = \
				img[new_src_rect[1]:new_src_rect[3], new_src_rect[0]:new_src_rect[2]]

			# resize
			sized = cv2.resize(cropped, (w, h), cv2.INTER_LINEAR)

		# flip
		if flip:
			# cv2.Mat cropped
			sized = cv2.flip(sized, 1)  # 0 - x-axis, 1 - y-axis, -1 - both axes (x & y)

		# HSV augmentation
		# cv2.COLOR_BGR2HSV, cv2.COLOR_RGB2HSV, cv2.COLOR_HSV2BGR, cv2.COLOR_HSV2RGB
		if dsat != 1 or dexp != 1 or dhue != 0:
			if img.shape[2] >= 3:
				hsv_src = cv2.cvtColor(sized.astype(np.float32), cv2.COLOR_RGB2HSV)  # RGB to HSV
				hsv = cv2.split(hsv_src)
				hsv[1] *= dsat
				hsv[2] *= dexp
				hsv[0] += 179 * dhue
				hsv_src = cv2.merge(hsv)
				sized = np.clip(cv2.cvtColor(hsv_src, cv2.COLOR_HSV2RGB), 0, 255)  # HSV to RGB (the same as previous)
			else:
				sized *= dexp

		if blur:
			if blur == 1:
				dst = cv2.GaussianBlur(sized, (17, 17), 0)
				# cv2.bilateralFilter(sized, dst, 17, 75, 75)
			else:
				ksize = (blur / 2) * 2 + 1
				dst = cv2.GaussianBlur(sized, (ksize, ksize), 0)

			if blur == 1:
				img_rect = [0, 0, sized.cols, sized.rows]
				for b in truth:
					left = (b.x - b.w / 2.) * sized.shape[1]
					width = b.w * sized.shape[1]
					top = (b.y - b.h / 2.) * sized.shape[0]
					height = b.h * sized.shape[0]
					roi(left, top, width, height)
					roi = roi & img_rect
					dst[roi[0]:roi[0] + roi[2], roi[1]:roi[1] + roi[3]] = sized[roi[0]:roi[0] + roi[2],
																		  roi[1]:roi[1] + roi[3]]

			sized = dst

		if gaussian_noise:
			noise = np.array(sized.shape)
			gaussian_noise = min(gaussian_noise, 127)
			gaussian_noise = max(gaussian_noise, 0)
			cv2.randn(noise, 0, gaussian_noise)  # mean and variance
			sized = sized + noise
	except:
		print("OpenCV can't augment image: " + str(w) + " x " + str(h))
		sized = mat

	return sized


def filter_truth(bboxes, dx, dy, sx, sy, xd, yd):
	bboxes[:, 0] -= dx
	bboxes[:, 2] -= dx
	bboxes[:, 1] -= dy
	bboxes[:, 3] -= dy

	bboxes[:, 0] = np.clip(bboxes[:, 0], 0, sx)
	bboxes[:, 2] = np.clip(bboxes[:, 2], 0, sx)

	bboxes[:, 1] = np.clip(bboxes[:, 1], 0, sy)
	bboxes[:, 3] = np.clip(bboxes[:, 3], 0, sy)

	out_box = list(np.where(((bboxes[:, 1] == sy) & (bboxes[:, 3] == sy)) |
							((bboxes[:, 0] == sx) & (bboxes[:, 2] == sx)) |
							((bboxes[:, 1] == 0) & (bboxes[:, 3] == 0)) |
							((bboxes[:, 0] == 0) & (bboxes[:, 2] == 0)))[0])
	list_box = list(range(bboxes.shape[0]))
	for i in out_box:
		list_box.remove(i)
	bboxes = bboxes[list_box]

	bboxes[:, 0] += xd
	bboxes[:, 2] += xd
	bboxes[:, 1] += yd
	bboxes[:, 3] += yd

	return bboxes


def blend_truth_mosaic(out_img, img, bboxes, w, h, cut_x, cut_y, i_mixup,
						left_shift, right_shift, top_shift, bot_shift):
	left_shift = min(left_shift, w - cut_x)
	top_shift = min(top_shift, h - cut_y)
	right_shift = min(right_shift, cut_x)
	bot_shift = min(bot_shift, cut_y)

	if i_mixup == 0:
		bboxes = filter_truth(bboxes, left_shift, top_shift, cut_x, cut_y, 0, 0)
		out_img[:cut_y, :cut_x] = img[top_shift:top_shift + cut_y, left_shift:left_shift + cut_x]
	if i_mixup == 1:
		bboxes = filter_truth(bboxes, cut_x - right_shift, top_shift, w - cut_x, cut_y, cut_x, 0)
		out_img[:cut_y, cut_x:] = img[top_shift:top_shift + cut_y, cut_x - right_shift:w - right_shift]
	if i_mixup == 2:
		bboxes = filter_truth(bboxes, left_shift, cut_y - bot_shift, cut_x, h - cut_y, 0, cut_y)
		out_img[cut_y:, :cut_x] = img[cut_y - bot_shift:h - bot_shift, left_shift:left_shift + cut_x]
	if i_mixup == 3:
		bboxes = filter_truth(bboxes, cut_x - right_shift, cut_y - bot_shift, w - cut_x, h - cut_y, cut_x, cut_y)
		out_img[cut_y:, cut_x:] = img[cut_y - bot_shift:h - bot_shift, cut_x - right_shift:w - right_shift]

	return out_img, bboxes


def draw_box(img, bboxes):
	for b in bboxes:
		img = cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
	return img


def read_patch(data_dir):
	back_path=os.path.join(data_dir,"background")
	patch_path=os.path.join(data_dir,"patch")
	negative_path=os.path.join(data_dir,"negative")

	negative_label=[]
	for root, dirnames, filenames in os.walk(negative_path):
		for filename in filenames:
			negative_label.append(os.path.join(root,filename))

	back_label=[]
	for root, dirnames, filenames in os.walk(back_path):
		for filename in filenames:
			back_label.append(os.path.join(root,filename))

	label_map=[]
	patch_label=[]			
	for root, dirnames, filenames in os.walk(patch_path):
		
		if(len(filenames)<=0):
			continue
		label=os.path.basename(root)
		label_map.append(label)
		patch_list=[]
		for filename in filenames:
			patch_list.append(os.path.join(root,filename))
		patch_label.append({label:patch_list})
			
	return back_label,patch_label,negative_label,label_map
	
class Yolo_dataset(Dataset):
	def __init__(self, label_path, cfg, train=True ,dataset_capacity=8192):
		super(Yolo_dataset, self).__init__()
		if cfg.mixup == 2:
			print("cutmix=1 - isn't supported for Detector")
			raise
		elif cfg.mixup == 2 and cfg.letter_box:
			print("Combination: letter_box=1 & mosaic=1 - isn't supported, use only 1 of these parameters")
			raise

		self.cfg = cfg
		self.train = train
		
		self.back_labels,self.patch_labels,self.negative_label,self.class_name=read_patch(label_path)
		self.cfg.classes=len(self.class_name)
		self.class_to_ind = dict(zip(self.class_name, range(self.cfg.classes)))	
		self.dataset_capacity=dataset_capacity
		self.val_list={}
		
	def __len__(self):
		return self.dataset_capacity
		
	def anns_to_bboxes(self, anns):
		bboxes = np.array(anns, dtype=np.float32)
		return bboxes
		
	def patch_to_data(self, max_objs=5):
		
		choice_background = np.random.choice(self.back_labels, 1)[0]

		choice_patch_ids = np.random.choice(list(range(0, len(self.patch_labels))), max_objs)
		patch_dict = {}
		for id in choice_patch_ids:
			label = self.patch_labels[id]
			key = list(label.keys())[0]
			patch = np.random.choice(label[key], 1)[0]
			patch_dict[patch] = key

		ret_img = cv2.imread(choice_background)


		if (ret_img is None):
			return None, []

		patch_img_list = []
		for patch_dir in patch_dict.keys():
			patch_img = cv2.imread(patch_dir, cv2.IMREAD_UNCHANGED)
			if (patch_img is None):
				continue
			ph, pw, pc = patch_img.shape
			if np.random.random() < 0.5:
				screw_scale = 0.4
				pts_base = [[0, 0], [pw, 0], [0, ph], [pw, ph]]

				tl = pts_base[0].copy()
				tl[0] = np.random.uniform(low=0, high=pw * screw_scale, size=1)[0]
				tl[1] = np.random.uniform(low=0, high=ph * screw_scale, size=1)[0]

				tr = pts_base[1].copy()
				tr[0] = tr[0] - np.random.uniform(low=0, high=pw * screw_scale, size=1)[0]
				tr[1] = np.random.uniform(low=0, high=ph * screw_scale, size=1)[0]

				bl = pts_base[2].copy()
				bl[0] = np.random.uniform(low=0, high=pw * screw_scale, size=1)[0]
				bl[1] = bl[1] - np.random.uniform(low=0, high=ph * screw_scale, size=1)[0]

				br = pts_base[3].copy()
				br[0] = br[0] - np.random.uniform(low=0, high=pw * screw_scale, size=1)[0]
				br[1] = br[1] - np.random.uniform(low=0, high=ph * screw_scale, size=1)[0]
				pts1 = np.float32([tl, tr, bl, br])

				pts_base = np.float32(pts_base)
				PM = cv2.getPerspectiveTransform(pts_base, pts1)
				patch_img = cv2.warpPerspective(patch_img, PM, (pw, ph))
				coords = np.argwhere(patch_img[:, :, 3] > 0)
				if (coords is None):
					continue
				if (coords.shape[0] < 4):
					continue
				x0, y0 = coords.min(axis=0)
				x1, y1 = coords.max(axis=0) + 1
				patch_img = patch_img[x0:x1, y0:y1]

			if np.random.random() < 0.5:
				patch_img= (patch_img.astype(np.float32) / 255.)
				color_aug(patch_img)
				patch_img=(patch_img*255).astype(np.uint8)

			ph, pw, pc = patch_img.shape

			if np.random.random() < 0.5:
				patch_img = patch_img[:, ::-1, :]
			if np.random.random() < 0.5:
				patch_img = patch_img[::-1, :, :]

			scale = np.random.uniform(low=0.3, high=1.2, size=1)[0]
			rotaion = np.random.choice([0, 90, 180, 270], 1)[0]

			image_center = (pw / 2, ph / 2)
			M = cv2.getRotationMatrix2D((image_center[0], image_center[1]), angle=rotaion, scale=scale)
			size, M = get_rotated_info(w=pw, h=ph, center=image_center, M=M)
			patch_img = cv2.warpAffine(patch_img, M, (size[0], size[1]))
			ph, pw, pc = patch_img.shape
			patch_img_list.append((patch_img, int(pw * ph), patch_dir))

		patch_img_list = sorted(patch_img_list, key=itemgetter(1), reverse=True)

		bh, bw, bc = ret_img.shape
		anns = []
		for patch_img, _, patch_dir in patch_img_list:
			if (patch_img is None):
				continue
			ph, pw, pc = patch_img.shape
			resize = False
			sfx = sfy = ph
			if (ph > bh):
				sfy = bh / ph
				resize = True
			if (pw > bw):
				sfx = bw / pw
				resize = True

			if (resize == True):
				sf = sfx if sfx < sfy else sfy
				sf = sf * 0.5
				patch_img = cv2.resize(patch_img, dsize=None, fx=sf, fy=sf)

			patch_img_rgb = patch_img[:, :, :3]
			patch_img_mask = patch_img[:, :, 3]

			ph, pw, pc = patch_img.shape
			xmax = bw - pw
			ymax = bh - ph
			# offset_x, offset_y = self.get_offset(xmax, ymax, pw, ph, anns)
			offset_x=0
			if(xmax>0):
				offset_x=np.random.randint(0, xmax, size=1)[0]
			offset_y=0
			if(ymax>0):
				offset_y=np.random.randint(0, ymax, size=1)[0]

			fg = cv2.bitwise_and(patch_img_rgb, patch_img_rgb, mask=patch_img_mask)
			patch_img_mask = cv2.bitwise_not(patch_img_mask)
			bg = ret_img[offset_y:offset_y + ph, offset_x:offset_x + pw]
			bg = cv2.bitwise_or(bg, bg, mask=patch_img_mask)
			ret_img[offset_y:offset_y + ph, offset_x:offset_x + pw] = cv2.bitwise_or(fg, bg)
			anns.append([offset_x, offset_y, pw+offset_x, ph+offset_y, self.class_to_ind[patch_dict[patch_dir]]])
			
		if np.random.random() < 0.5:
			prob = 0.05
			rnd = np.random.rand(ret_img.shape[0], ret_img.shape[1], ret_img.shape[2])
			ret_img[rnd < prob] = np.random.rand(1)
			
		return ret_img, anns		
		
	def __getitem__(self, index):
		if not self.train:
			return self._get_val_item(index)
		use_mixup = self.cfg.mixup
		if random.randint(0, 1):
			use_mixup = 0

		if use_mixup == 3:
			min_offset = 0.2
			cut_x = random.randint(int(self.cfg.w * min_offset), int(self.cfg.w * (1 - min_offset)))
			cut_y = random.randint(int(self.cfg.h * min_offset), int(self.cfg.h * (1 - min_offset)))

		r1, r2, r3, r4, r_scale = 0, 0, 0, 0, 0
		dhue, dsat, dexp, flip, blur = 0, 0, 0, 0, 0
		gaussian_noise = 0

		out_img = np.zeros([self.cfg.h, self.cfg.w, 3])
		out_bboxes = []

		for i in range(use_mixup+1):
			num_objs = 5
			img = None
			anns = []
		   # end = time.time()
			while ((len(anns) > 0 and img is not None) != True):
				img, anns = self.patch_to_data(max_objs=num_objs)	

			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			bboxes=self.anns_to_bboxes(anns)

			oh, ow, oc = img.shape
			dh, dw, dc = np.array(np.array([oh, ow, oc]) * self.cfg.jitter, dtype=np.int)

			dhue = rand_uniform_strong(-self.cfg.hue, self.cfg.hue)
			dsat = rand_scale(self.cfg.saturation)
			dexp = rand_scale(self.cfg.exposure)

			pleft = random.randint(-dw, dw)
			pright = random.randint(-dw, dw)
			ptop = random.randint(-dh, dh)
			pbot = random.randint(-dh, dh)

			flip = random.randint(0, 1) if self.cfg.flip else 0

			if (self.cfg.blur):
				tmp_blur = random.randint(0, 2)  # 0 - disable, 1 - blur background, 2 - blur the whole image
				if tmp_blur == 0:
					blur = 0
				elif tmp_blur == 1:
					blur = 1
				else:
					blur = self.cfg.blur

			if self.cfg.gaussian and random.randint(0, 1):
				gaussian_noise = self.cfg.gaussian
			else:
				gaussian_noise = 0

			if self.cfg.letter_box:
				img_ar = ow / oh
				net_ar = self.cfg.w / self.cfg.h
				result_ar = img_ar / net_ar
				# print(" ow = %d, oh = %d, w = %d, h = %d, img_ar = %f, net_ar = %f, result_ar = %f \n", ow, oh, w, h, img_ar, net_ar, result_ar);
				if result_ar > 1:  # sheight - should be increased
					oh_tmp = ow / net_ar
					delta_h = (oh_tmp - oh) / 2
					ptop = ptop - delta_h
					pbot = pbot - delta_h
					# print(" result_ar = %f, oh_tmp = %f, delta_h = %d, ptop = %f, pbot = %f \n", result_ar, oh_tmp, delta_h, ptop, pbot);
				else:  # swidth - should be increased
					ow_tmp = oh * net_ar
					delta_w = (ow_tmp - ow) / 2
					pleft = pleft - delta_w
					pright = pright - delta_w
					# printf(" result_ar = %f, ow_tmp = %f, delta_w = %d, pleft = %f, pright = %f \n", result_ar, ow_tmp, delta_w, pleft, pright);

			swidth = ow - pleft - pright
			sheight = oh - ptop - pbot

			truth, min_w_h = fill_truth_detection(bboxes, self.cfg.boxes, self.cfg.classes, flip, pleft, ptop, swidth,
												  sheight, self.cfg.w, self.cfg.h)
			if (min_w_h / 8) < blur and blur > 1:  # disable blur if one of the objects is too small
				blur = min_w_h / 8

			ai = image_data_augmentation(img, self.cfg.w, self.cfg.h, pleft, ptop, swidth, sheight, flip,
										 dhue, dsat, dexp, gaussian_noise, blur, truth)

			if use_mixup == 0:
				out_img = ai
				out_bboxes = truth
			if use_mixup == 1:
				if i == 0:
					old_img = ai.copy()
					old_truth = truth.copy()
				elif i == 1:
					out_img = cv2.addWeighted(ai, 0.5, old_img, 0.5)
					out_bboxes = np.concatenate([old_truth, truth], axis=0)
			elif use_mixup == 3:
				if flip:
					tmp = pleft
					pleft = pright
					pright = tmp

				left_shift = int(min(cut_x, max(0, (-int(pleft) * self.cfg.w / swidth))))
				top_shift = int(min(cut_y, max(0, (-int(ptop) * self.cfg.h / sheight))))

				right_shift = int(min((self.cfg.w - cut_x), max(0, (-int(pright) * self.cfg.w / swidth))))
				bot_shift = int(min(self.cfg.h - cut_y, max(0, (-int(pbot) * self.cfg.h / sheight))))

				out_img, out_bbox = blend_truth_mosaic(out_img, ai, truth.copy(), self.cfg.w, self.cfg.h, cut_x,
													   cut_y, i, left_shift, right_shift, top_shift, bot_shift)
				out_bboxes.append(out_bbox)

		
		if use_mixup == 3:
			out_bboxes = np.concatenate(out_bboxes, axis=0)

		out_bboxes1 = np.zeros([self.cfg.boxes, 5])
		out_bboxes1[:min(out_bboxes.shape[0], self.cfg.boxes)] = out_bboxes[:min(out_bboxes.shape[0], self.cfg.boxes)]
		return out_img, out_bboxes1

	def _get_val_item(self,index):
		"""
		"""
		if(index in self.val_list.keys()):
			img=self.val_list[index][0]
			anns=self.val_list[index][1]
		else:
			num_objs = 5
			img = None
			anns = []
			while ((len(anns) > 0 and img is not None) != True):
				img, anns = self.patch_to_data(max_objs=num_objs)
			self.val_list[index]=(img,anns)
 
		bboxes_with_cls_id = self.anns_to_bboxes(anns)
		# img_height, img_width = img.shape[:2]
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		# img = cv2.resize(img, (self.cfg.w, self.cfg.h))
		# img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
		num_objs = len(bboxes_with_cls_id)
		target = {}
		# boxes to coco format
		boxes = bboxes_with_cls_id[...,:4]
		boxes[..., 2:] = boxes[..., 2:] - boxes[..., :2]  # box width, box height
		target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
		target['labels'] = torch.as_tensor(bboxes_with_cls_id[...,-1].flatten(), dtype=torch.int64)
		target['image_id'] = torch.tensor([index])
		target['area'] = (target['boxes'][:,3])*(target['boxes'][:,2])
		target['iscrowd'] = torch.zeros((num_objs,), dtype=torch.int64)

		return img, target


def get_image_id(filename:str) -> int:
	raise NotImplementedError("Create your own 'get_image_id' function")
	lv, no = os.path.splitext(os.path.basename(filename))[0].split("_")
	lv = lv.replace("level", "")
	no = f"{int(no):04d}"
	return int(lv+no)


if __name__ == "__main__":
	from cfg import Cfg
	import matplotlib.pyplot as plt

	random.seed(2020)
	np.random.seed(2020)
	Cfg.dataset_dir = 'C:/SYN_DATA/'
	dataset = Yolo_dataset('C:/SYN_DATA/', Cfg)
	print(Cfg.classes)
	for i in range(100):
		out_img, out_bboxes = dataset.__getitem__(i)
		a = draw_box(out_img.copy(), out_bboxes.astype(np.int32))
		plt.imshow(a.astype(np.int32))
		plt.show()
