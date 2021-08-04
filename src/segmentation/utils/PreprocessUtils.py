import numpy as np

class PreprocessUtils:

	def __init__(self):
		self.SMOOTH = 1e-6

	def RGB_to_Binary_OneHot(self, num_classes, img_arr, classes):

		(h, w, d) = img_arr.shape
		new_image_1 = np.zeros((h, w))
		new_image_2 = np.zeros((h, w))
		new_image_3 = np.zeros((h, w))
		new_image_4 = np.zeros((h, w))
		new_image_5 = np.zeros((h, w))
		new_image_6 = np.zeros((h, w))
		new_image_7 = np.zeros((h, w))

		if num_classes == 1:
			c1, c2, c3, c4, c5, c6 = classes

		c_inds = np.where(np.all(img_arr == c1, axis=2))
		r_inds = np.where(np.all(img_arr == c2, axis=2))
		bg_inds = np.where(np.all(img_arr == c3, axis=2))
		bg_inds1 = np.where(np.all(img_arr == c4, axis=2))
		bg_inds2 = np.where(np.all(img_arr == c5, axis=2))
		bg_inds3 = np.where(np.all(img_arr == c6, axis=2))

		new_image_1[c_inds] = 1
		new_image_2[r_inds] = 1
		new_image_3[bg_inds] = 1
		new_image_4[bg_inds1] = 1
		new_image_5[bg_inds2] = 1
		new_image_6[bg_inds3] = 1




    
		new_image_1 = np.reshape(new_image_1, (h, w, 1))
		new_image_2 = np.reshape(new_image_2, (h, w, 1))
		new_image_3 = np.reshape(new_image_3, (h, w, 1))
		new_image_4 = np.reshape(new_image_4, (h, w, 1))
		new_image_5 = np.reshape(new_image_5, (h, w, 1))
		new_image_6 = np.reshape(new_image_6, (h, w, 1))

		new_image_7 = np.reshape(new_image_7, (h, w, 1))


		#new_image = np.concatenate((new_image_1, new_image_2, new_image_3, new_image_4, new_image_5, new_image_6), axis=2)
		new_image = np.concatenate((new_image_1, new_image_2, new_image_3, new_image_4, new_image_5, new_image_6), axis=2)

		return new_image_2
