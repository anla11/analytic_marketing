'''
	@author: anla-ds
	created date: 8 July, 2020
'''
import numpy as np

class ImageThreshold(): # applying adaptive threshold in image processing
	def __init__(self, arr):
		self.image = np.array(arr).astype(int)
		self.hist, self.bin_centers = None, None

	def __get_hist(self):
		if self.hist is None:
			image = self.image
			image_min, image_max = np.min(image), np.max(image)
			hist = np.bincount(image.ravel(), minlength=image_max - image_min + 1)
			bin_centers = np.arange(image_min, image_min + len(hist))
			self.hist, self.bin_centers = hist.astype(float), bin_centers
		return self.hist, self.bin_centers

	def otsu_threshold(self):
		hist, bin_centers = self.__get_hist()

		# class probabilities for all possible thresholds
		weight1 = np.cumsum(hist)
		weight2 = np.cumsum(hist[::-1])[::-1]
		# class means for all possible thresholds
		mean1 = np.cumsum(hist * bin_centers) / weight1
		mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]

		# Clip ends to align class 1 and class 2 variables:
		# The last value of ``weight1``/``mean1`` should pair with zero values in
		# ``weight2``/``mean2``, which do not exist.
		variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

		idx = np.argmax(variance12)
		threshold = bin_centers[:-1][idx]
		return threshold

	def yen_threshold(self):
		'''
		References
		----------
		.. [1] Yen J.C., Chang F.J., and Chang S. (1995) "A New Criterion
			for Automatic Multilevel Thresholding" IEEE Trans. on Image
			Processing, 4(3): 370-378. :DOI:`10.1109/83.366472`
		.. [2] Sezgin M. and Sankur B. (2004) "Survey over Image Thresholding
			Techniques and Quantitative Performance Evaluation" Journal of
			Electronic Imaging, 13(1): 146-165, :DOI:`10.1117/1.1631315`
			http://www.busim.ee.boun.edu.tr/~sankur/SankurFolder/Threshold_survey.pdf
		.. [3] ImageJ AutoThresholder code, http://fiji.sc/wiki/index.php/Auto_Threshold
		'''
		hist, bin_centers = self.__get_hist()

		# Calculate probability mass function
		pmf = hist.astype(np.float32) / hist.sum()
		P1 = np.cumsum(pmf)  # Cumulative normalized histogram
		P1_sq = np.cumsum(pmf ** 2)
		# Get cumsum calculated from end of squared array:
		P2_sq = np.cumsum(pmf[::-1] ** 2)[::-1]
		# P2_sq indexes is shifted +1. I assume, with P1[:-1] it's help avoid
		# '-inf' in crit. ImageJ Yen implementation replaces those values by zero.
		crit = np.log(((P1_sq[:-1] * P2_sq[1:]) ** -1) *
					(P1[:-1] * (1.0 - P1[:-1])) ** 2)
		return bin_centers[crit.argmax()]

	def iso_threshold(self, return_all=False):
		'''
		References
		.. [1] Ridler, TW & Calvard, S (1978), "Picture thresholding using an
			iterative selection method"
			IEEE Transactions on Systems, Man and Cybernetics 8: 630-632,
			:DOI:`10.1109/TSMC.1978.4310039`
		.. [2] Sezgin M. and Sankur B. (2004) "Survey over Image Thresholding
			Techniques and Quantitative Performance Evaluation" Journal of
			Electronic Imaging, 13(1): 146-165,
			http://www.busim.ee.boun.edu.tr/~sankur/SankurFolder/Threshold_survey.pdf
			:DOI:`10.1117/1.1631315`
		.. [3] ImageJ AutoThresholder code,
			http://fiji.sc/wiki/index.php/Auto_Threshold
		'''
		hist, bin_centers = self.__get_hist()
		
		# csuml and csumh contain the count of pixels in that bin or lower, and
		# in all bins strictly higher than that bin, respectively
		csuml = np.cumsum(hist)
		csumh = csuml[-1] - csuml

		# intensity_sum contains the total pixel intensity from each bin
		intensity_sum = hist * bin_centers

		# l and h contain average value of all pixels in that bin or lower, and
		# in all bins strictly higher than that bin, respectively.
		# Note that since exp.histogram does not include empty bins at the low or
		# high end of the range, csuml and csumh are strictly > 0, except in the
		# last bin of csumh, which is zero by construction.
		# So no worries about division by zero in the following lines, except
		# for the last bin, but we can ignore that because no valid threshold
		# can be in the top bin.
		# To avoid the division by zero, we simply skip over the last element in
		# all future computation.
		csum_intensity = np.cumsum(intensity_sum)
		lower = csum_intensity[:-1] / csuml[:-1]
		higher = (csum_intensity[-1] - csum_intensity[:-1]) / csumh[:-1]

		# isodata finds threshold values that meet the criterion t = (l + m)/2
		# where l is the mean of all pixels <= t and h is the mean of all pixels
		# > t, as calculated above. So we are looking for places where
		# (l + m) / 2 equals the intensity value for which those l and m figures
		# were calculated -- which is, of course, the histogram bin centers.
		# We only require this equality to be within the precision of the bin
		# width, of course.
		all_mean = (lower + higher) / 2.0
		bin_width = bin_centers[1] - bin_centers[0]

		# Look only at thresholds that are below the actual all_mean value,
		# for consistency with the threshold being included in the lower pixel
		# group. Otherwise can get thresholds that are not actually fixed-points
		# of the isodata algorithm. For float images, this matters less, since
		# there really can't be any guarantees anymore anyway.
		distances = all_mean - bin_centers[:-1]
		thresholds = bin_centers[:-1][(distances >= 0) & (distances < bin_width)]
		if return_all:
			return thresholds
		else:
			return thresholds[0]

