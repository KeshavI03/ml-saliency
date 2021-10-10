import cv2
import numpy as np


# image_c = cv2.imread('Images/shift_1.JPG')

# image = cv2.imread('Images/motor.JPG', cv2.IMREAD_GRAYSCALE)

# image = cv2.imread('Images/ice_big.JPEG', cv2.IMREAD_GRAYSCALE)

# image = cv2.imread('Images/ice.JPG', cv2.IMREAD_GRAYSCALE)

# image_c = cv2.imread('Images/motor.JPG')

# image = cv2.imread('Images/tiger.JPEG', cv2.IMREAD_GRAYSCALE)

# image_c = cv2.imread('Images/person.JPG')

# image_c = cv2.imread('Images/boat.JPG')

# image_c = cv2.imread('Images/city.JPG')

# image_c = cv2.imread('Images/reef.JPG')

# image_c = cv2.imread('Images/car.JPG')

image_c = cv2.imread('Images/dog.JPG')


image = cv2.cvtColor(image_c, cv2.COLOR_BGR2GRAY)

ratio = 1

resized = (int(image.shape[1] / ratio), int(image.shape[0] / ratio))


imS = (cv2.resize(image, resized).astype('int8') + 256) % 256
imS_c = (cv2.resize(image_c, resized).astype('int8') + 256) % 256


# vert_deriv  = np.add(np.divide(np.diff(imS, axis = 1), 2), 128).astype('uint8')
# horiz_deriv  = np.add(np.divide(np.diff(imS, axis = 0), 2), 128).astype('uint8')

vert_deriv = np.where(np.absolute(np.diff(imS, axis = 1)) > 50, 255, 0).astype('uint8')
horiz_deriv = np.where(np.absolute(np.diff(imS, axis = 0)) > 50, 255, 0).astype('uint8')

combine = np.add(vert_deriv[:-1,:], horiz_deriv[:,:-1])





gauss1 = cv2.GaussianBlur(imS,(3,3),cv2.BORDER_DEFAULT)

vert_deriv_g1  = np.where(np.absolute(np.diff(gauss1, axis = 1)) > 12, 255, 0).astype('uint8')
horiz_deriv_g1  = np.where(np.absolute(np.diff(gauss1, axis = 0)) > 20, 255, 0).astype('uint8')

combine_g1 = np.add(vert_deriv_g1[:-1,:], horiz_deriv_g1[:,:-1])



gauss = cv2.GaussianBlur(imS,(5,5),cv2.BORDER_DEFAULT)

vert_deriv_g  = np.where(np.absolute(np.diff(gauss, axis = 1)) > 38, 255, 0).astype('uint8')
horiz_deriv_g  = np.where(np.absolute(np.diff(gauss, axis = 0)) > 35, 255, 0).astype('uint8')

combine_g = np.add(vert_deriv_g[:-1,:], horiz_deriv_g[:,:-1])



gauss2 = cv2.GaussianBlur(imS,(7,7),cv2.BORDER_DEFAULT)
gauss2_color = cv2.GaussianBlur(imS_c, (7,7), cv2.BORDER_DEFAULT)

vert_deriv_g2  = np.where(np.absolute(np.diff(gauss2, axis = 1)) > 28,  255, 0).astype('uint8')
horiz_deriv_g2  = np.where(np.absolute(np.diff(gauss2, axis = 0)) > 20, 255, 0).astype('uint8')

combine_g2 = np.add(vert_deriv_g2[:-1,:], horiz_deriv_g2[:,:-1])





med = cv2.medianBlur(imS.astype('uint8'), 5).astype('int8')

vert_deriv_m  = np.where(np.absolute(np.diff(med, axis = 1)) > 30, 255, 0).astype('uint8')
horiz_deriv_m  = np.where(np.absolute(np.diff(med, axis = 0)) > 30, 255, 0).astype('uint8')

combine_m = np.add(vert_deriv_m[:-1,:], horiz_deriv_m[:,:-1])



med_g = cv2.medianBlur(combine_g.astype('uint8'), 3).astype('int8')
med_g1 = cv2.medianBlur(combine_g1.astype('uint8'), 3).astype('int8')
med_g2 = cv2.medianBlur(combine_g2.astype('uint8'), 3).astype('int8')

# add_med = np.divide(med_g1 + med_g + med_g2, 3)
add_med = med_g1 + med_g + med_g2
add_med_t = add_med + combine_m


# kernel = np.ones((5,5), np.uint8)

kernel = np.array([[0, 0, 1, 1, 1, 0, 0],
				   [0, 1, 1, 1, 1, 1, 0],
				   [1, 1, 1, 1, 1, 1, 1],
				   [1, 1, 1, 1, 1, 1, 1],
				   [1, 1, 1, 1, 1, 1, 1],
				   [0, 1, 1, 1, 1, 1, 0],
				   [0, 0, 1, 1, 1, 0, 0]], dtype = 'uint8')




gauss_dialate = cv2.dilate(add_med.astype('uint8'), kernel, iterations = 4)

contours, hierarchy = cv2.findContours(image=gauss_dialate, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

# print(contours)
final = np.zeros(gauss_dialate.shape)



def get_dist(contour):
	dist_x = 0
	dist_y = 0
	hamming_dist = 0
	contour_size = len(contour)
	# print(cv2.contourArea(contour))
	for point in contour:
		dist_x += abs(final.shape[1] - point[0][1])
		dist_y += abs(final.shape[0] - point[0][0])
		hamming_dist = dist_y + dist_x

	hamming_dist = float(hamming_dist) / contour_size

	# print(hamming_dist)

	return cv2.contourArea(contour) / hamming_dist


# get_dist(contours[1])

# cnt_sorted = sorted(contours, key= (cv2.contourArea / get_dist), reverse = True)
cnt_sorted = sorted(contours, key= (get_dist), reverse = True)

# print(type(contours[0]))




cv2.drawContours(final, cnt_sorted[0:1], -1, (255, 255, 0), cv2.FILLED)

resized_l = (int(imS.shape[1] * ratio), int(imS.shape[0] * ratio))

im_i = cv2.resize(final, resized_l)
# im_g = cv2.resize(gauss2, resized_l)
resized_g_c = cv2.resize(gauss2_color, resized_l)

# blur_c = np.where(im_i != 0, image[:,:], 0)
blur_c_b = np.where(im_i != 0, image_c[:,:,0], resized_g_c[:,:,0])
blur_c_g = np.where(im_i != 0, image_c[:,:,1], resized_g_c[:,:,1])
blur_c_r = np.where(im_i != 0, image_c[:,:,2], resized_g_c[:,:,2])


final_colorized = cv2.merge([blur_c_b,blur_c_g,blur_c_r])

# print(im_i.shape)
# print(im_g.shape)

# print(blur_c)


# cv2.imshow('vert deriv', imS.astype('uint8'))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow('vert deriv', vert_deriv)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow('horiz deriv', horiz_deriv)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow('combine', combine)
# cv2.waitKey(0)
# cv2.destroyAllWindows()





# cv2.imshow('gauss', gauss1.astype('uint8'))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow('gaus vert', vert_deriv_g1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow('gauss horiz', horiz_deriv_g1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow('combine', combine_g1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow('gauss horiz', med_g1.astype('uint8'))
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# cv2.imshow('gauss', gauss.astype('uint8'))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow('gaus vert', vert_deriv_g)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow('gauss horiz', horiz_deriv_g)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow('combine', combine_g)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# cv2.imshow('gauss horiz', med_g.astype('uint8'))
# cv2.waitKey(0)
# cv2.destroyAllWindows()




# cv2.imshow('gauss', gauss2.astype('uint8'))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow('gaus vert', vert_deriv_g2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow('gauss horiz', horiz_deriv_g2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow('combine', combine_g2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow('gauss horiz', med_g2.astype('uint8'))
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# cv2.imshow('gauss horiz', med.astype('uint8'))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow('gaus vert', vert_deriv_m)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow('gauss horiz', horiz_deriv_m)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow('combine', combine_m)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # kernel = np.ones((5,5), np.uint8)

# cv2.imshow('combine', add_med.astype('uint8'))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow('combine', add_med_t.astype('uint8'))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cv2.imshow('combine', final.astype('uint8'))
cv2.waitKey(0)
cv2.destroyAllWindows()

# cv2.imshow('combine', blur_c.astype('uint8'))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cv2.imwrite('test.png', final_colorized)