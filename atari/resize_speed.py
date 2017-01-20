import numpy as np
import cv2
from timeit import default_timer as timer
import scipy.misc

#########################################################################

# RESIZE

imgs = list()

# for _ in range(1000):
#     imgs.append(np.random.rand(210, 160).astype(np.float32))
# print("testing float32")

for _ in range(1000):
    imgs.append(np.random.randint(low=0, high=255, size=(210, 160), dtype=np.uint8))
print("testing uint8")


# RESIZE CV2

t0 = timer()
for img in imgs:
    z = cv2.resize(img, (84, 84), interpolation=cv2.INTER_CUBIC)
t1 = timer()
print("CV2 CUBIC time: ", t1 - t0)

t0 = timer()
for img in imgs:
    z = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
t1 = timer()
print("CV2 AREA time: ", t1 - t0)

t0 = timer()
for img in imgs:
    z = cv2.resize(img, (84, 84), interpolation=cv2.INTER_LINEAR)
t1 = timer()
print("CV2 LINEAR time: ", t1 - t0)

t0 = timer()
for img in imgs:
    z = cv2.resize(img, (84, 84), interpolation=cv2.INTER_LANCZOS4)
t1 = timer()
print("CV2 LANCSOZ time: ", t1 - t0)

t0 = timer()
for img in imgs:
    z = cv2.resize(img, (84, 84), interpolation=cv2.INTER_NEAREST)
t1 = timer()
print("CV2 NEAREST time: ", t1 - t0)

###############################################################

# RESIZE scipy

# t0 = timer()
# for img in imgs:
#     z = scipy.misc.imresize(img, (84, 84), interp='nearest')
# t1 = timer()
# print("scipy NEAREST time: ", t1 - t0)

# t0 = timer()
# for img in imgs:
#     z = scipy.misc.imresize(img, (84, 84), interp='lanczos')
# t1 = timer()
# print("scipy LANCZOS time: ", t1 - t0)
# t0 = timer()
# for img in imgs:
#     z = scipy.misc.imresize(img, (84, 84), interp='bilinear')
# t1 = timer()
# print("scipy BILINEAR time: ", t1 - t0)
# t0 = timer()
# for img in imgs:
#     z = scipy.misc.imresize(img, (84, 84), interp='bicubic')
# t1 = timer()
# print("scipy BICUBIC time: ", t1 - t0)
# t0 = timer()
# for img in imgs:
#     z = scipy.misc.imresize(img, (84, 84), interp='cubic')
# t1 = timer()
# print("scipy CUBIC time: ", t1 - t0)

############################################################

# GRAY

# imgs3 = list()
# imgs4 = list()

# for _ in range(1000):
#     imgs3.append(np.random.rand(210, 160, 3).astype(np.float32))
# print("testing float32")

# for _ in range(1000):
#     imgs4.append(np.random.rand(210, 160, 4).astype(np.float32))
# print("testing float32")

# CV2_BGR2GRAY = np.array([0.114, 0.587, 0.299], dtype=np.float32)
# t0 = timer()
# for img in imgs:
#     z = img.dot(CV2_BGR2GRAY)
# t1 = timer()
# print("numpy GRAY time: ", t1 - t0)

# t0 = timer()
# for img in imgs3:
#     z = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# t1 = timer()
# print("cv2 GRAY time: ", t1 - t0)


# t0 = timer()
# for img in imgs4:
#     y = img[:, :, 0:3]
#     z = cv2.cvtColor(y, cv2.COLOR_BGR2GRAY)
# t1 = timer()
# print("cv2 GRAY (3-slice) time: ", t1 - t0)

# t0 = timer()
# for img in imgs4:
#     y = img[:, :, 0:3].copy()
#     z = cv2.cvtColor(y, cv2.COLOR_BGR2GRAY)
# t1 = timer()
# print("cv2 GRAY (3-copy) time: ", t1 - t0)
