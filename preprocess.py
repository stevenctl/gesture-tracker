import cv2
import numpy as np
import math
import video_util

"""
Credit: P-bhandar on github for original C++ code
Ported and modified from https://github.com/P-bhandari/Opencv-Tutorials/blob/master/OpenCv%20Tutorials/FingerDetection.cpp
"""


class Context:
    image = None  # input image

    filter_image = None  # filtered through threshold
    temp_image1 = None  # 1 channel
    temp_st1 = None  # 1 channel
    temp_image3 = None  # 3 channel

    def set_image(self, img: np.ndarray):
        self.image = img
        self.temp_image3 = np.zeros_like(img)
        self.temp_image1 = np.zeros((*img.shape[:2], 1), img.dtype)
        self.filter_image = np.zeros((*img.shape[:2], 1), img.dtype)

    contours = None  # hand contour

    hull = None  # hand convex hull

    defects = None

    hand_center = None
    fingers = None

    kernel = None  # for morph operations

    num_fingers = None
    hand_radius = None
    num_defects = None

    skin_samples = []
    skin_color = None
    skin_variance = None
    tolerance = 100

    contour_idx = 0

    max_fingers = 5
    max_finger_dist = 30000


def compute_skin_color(ctx: Context):
    """
    compute_skin_color uses the median value from each channel to approximate the skin color

    Context variables used:
    - ctx.skin_samples

    Context variables set:
    - ctx.skin_color
    - ctx.skin_variance

    """
    if len(ctx.skin_samples) < 3:
        return

    samples = np.array(ctx.skin_samples)

    # get the median for each channel
    r = samples[:, 0, 0]
    g = samples[:, 0, 1]
    b = samples[:, 0, 2]
    ctx.skin_color = np.array((np.median(r), np.median(g), np.median(b)))

    # compute the maximum variance from the median in each channel
    # TODO(slandow) prune outliers before computing variance?
    # this works fine with manual sampling, but automatic sampling may have noisy samples
    variances = np.abs(samples[:] - ctx.skin_color)
    r = variances[:, 0, 0]
    g = variances[:, 0, 1]
    b = variances[:, 0, 2]
    ctx.skin_variance = np.array([np.max(r), np.max(g), np.max(b)])


def filter_and_threshold(ctx: Context):
    """
     filter_and_threshold applies some smoothing before filtering to colors near the computed skin value

     Context variables used:
     - skin_color
     - skin_variance
     - image

     Context variables set:
     - temp_image3
     - trh_image
    """
    if ctx.skin_color is None:
        return

    # Soften image
    cv2.GaussianBlur(ctx.image, (11, 11), 0, ctx.temp_image3)
    # Denoise
    cv2.medianBlur(ctx.temp_image3, 11, ctx.temp_image3)

    # Look for approximated skin color
    tolerance = (ctx.tolerance / 100) * ctx.skin_variance
    cv2.inRange(ctx.temp_image3, ctx.skin_color - tolerance, ctx.skin_color + tolerance, ctx.filter_image)

    cv2.morphologyEx(ctx.filter_image, cv2.MORPH_OPEN, None, ctx.filter_image)
    cv2.morphologyEx(ctx.filter_image, cv2.MORPH_CLOSE, None, ctx.filter_image)

    cv2.GaussianBlur(ctx.image, (3, 3), 0, ctx.filter_image)


def find_contour(ctx: Context):
    """
    find_contour will set ctx.contour if successful

    Context variables used:
    - trh_image

    Context variables set:
    - temp_image1
    - contours
    """
    cv2.copyTo(ctx.filter_image, np.ones_like(ctx.temp_image1), ctx.temp_image1)
    contours, _ = cv2.findContours(ctx.temp_image1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # take the 5 biggest areas
    contours = sorted(contours, key=lambda c: math.fabs(cv2.contourArea(c)), reverse=True)[:5]

    # approximate contours with poly line
    ctx.contours = [cv2.approxPolyDP(c, 2, True) for c in contours]


def find_convex_hull(ctx: Context):
    """
    find_convex_hull computes the hull and convex defects based on the contours
    :param ctx:
    :return:
    """
    if ctx.contours is None or len(ctx.contours) == 0:
        return 0

    max_defects = 0
    defects = None
    contour = None
    for c in ctx.contours:
        ctx.hull = cv2.convexHull(c, False, False, False)
        if ctx.hull is not None:
            d = cv2.convexityDefects(c, ctx.hull, None)
            if d is not None:
                if len(d) <= max_defects:
                    continue
                ctx.num_defects = len(d)
                max_defects = ctx.num_defects
                defects = d
                contour = c

    if defects is None:
        return

    # calculate hand center via mean of defect depth points
    x = 0
    y = 0
    for d in defects:
        depth_point = contour[d[0][2]]
        x += depth_point[0][0]
        y += depth_point[0][1]

    x = int(x / len(defects))
    y = int(y / len(defects))
    ctx.hand_center = (x, y)

    # calculate hand radius as mean of distances
    dist = 0
    for d in defects:
        depth_point = contour[d[0][2]]
        dx, dy = depth_point[0][0], depth_point[0][1]
        dist += math.sqrt(math.pow(x - dx, 2) * math.pow(y - dy, 2))
    ctx.hand_radius = int(dist / len(defects))


def find_fingers(ctx: Context):
    if ctx.contours is None or ctx.hull is None or ctx.hand_center is None:
        return

    if ctx.contour_idx >= len(ctx.contours):
        return

    ctx.fingers = [ctx.hand_center for _ in range(ctx.max_fingers)]
    max_distances = [0 for _ in range(ctx.max_fingers)]
    found_fingers = 0

    # looking at each point
    # such that d1, d2, d3 are consecutive neighbors
    d1, d2, d3 = 0, 0, 0
    contour = ctx.contours[ctx.contour_idx]
    cx, cy = ctx.hand_center

    p2 = None
    for p3 in contour:
        p3x, p3y = p3[0]

        # distance to center
        d3 = math.pow(cx - p3x, 2) + math.pow(cy - p3y, 2)

        # local maximum (d1 > 0 so we don't pass the test without 3 points)
        if d2 > d1 and d2 > d3:
            # only the best fingers shall survive
            for i in range(ctx.max_fingers):
                if max_distances[i] < d2:
                    if d2 > ctx.max_finger_dist:
                        print(d2)
                        continue
                    max_distances[i] = d2
                    ctx.fingers[i] = p2
                    found_fingers += 1
                    break
        if found_fingers < ctx.max_fingers:
            print("only %d/%d fingers" % (found_fingers, ctx.max_fingers))
        # keep track of last two distances and last point
        d1 = d2
        d2 = d3
        p2 = (p3x, p3y)


CONTOUR_COLORS = [(255, 50, 50), (0, 255, 50), (50, 0, 255), (255, 255, 0), (0, 255, 255)]


# display renders necessary on top of the original image and mutates ctx.image only
def display(ctx: Context):
    # draw contours each with their own color
    if ctx.contours is not None:
        for i in range(len(ctx.contours)):
            ctx.image = cv2.drawContours(ctx.image, ctx.contours, i, CONTOUR_COLORS[i])

    # draw a line from hand center to each finger
    if ctx.fingers is not None and ctx.hand_center is not None:
        cv2.cvtColor(ctx.filter_image, cv2.COLOR_GRAY2RGB, ctx.filter_image)
        for finger in ctx.fingers:
            cv2.circle(ctx.temp_image3, ctx.hand_center, ctx.hand_radius, (0, 255, 0))
            cv2.circle(ctx.temp_image3, ctx.hand_center, 10, (255, 0, 0), 5)
            cv2.circle(ctx.temp_image3, finger, 2, (255, 0, 0), 3)
            cv2.line(ctx.temp_image3, ctx.hand_center, finger, (255, 0, 255), 5)


def main():
    # detection_graph, sess = detector_utils.load_inference_graph()
    # sess = tf.Session(graph=detection_graph)

    cv2.namedWindow('hsv')
    cv2.namedWindow('fingers')
    cv2.namedWindow('filtered')

    capture = video_util.WebcamVideoStream(0, width=300, height=200).start()
    ctx = Context()

    def capture_skin_color(e, x, y, _flags, _param):
        if e != cv2.EVENT_LBUTTONDOWN or ctx.image is None:
            return
        ctx.skin_samples.append([ctx.image[y][x]])
        if len(ctx.skin_samples) > 10:
            ctx.skin_samples.remove(ctx.skin_samples[0])

    cv2.setMouseCallback('hsv', capture_skin_color)
    cv2.createTrackbar('tolerance', 'hsv', 100, 100, lambda v: ctx.__setattr__('tolerance', v))
    cv2.createTrackbar('contour', 'filtered', 0, 4, lambda v: ctx.__setattr__('contour', v))
    cv2.createTrackbar('max finger distance', 'fingers', 30000, 40000, lambda v: ctx.__setattr__('max_finger_dist', v))
    cv2.createTrackbar('max num fingers', 'fingers', 5, 20, lambda v: ctx.__setattr__('max_fingers', v))

    while 1:
        rawimg = capture.read()
        if rawimg is None:
            continue

        # OpenCV based filtering
        ctx.set_image(cv2.cvtColor(rawimg, cv2.COLOR_RGB2HSV))
        compute_skin_color(ctx)
        filter_and_threshold(ctx)

        find_contour(ctx)
        find_convex_hull(ctx)
        find_fingers(ctx)
        display(ctx)

        if ctx.image is not None:
            cv2.imshow('hsv', ctx.image)

        if ctx.filter_image is not None:
            print()
            cv2.imshow('filtered', ctx.filter_image)

        if ctx.temp_image3 is not None:
            cv2.imshow('fingers', ctx.temp_image3)

        cv2.waitKey(1)

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    capture.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
