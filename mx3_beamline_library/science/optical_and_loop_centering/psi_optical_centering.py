import datetime
import logging
import os
import pwd

import cv2
import numpy as np

logger = logging.getLogger("AlcDriver")


class loopImageProcessing(object):
    def __init__(self, image):
        # Kuba TODO: move huMomSubsets to separate routine.
        self.image = image
        # if the image is not given as numpy ndarray, try to open it for the FileExistsError
        if type(self.image) is not np.ndarray:
            self.filename = image

            self.image = cv2.imread(self.filename)
            if self.image is None:
                raise Exception("loopImageProcessing: cannot find image.")
            elif len(self.image.shape) < 3:
                raise Exception("loopImageProcessing: image is grayscale not RGB.")

        self.contour = None
        self.threshold = None
        self.huMoments = None

        self.adaptiveThreshold = False
        self.threshold = False
        self.dilate = False
        self.erode = False
        self.roi = False

    def __setOpenCVParams(self, beamline, zoomLevel):
        # Set of parameters for openCV image processing for each zoom level

        params_MX3 = {
            "1": {
                "adaptiveThreshold": True,
                "adaptConst": 3,
                "blockSize": 35,
                "dilate": True,
                "dilateIter": 1,
                "dilateKernel": np.ones((1, 1), np.uint8),
            },
        }

        paramsX06DA = {
            "-500.0": {
                "adaptiveThreshold": True,
                "adaptConst": 3,
                "blockSize": 35,
                "dilate": True,
                "dilateIter": 1,
                "dilateKernel": np.ones((1, 1), np.uint8),
            },
            "test": {
                "threshold": True,
                "thresholdValue": 200,
                "dilate": True,
                "dilateIter": 5,
                "dilateKernel": np.ones((3, 3), np.uint8),
            },
            "tensorflow": {
                "adaptiveThreshold": True,
                "adaptConst": 1,
                "blockSize": 121,
                "erode": True,
                "erodeIter": 2,
                "erodeKernel": np.ones((3, 3), np.uint8),
                "dilate": True,
                "dilateIter": 25,
                "dilateKernel": np.ones((3, 3), np.uint8),
            },
            "-208.0": {
                "adaptiveThreshold": True,
                "adaptConst": 1,
                "blockSize": 121,
                "erode": True,
                "erodeIter": 2,
                "erodeKernel": np.ones((3, 3), np.uint8),
                "dilate": True,
                "dilateIter": 1,
                "dilateKernel": np.ones((3, 3), np.uint8),
            },
        }

        paramsX06SA = {
            "topcam": {
                "threshold": True,
                "thresholdValue": 127,  # was 41
                "roi": True,
                "roiDimensions": [[125, 900], [200, 600]],  # was 150
                # 'roi': True, 'roiDimensions': [[220, 800], [100, 550]], # was 150
                "erode": False,
                "erodeIter": 3,
                "erodeKernel": np.ones((3, 3), np.uint8),
                "dilate": False,
                "dilateIter": 3,
                "dilateKernel": np.ones((7, 7), np.uint8),
            },
            "4.0": {
                "adaptiveThreshold": True,
                "adaptConst": 3,
                "blockSize": 35,
                "dilate": True,
                "dilateIter": 1,
                "dilateKernel": np.ones((1, 1), np.uint8),
            },
            "6.0": {
                "adaptiveThreshold": True,
                "adaptConst": 3,
                "blockSize": 35,
                "dilate": True,
                "dilateIter": 1,
                "dilateKernel": np.ones((1, 1), np.uint8),
            },
            "8.0": {
                "adaptiveThreshold": True,
                "adaptConst": 3,
                "blockSize": 35,
                "dilate": True,
                "dilateIter": 1,
                "dilateKernel": np.ones((1, 1), np.uint8),
            },
        }

        paramsX10SA = {
            "topcam": {
                "threshold": True,
                "thresholdValue": 127,
                "roi": True,
                "roiDimensions": [[50, 850], [150, 650]],  # [[100, 900], [100, 600]],
                "erode": False,
                "erodeIter": 3,
                "erodeKernel": np.ones((3, 3), np.uint8),
            },
            "1.0": {
                "adaptiveThreshold": True,
                "adaptConst": 3,
                "blockSize": 35,
                "dilate": True,
                "dilateIter": 1,
                "dilateKernel": np.ones((1, 1), np.uint8),
            },
            "200.0": {
                "adaptiveThreshold": True,
                "adaptConst": 3,
                "blockSize": 35,
                "dilate": True,
                "dilateIter": 1,
                "dilateKernel": np.ones((1, 1), np.uint8),
            },
            "400.0": {
                "adaptiveThreshold": True,
                "adaptConst": 3,
                "blockSize": 35,
                "dilate": True,
                "dilateIter": 1,
                "dilateKernel": np.ones((1, 1), np.uint8),
            },
            "600.0": {
                "adaptiveThreshold": True,
                "adaptConst": 3,
                "blockSize": 35,
                "dilate": True,
                "dilateIter": 1,
                "dilateKernel": np.ones((1, 1), np.uint8),
            },
            "700.0": {
                "adaptiveThreshold": True,
                "adaptConst": 3,
                "blockSize": 35,
                "dilate": True,
                "dilateIter": 1,
                "dilateKernel": np.ones((1, 1), np.uint8),
            },
            "800.0": {
                "adaptiveThreshold": True,
                "adaptConst": 3,
                "blockSize": 35,
                "dilate": True,
                "dilateIter": 1,
                "dilateKernel": np.ones((1, 1), np.uint8),
            },
            "900.0": {
                "adaptiveThreshold": True,
                "adaptConst": 3,
                "blockSize": 35,
                "dilate": True,
                "dilateIter": 1,
                "dilateKernel": np.ones((1, 1), np.uint8),
            },
            "950.0": {
                "adaptiveThreshold": True,
                "adaptConst": 3,
                "blockSize": 35,
                "dilate": True,
                "dilateIter": 1,
                "dilateKernel": np.ones((1, 1), np.uint8),
            },
        }

        X06SAbounding = {
            "6.0": {
                "adaptiveThreshold": True,
                "adaptConst": 1,
                "blockSize": 121,
                "dilate": True,
                "dilateIter": 2,
                "dilateKernel": np.ones((3, 3), np.uint8),
            }
        }

        X06DAbounding = {
            "-208.0": {
                "adaptiveThreshold": True,
                "adaptConst": 1,
                "blockSize": 49,
                "erode": False,
                "erodeIter": 2,
                "erodeKernel": np.ones((3, 3), np.uint8),
                "dilate": True,
                "dilateIter": 2,
                "dilateKernel": np.ones((3, 3), np.uint8),
            }
        }

        X10SAbounding = {
            "400.0": {
                "adaptiveThreshold": True,
                "adaptConst": 1,
                "blockSize": 121,
                "dilate": True,
                "dilateIter": 2,
                "dilateKernel": np.ones((3, 3), np.uint8),
            },
        }

        params = {
            "X06SA": paramsX06SA,
            "X06DA": paramsX06DA,
            "X10SA": paramsX10SA,
            "paramsX06SAbounding": X06SAbounding,
            "paramsX06DAbounding": X06DAbounding,
            "paramsX10SAbounding": X10SAbounding,
            "MX3": params_MX3,
        }

        self.adaptiveThreshold = params[beamline][zoomLevel].pop(
            "adaptiveThreshold", False
        )
        self.threshold = params[beamline][zoomLevel].pop("threshold", False)
        self.dilate = params[beamline][zoomLevel].pop("dilate", False)
        self.erode = params[beamline][zoomLevel].pop("erode", False)
        self.roi = params[beamline][zoomLevel].pop("roi", False)

        if self.threshold and self.adaptiveThreshold:
            raise Exception(
                "Cannot apply regular thresholding and adaptive thresholding at the same time."
            )

        # set image processing variables to None
        parsed_params = dict.fromkeys(
            [
                "adaptConst",
                "blockSize",
                "dilateIter",
                "dilateKernel",
                "thresholdValue",
                "erodeKernel",
                "dilateIter",
            ],
            None,
        )

        if self.adaptiveThreshold:
            try:
                parsed_params["adaptConst"] = params[beamline][zoomLevel]["adaptConst"]
                parsed_params["blockSize"] = params[beamline][zoomLevel]["blockSize"]
            except KeyError as e:
                raise Exception(
                    f"No adaptive Threshold parameters for zoomLevel {zoomLevel}. Error: {e}"
                )

        if self.threshold:
            try:
                parsed_params["thresholdValue"] = params[beamline][zoomLevel][
                    "thresholdValue"
                ]
            except KeyError as e:
                raise Exception(
                    f"No Threshold parameters for zoomLevel {zoomLevel}. Error: {e}"
                )

        if self.dilate:
            try:
                parsed_params["dilateKernel"] = params[beamline][zoomLevel][
                    "dilateKernel"
                ]
                parsed_params["dilateIter"] = params[beamline][zoomLevel]["dilateIter"]
            except KeyError as e:
                raise Exception(
                    f"No dilate parameters for zoomLevel {zoomLevel}. Error: {e}"
                )

        if self.erode:
            try:
                parsed_params["erodeKernel"] = params[beamline][zoomLevel][
                    "erodeKernel"
                ]
                parsed_params["erodeIter"] = params[beamline][zoomLevel]["erodeIter"]
            except KeyError as e:
                raise Exception(
                    f"No erode parameters for zoomLevel {zoomLevel}. Error: {e}"
                )

        if self.roi:
            try:
                parsed_params["roiDimensions"] = params[beamline][zoomLevel][
                    "roiDimensions"
                ]
            except KeyError as e:
                raise Exception(
                    f"No ROI parameters for zoomLevel {zoomLevel}. Error: {e}"
                )

        # return adaptConst, blockSize, dilate, dilateKernel, dilateIter
        return parsed_params

    def __findTwoBiggestContours(self, cnts):
        """Return two biggest contours"""
        maxAreas = [-1, -1]
        maxCnts = [0, 0]
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area > maxAreas[0]:
                maxAreas[1] = maxAreas[0]
                maxCnts[1] = maxCnts[0]
                maxAreas[0] = area
                maxCnts[0] = cnt
            elif area > maxAreas[1]:
                maxAreas[1] = area
                maxCnts[1] = cnt
        return (maxCnts), maxAreas

    def __joinContours(self, cnts, minArea=700):
        """Goes through the list of contours, removes the contours smaller
        than minArea, and joins the remaining contours"""

        c = [i for i, cnt in enumerate(cnts) if cv2.contourArea(cnt) > minArea]

        if len(c) == 0:
            raise Exception("Did not find loop contour.")

        val = []
        for i in c:
            val.append(cnts[i])

        cont = np.vstack(val)

        return cont

    def __cv2wait(self, wait=0):
        cv2.waitKey(wait)
        cv2.destroyAllWindows()

    def findContour(
        self,
        zoom,
        beamline=None,
        hull=False,
        calc_hu_moments=False,
        huMomSubset=(0, 1, 2, 3, 4, 5, 6),
    ):

        if beamline is None:
            beamline = os.getenv("BEAMLINE_XNAME", "bogus").upper()

        zoomLevel = str(zoom)
        nFeatures = len(huMomSubset)
        # adaptConst, blockSize, dilate, dilateKernel, dilateIter = self.__setOpenCVParams(
        # beamline, zoomLevel)
        image_processing_params = self.__setOpenCVParams(beamline, zoomLevel)

        if self.roi:
            roi_x = image_processing_params["roiDimensions"][0]  # list [xstart, xend]
            roi_y = image_processing_params["roiDimensions"][1]  # list [ystart, yend]
            self.image = self.imageROI(self.image, roi_x, roi_y)

        try:
            grayimg = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        except cv2.error:
            grayimg = self.image

        if self.adaptiveThreshold:
            thresh = cv2.adaptiveThreshold(
                grayimg,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                image_processing_params["blockSize"],
                image_processing_params["adaptConst"],
            )
        elif self.threshold:
            ret, thresh = cv2.threshold(
                grayimg,
                image_processing_params["thresholdValue"],
                255,
                cv2.THRESH_BINARY,
            )
            # ret, thresh = cv2.threshold(grayimg,
            # image_processing_params['thresholdValue'],
            # 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            raise Exception(
                "Image processing type was not correctly specified. Choose Normal "
                "Thresholding or Adaptive Thresholding"
            )

        if self.erode:
            erosion = cv2.erode(
                thresh,
                image_processing_params["erodeKernel"],
                iterations=image_processing_params["erodeIter"],
            )
        else:
            erosion = thresh  # so next line holds irrespective of the 'if' result

        self.threshold = erosion.copy()

        if self.dilate:
            dilation = cv2.dilate(
                erosion,
                image_processing_params["dilateKernel"],
                iterations=image_processing_params["dilateIter"],
            )
        else:
            dilation = erosion  # so next line holds irrespective of the 'if' result

        self.threshold = dilation.copy()

        if cv2.__version__[0] == "3":
            _, contours, h = cv2.findContours(
                erosion, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
            )
        elif cv2.__version__[0] == "4":
            contours, h = cv2.findContours(
                erosion, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
            )
        else:
            raise Exception(
                f"Unsupported version of openCV: {cv2.__version__}. "
                "Supported versions 3.* & 4.*"
            )

        if hull:
            # Find hull of the sum of the contours
            self.contour = self.__joinContours(contours)
            self.hull = cv2.convexHull(self.contour, returnPoints=True)
            # Use hull as contour from now on
            self.contour = self.hull.copy()
        else:
            # The contour will be set to the biggest contour
            cnts, areas = self.__findTwoBiggestContours(contours)
            self.contour = cnts[0]

        # TODO: Kuba - how well are HuMoments are defined for hull
        # TODO: Kuba - this can be seperate method
        if calc_hu_moments:
            mom = cv2.moments(self.contour)
            huMoments = cv2.HuMoments(mom)
            # Reduced Hu Moments
            huMomentsSmall = np.zeros((1, nFeatures), dtype=np.float32)
            for i, f in enumerate(huMomSubset):
                huMomentsSmall[0][i] = huMoments[f][0]
            huMoments = huMomentsSmall

            self.huMoments = huMoments

    def contourArea(self):
        return cv2.contourArea(self.contour)

    def contourCircularity(self):
        """Measures circularity of the contour [0, 1]
        https://www.programcreek.com/python/example/89409/cv2.fitEllipse
        """
        arclen = cv2.arcLength(self.contour, True)
        pi_4 = np.pi * 4
        circularity = (pi_4 * self.contourArea()) / (arclen * arclen)
        return circularity

    def findTip(self):
        if self.contour is None:
            raise Exception("Image needs to be processed first")

        tmp = tuple(self.contour[self.contour[:, :, 0].argmin()][0])
        return np.array(tmp)

    def findExtremes(self):
        if self.contour is None:
            raise Exception("Image needs to be processed first")

        leftmost = np.array(self.contour[self.contour[:, :, 0].argmin()][0])
        rightmost = np.array(self.contour[self.contour[:, :, 0].argmax()][0])
        topmost = np.array(self.contour[self.contour[:, :, 1].argmin()][0])
        bottommost = np.array(self.contour[self.contour[:, :, 1].argmax()][0])
        return {
            "top": topmost,
            "bottom": bottommost,
            "right": rightmost,
            "left": leftmost,
        }

    def fitEllipse(self, draw_bounding_boxes=False, draw_contours=False):
        """
        Fit ellipse around the loop, and return minimum rectangle encompassing that ellipse
        :param draw_bounding_boxes:
        :param draw_contours:
        :return: :return: rectangle - dictionary,
        {'top_left': np.array([x_t, y_t]), 'bottom_right': np.array([x_b, y_b])}
        """
        img_height, img_width = self.image.shape[:2]

        # Fit ellipse
        ellipse = cv2.fitEllipse(
            self.contour
        )  # output -> ellipse =  ((x, y), (MA, ma), angle) -> MA-major axis, ma-minor

        # https://docs.opencv.org/master/dd/d49/tutorial_py_contour_features.html
        # Find rotated rectangle enclosing ellipse
        box = cv2.boxPoints(ellipse)
        box = np.int0(box)

        # find straight rectangle enclosing rotated rectangle
        x, y, w, h = cv2.boundingRect(box)
        top_left = np.array([x, y])
        # check to ensure bottom right value is larger than top left value
        if x < img_width and y < img_height:
            bottom_right = np.array([min(x + w, img_width), min(y + h, img_height)])
        else:
            bottom_right = np.array([(x + w), (y + h)])

        if draw_contours:
            cv2.drawContours(self.image, [self.contour], 0, (0, 255, 0), 3)

        if draw_bounding_boxes:
            cv2.ellipse(self.image, ellipse, (0, 0, 255), thickness=3)
            cv2.drawContours(self.image, [box], 0, (0, 0, 255), 2)

        return dict(top_left=top_left, bottom_right=bottom_right)

    def detectCircles(self):
        """
        Detect circles in the image
        https://www.pyimagesearch.com/2014/07/21/detecting-circles-images-using-opencv-hough-circles/
        :return:
        """  # noqa
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        accumulator = 1.0  # the higher the most likely to find a circle
        min_dist = 100
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, accumulator, min_dist)

        if circles is not None:
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                # draw the circle in the output image, then draw a rectangle
                # corresponding to the center of the circle
                cv2.circle(self.image, (x, y), r, (255, 255, 255), 4)

    def fitRectangle(self, draw_rectangle_box=False):
        """
        Based on three extreme points of loop contour (top, bottom, leftmost)
        finds rectangle bounding the loop.

        - the height of the box is the distance between top-most and bottom-most-point
        - the width of the the  box is the distance between left-most point and
          top-most or bottom-most point, which ever is bigger, multiplied by two.

        :return: rectangle - dictionary,
            {'top_left': np.array([x_t, y_t]), 'bottom_right': np.array([x_b, y_b])}
            => keys [top_left, bottom_left, top_right, bottom_right].
            => values numpy arrays with coordinates np.array[x,y]

        Note: using self.showExtremes(draw_loop_bounding_box=True) will draw the points
        on the image to draw rectangle itself
        """
        extremes = self.findExtremes()

        rectangle = {}
        img_height, img_width = self.image.shape[:2]

        rectangle["top_left"] = np.array([extremes["left"][0], extremes["top"][1]])

        # Create rectangle based on extreme point that more away from the tip
        x_top_right_t = extremes["left"][0] + 2 * (
            extremes["top"][0] - extremes["left"][0]
        )
        x_top_right_b = extremes["left"][0] + 2 * (
            extremes["bottom"][0] - extremes["left"][0]
        )

        x_top_right = max(x_top_right_t, x_top_right_b)

        # If calculations are more than image width, use image width
        x_top_right = min(x_top_right, img_width)

        rectangle["bottom_right"] = np.array([x_top_right, extremes["bottom"][1]])

        if draw_rectangle_box:
            cv2.rectangle(
                self.image,
                tuple(rectangle["top_left"]),
                tuple(rectangle["bottom_right"]),
                (0, 255, 0),
                3,
            )

        return rectangle

    def findLoopBoundingBox(
        self, beamline=None, draw_rectangle_boxes=False, draw_ellipse_boxes=False
    ):
        """
        Finds binding box trying to fit rectangle and ellipse to contour of the loop.
        :param: draw_rectangle_boxes: if True, box from fitRectangle will be drawn on the image
        :param: draw_ellipse_boxes: if True, box from fitEllipse will be drawn on the image
        :return: corners of the bounding box, dict, {'top_left': [], 'bottom_right': []}
        """
        box_from_rectangle = self.fitRectangle(draw_rectangle_box=draw_rectangle_boxes)
        # box_from_ellipse = self.fitEllipse(draw_bounding_boxes=draw_ellipse_boxes)

        # Choose box that with edge closer to the image boundary
        return box_from_rectangle

    def imageROI(self, image, x, y):
        """
        :param image:
        :param x: [xstart, xend], horizontal dimension  of the images
        :param y: [ystart, yend], vertical dimension of the image
        :return:
        """
        # using numpy notation, x and y are flipped.
        return image[y[0] : y[1], x[0] : x[1]]

    def findClosestPointOnContour(self, point):
        # Finds the nearest contour point to the "point" given as argument, along the Y axis
        if self.contour is None:
            raise Exception("Image needs to be processed first")

        x = point[0]
        y = point[1]

        delta = (
            (self.contour[:, :, 0] - x) ** 2 + (self.contour[:, :, 1] - y) ** 2
        ) ** 0.5
        closest = delta.argmin()
        p = self.contour[closest][0]
        return p

    def findEllipse(self):
        ellipse = cv2.fitEllipse(self.contour)
        cv2.ellipse(self.image, ellipse, (0, 255, 0), 2)

    def showImage(self, wait=0):
        cv2.imshow("img", self.image)
        self.__cv2wait(wait=wait)

    def showBackground(self, wait=0):
        cv2.imshow("img", self.background)
        self.__cv2wait(wait=wait)

    def showContour(self, wait=0, save_img=False):
        if self.contour is None:
            raise Exception("Image needs to be processed first")
        cv2.drawContours(self.image, [self.contour], 0, (0, 255, 0), 3)

        if save_img:
            path = self.saveImage()
            return path
        else:
            self.showImage(wait=wait)

    def showProcessedImage(self, wait=0):
        if self.threshold is None:
            raise Exception("Image needs to be processed first")
        self.image = self.threshold
        self.showImage(wait=wait)

    def saveImage(
        self,
        directory="/scratch/alc_data/alc_images",
        filename=None,
        create_new_dir=True,
        user=None,
    ):

        if user:  # if user is defined, change permission to this user
            uinfo = pwd.getpwnam(user)
            logger.debug("userinfo: {str(uinfo)}")

            logger.debug(
                f"dropping group privileges from gid={os.getegid()} to gid={uinfo.pw_gid}"
            )
            os.setegid(uinfo.pw_gid)
            logger.debug(
                f"dropping user privileges from uid={os.geteuid()} to uid={uinfo.pw_uid}"
            )
            os.seteuid(uinfo.pw_uid)

        if create_new_dir:
            dirname = datetime.datetime.now().strftime("%Y%m%d")
            directory = os.path.join(directory, dirname)

        if filename is None:
            filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S.jpg")

        try:
            if not os.path.exists(directory):
                os.makedirs(directory)

            filepath = os.path.join(directory, filename)
            cv2.imwrite(filepath, self.image)

        except Exception as e:
            raise Exception(f"Cannot write image of sample to {e}")

        if user:
            logger.debug("re-gaining root privileges")
            os.seteuid(0)
            os.setegid(0)

        return filepath

    def saveContour(self, directory="/scratch/kaminski_j/alc/images"):
        if self.contour is None:
            raise Exception("Image needs to be processed first")

        dirname = datetime.datetime.now().strftime("%Y%m%d")
        filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S.jpg")
        directory = os.path.join(directory, dirname)
        directory_raw = os.path.join(directory, "raw")
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)

            if not os.path.exists(directory_raw):
                os.makedirs(directory_raw)

            filepath = os.path.join(directory, filename)
            filepath_raw = os.path.join(directory_raw, filename)
            # Copy the image to avoid drawing contour on original image
            newImg = self.image.copy()
            cv2.drawContours(newImg, [self.contour], 0, (0, 255, 0), 3)
            cv2.imwrite(filepath, newImg)
            cv2.imwrite(filepath_raw, self.image)

        except Exception as e:
            raise Exception(f"Cannot write image with contours to {e}")

        return filepath

    def showTip(self, wait=0):
        tipPos = tuple(self.findTip())
        cv2.circle(self.image, tipPos, 8, (0, 0, 255), -1)
        cv2.drawContours(self.image, [self.contour], 0, (0, 255, 0), 3)
        print("Tip Position", tipPos)
        self.showImage(wait=wait)

    def showExtremes(self, wait=0, draw_loop_bounding_box=False, show_img=True):
        """
        :param wait:
        :param draw_loop_bounding_box: If true, additionally uses
            self.findLoopBoundingBox to and draws them
        :return:
        """
        extremes = self.findExtremes()

        cv2.circle(self.image, tuple(extremes["left"]), 8, (0, 0, 255), -1)
        cv2.circle(self.image, tuple(extremes["top"]), 8, (0, 0, 255), -1)
        cv2.circle(self.image, tuple(extremes["bottom"]), 8, (0, 0, 255), -1)

        if draw_loop_bounding_box:
            box = self.findLoopBoundingBox()
            cv2.circle(self.image, tuple(box["top_left"]), 8, (255, 0, 0), -1)
            cv2.circle(self.image, tuple(box["top_right"]), 8, (255, 0, 0), -1)
            cv2.circle(self.image, tuple(box["bottom_left"]), 8, (255, 0, 0), -1)
            cv2.circle(self.image, tuple(box["bottom_right"]), 8, (255, 0, 0), -1)

        cv2.drawContours(self.image, [self.contour], 0, (0, 255, 0), 3)
        if show_img:
            self.showImage(wait=wait)

    def showLoopBoundingBox(
        self,
        wait=0,
        show_extremes=False,
        draw_contours=False,
        draw_rectangle_boxes=False,
        draw_ellipse_boxes=False,
        save_img=False,
        directory=None,
    ):
        """

        :param wait:
        :param show_extremes: additionally shows extreme points on image
        :param save_img:  saves image in directory. If False,  displays on screen
        :param directory: directory where to save image. If None, image is displayed on screen
        :param draw_rectangle_boxes - if true will draw bounding box resulting from rectangle
        :param draw_ellipse_boxes - if True and bounding box resulting from ellipse fit
        :return:
        """
        box = self.findLoopBoundingBox(
            draw_ellipse_boxes=draw_ellipse_boxes,
            draw_rectangle_boxes=draw_rectangle_boxes,
        )

        cv2.rectangle(
            self.image,
            tuple(box["top_left"]),
            tuple(box["bottom_right"]),
            (255, 0, 0),
            3,
        )
        if show_extremes:
            extremes = self.findExtremes()
            cv2.circle(self.image, tuple(extremes["left"]), 8, (0, 0, 255), -1)
            cv2.circle(self.image, tuple(extremes["top"]), 8, (0, 0, 255), -1)
            cv2.circle(self.image, tuple(extremes["bottom"]), 8, (0, 0, 255), -1)
        if draw_contours:
            cv2.drawContours(self.image, [self.contour], 0, (0, 255, 0), 3)
        if save_img:
            path = self.saveImage(directory=directory)
            return path
        else:
            self.showImage(wait=wait)

    def floodFill(self):
        im_floodfill = self.threshold.copy()
        # Mask used to flood filling.
        # Notice the size needs to be 2 pixels than the image.
        h, w = self.threshold.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)

        # Floodfill from point (0, 0)
        cv2.floodFill(im_floodfill, mask, (0, 0), 255)

        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)

        # Combine the two images to get the foreground.
        self.flooded = self.threshold | im_floodfill_inv

    def showFlooded(self, wait=0):
        if self.threshold is None:
            raise Exception("Image needs to be processed first")
        self.image = self.flooded
        self.showImage(wait=wait)

    def saveFlooded(self, directory, filename):
        self.image = self.flooded.copy()
        self.saveImage(directory, filename, create_new_dir=False)

    def drawGridLines(self, lines, show_image=False):
        """
        From list of line coordinates draws a grid on current image
        :param lines: list it list [[[x_start, y_start], [x_end, y_end ]]]
        :param show_image:
        :return:
        """

        for line in lines:
            start = tuple(line[0])
            end = tuple(line[1])
            cv2.line(self.image, start, end, color=(0, 255, 0, 2))

        if show_image:
            self.showImage(0)

    def drawGridPoints(self, grid, show_image=False):
        """
        From a list of grid points coordinates (middle of the grid cells) draws
        points of the current image
        :param lines: list it list [[[x_start, y_start], [x_end, y_end ]]]
        :param show_image:
        :return:
        """

        for point in grid:
            p = (int(point[0]), int(point[1]))
            cv2.circle(self.image, p, 4, (0, 0, 255), -1)

        if show_image:
            self.showImage(0)

    def drawBeamBox(self, x, y, width_px, height_px, show_image=False):
        """
        Draws beambox on the image
        :param x: x coordinate of beam center
        :param y: y coordinate of beam center
        :param width_px: width of the beam box in pixels
        :param height_px: height of the beam box in pixels
        :param show_image
        :return:
        """

        top_left = (int(x - (width_px / 2)), int(y - (height_px / 2)))
        bottom_right = (int(x + (width_px / 2)), int(y + (height_px / 2)))

        cv2.rectangle(self.image, tuple(top_left), tuple(bottom_right), (255, 0, 0), 2)

        if show_image:
            self.showImage(0)

    def rasterHeatMap(
        self,
        grid_pos,
        grid_hits,
        grid_cell_width_px,
        grid_cell_height_px,
        beamline,
        max_value,
        grid_lines_overlay=None,
        show_image=False,
    ):
        """
        Generates raster heatmap on image.
        :param grid_pos: list,  in pixels, list of lists [[x,y], [x,y], [x,y]].
        :param grid_hits: numpy array, n_rows, n_cols. Array with number of spots
            at given grid position
        :param grid_cell_width_px: in pixels, width of one grid cell
        :param grid_cell_height_px: in pixels, height of one grid cell
        :param beamline: which beamline ALC is running, used to query hub specific values
        :param max_value: best box value from SDU (diffCenter, rasterGridAnalyser),
            put in image watermark
        :param grid_lines_overlay: optional, coordinates of the grid lines to draw as
            cell separators
        :param show_image: True/False. If true will show image
        :return:
        """
        # in opencv hue is in range [0, 179], saturation and value [0, 255]
        try:
            import hubclient

            hub = hubclient.Hub(beamline.lower())
        except ModuleNotFoundError:
            hub = False
        max_hue = 128  # 128 HSV(128, 255, 255) is dark blue, HSV(0, 255, 255) is red

        grid_hits = np.array(grid_hits)
        max_val = int(max_value)

        # Normalize raster hits to HUE value between 0 (red) and  255 (violet)
        if max_val > 0:
            factor = max_val / max_hue
            grid_hits = grid_hits / factor
        grid_hits = grid_hits.astype(int)
        try:
            spot_threshold = int(hub.getd("sdu_grid_threshold"))
        except AttributeError:
            spot_threshold = 20

        if max_val >= spot_threshold:
            # Invert values so max hit is red (hue=0), no hit is violet (hue = 255)
            grid_hits = abs(grid_hits - max_hue)
        else:
            # Set all below threshold to be violet (hsv = 128)
            grid_hits[grid_hits > -1] = 128
        grid_hits = grid_hits.flatten()

        half_box_x = int(grid_cell_width_px / 2)
        half_box_y = int(grid_cell_height_px / 2)

        # Start creating overlays on original image
        for i, point in enumerate(grid_pos):
            overlay = self.image.copy()

            x = point[0]
            y = point[1]
            hue = grid_hits[i]

            top_left = (x - half_box_x, y - half_box_y)
            bottom_right = (x + half_box_x, y + half_box_y)

            top_left = [int(i) for i in top_left]
            bottom_right = [int(i) for i in bottom_right]

            hsv_color = np.uint8([[[hue, 255, 255]]])
            bgr_color = cv2.cvtColor(
                hsv_color, cv2.COLOR_HSV2BGR
            )  # -> array([[[ b, g, r]]], dtype=uint8)
            color = tuple(bgr_color[0][0].tolist())
            # Draw rectangle filled with bgr_color
            cv2.rectangle(overlay, tuple(top_left), tuple(bottom_right), color, -1)

            # Overlay rectangle on loop image
            # https://www.pyimagesearch.com/2016/03/07/transparent-overlays-with-opencv/
            alpha = 0.5  # transparency level
            cv2.addWeighted(overlay, alpha, self.image, 1 - alpha, 0, self.image)

        # Overlay the spot total
        cv2.putText(
            self.image,
            "Raster spot counts",
            (25, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
        )
        cv2.putText(
            self.image,
            f"Best gridbox: {max_val}",
            (25, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
        )
        cv2.putText(
            self.image,
            f"SDU threshold: {spot_threshold}",
            (25, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
        )

        if grid_lines_overlay:
            self.drawGridLines(grid_lines_overlay)

        if show_image:
            self.showImage(0)

    def beamlineWatermark(
        self, beamline, x, y, width_px, height_px, beam_width, beam_height
    ):
        top_left = (int(x - (width_px / 2)), int(y - (height_px / 2)))
        bottom_right = (int(x + (width_px / 2)), int(y + (height_px / 2)))

        cv2.rectangle(self.image, tuple(top_left), tuple(bottom_right), (255, 0, 0), 2)

        cv2.putText(
            self.image,
            f"Beamline: {beamline}",
            (25, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
        )
        cv2.putText(
            self.image,
            f"Beam size: {int(beam_width)}x{int(beam_height)}um",
            (25, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
        )


if __name__ == "__main__":
    from os import environ

    environ["BL_ACTIVE"] = "True"
    environ["BLUESKY_DEBUG_CALLBACKS"] = "1"
    environ["SETTLE_TIME"] = "0.2"

    import time

    import matplotlib.pyplot as plt
    import numpy.typing as npt

    from mx3_beamline_library.devices import detectors, motors
    from mx3_beamline_library.devices.classes.detectors import BlackFlyCam

    testrig = motors.testrig
    motor_x = testrig.x
    motor_x.wait_for_connection()
    motor_z = testrig.z
    motor_z.wait_for_connection()
    motor_y = testrig.y
    motor_y.wait_for_connection()
    motor_phi = testrig.phi
    motor_phi.wait_for_connection()

    motor_y.move(-0.2, wait=True)
    motor_x.move(0, wait=True)
    motor_phi.move(0, wait=True)

    def take_snapshot_extremes(
        camera: BlackFlyCam, filename: str, screen_coordinates: dict
    ) -> None:
        """
        Saves an image given the ophyd camera object,
        and draws a red cross at the screen_coordinates.


        Parameters
        ----------
        camera : BlackFlyCam
            A blackfly camera ophyd device
        filename : str
            The filename
        screen_coordinates : tuple[int, int], optional
            The screen coordinates, by default (612, 512)

        Returns
        -------
        None
        """
        plt.figure()
        array_data: npt.NDArray = camera.array_data.get()
        data = array_data.reshape(
            camera.height.get(), camera.width.get(), camera.depth.get()
        )
        plt.imshow(data)
        plt.scatter(
            screen_coordinates["top"][0],
            screen_coordinates["top"][1],
            s=200,
            c="r",
            marker="+",
        )
        plt.scatter(
            screen_coordinates["bottom"][0],
            screen_coordinates["bottom"][1],
            s=200,
            c="r",
            marker="+",
        )
        plt.scatter(
            screen_coordinates["right"][0],
            screen_coordinates["right"][1],
            s=200,
            c="r",
            marker="+",
        )
        plt.scatter(
            screen_coordinates["left"][0],
            screen_coordinates["left"][1],
            s=200,
            c="r",
            marker="+",
        )

        plt.savefig(filename)
        plt.close()

    camera = detectors.blackfly_camera
    array_data: npt.NDArray = camera.array_data.get()
    data = array_data.reshape(
        camera.height.get(), camera.width.get(), camera.depth.get()
    ).astype(
        np.uint8
    )  # the code only works with np.uint8 data types
    print(data.shape)

    t = time.time()

    procImg = loopImageProcessing(data)
    procImg.findContour(zoom="-208.0", beamline="X06DA")
    tip = procImg.findTip()
    extremes = procImg.findExtremes()
    print("time to find extremes:", time.time() - t)
    rectangle_coordinates = procImg.fitRectangle()
    print("tip:", tip)

    print("extremes:", extremes)
    print("rectangle_limits", rectangle_coordinates)

    take_snapshot_extremes(
        camera,
        "extremes",
        extremes,
    )
