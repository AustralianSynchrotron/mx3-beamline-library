import cv2
from mx3_beamline_library.devices.motors import md3
import numpy as np

from mx3_beamline_library.plans.image_analysis import get_image_from_md3_camera
import matplotlib.pyplot as plt
from mx3_beamline_library.plans.plan_stubs import md3_move


class ImageAnalysis:
    def sobel_edge_detector(self, gray_image, threshold=10):
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobel_y ** 2)
        edges = magnitude > threshold
        return edges

    def move_to_center(self, x, y):
        current_position = [
            md3.alignment_y.position,
            md3.plate_translation.position
        ]
        print(f"Current position: {current_position}")
        newx = current_position[1] - x
        newy = current_position[0] + y
        print(f"Moving to: x={newx}, y={newy}")
        yield from md3_move(
            md3.alignment_y,newy,
            md3.plate_translation,newx
        )

    def run(self, reference):
        img = get_image_from_md3_camera(np.uint8)
        gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edge_image = self.sobel_edge_detector(gray_image, threshold=12)

        template_img = cv2.imread(reference)
        gray_template = cv2.cvtColor(template_img, cv2.COLOR_RGB2GRAY)
        template_edge = self.sobel_edge_detector(gray_template, threshold=30)

        edge_image_uint8 = (edge_image * 255).astype(np.uint8)
        template_edge_uint8 = (template_edge * 255).astype(np.uint8)

        result = cv2.matchTemplate(edge_image_uint8, template_edge_uint8, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        h, w = template_edge.shape
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        print(f"Match location: {max_loc}")

        offsetx = (max_loc[0] - 530) / 513
        offsety = (max_loc[1] - 460) / 527
        print(f"Offset: x={offsetx:.3f} mm, y={offsety:.3f} mm")

        yield from self.move_to_center(offsetx, offsety)

        cv2.rectangle(edge_image_uint8, top_left, bottom_right, (255, 0, 0), 2)
        plt.imshow(edge_image_uint8, cmap='gray')
        plt.axis('off')
        plt.show()
