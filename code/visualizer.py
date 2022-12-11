from PIL import Image
# import requests
import io
import cv2 as cv
from matplotlib import pyplot as plt
from PIL import ImageDraw
from PIL import ImageFont
import numpy as np
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose


# 分类结果可视化
class PoseClassificationVisualizer(object):
    """Keeps track of claassifcations for every frame and renders them."""

    def __init__(self,
                 class_name,
                 plot_location_x=0.03,
                 plot_location_y=0.03,
                 plot_max_width=0.4,
                 plot_max_height=0.4,
                 plot_figsize=(9, 4),
                 plot_x_max=None,
                 plot_y_max=None,
                 counter_location_x=0.85,
                 counter_location_y=0.05,
                 #                counter_font_path='https://github.com/googlefonts/roboto/blob/main/src/hinted/Roboto-Regular.ttf?raw=true',
                 counter_font_color='red',
                 counter_font_size=0.15):
        self._class_name = class_name
        self._plot_location_x = plot_location_x
        self._plot_location_y = plot_location_y
        self._plot_max_width = plot_max_width
        self._plot_max_height = plot_max_height
        self._plot_figsize = plot_figsize
        self._plot_x_max = plot_x_max
        self._plot_y_max = plot_y_max
        self._counter_location_x = counter_location_x
        self._counter_location_y = counter_location_y
        #     self._counter_font_path = counter_font_path
        self._counter_font_color = counter_font_color
        self._counter_font_size = counter_font_size

        self._counter_font = None

        self._pose_classification_history = []
        self._pose_classification_filtered_history = []

        self.result_img = 255 * np.ones((600, 700, 3)).astype(np.uint8)

    def __call__(self,
                 frame,
                 pose_classification,
                 pose_classification_filtered,
                 repetitions_count,
                 pose_landmarks):
        """Renders pose classifcation and counter until given frame."""
        # Extend classification history.
        self._pose_classification_history.append(pose_classification)
        self._pose_classification_filtered_history.append(pose_classification_filtered)

        # Output frame with classification plot and counter.
        output_img = Image.fromarray(frame)

        output_width = output_img.size[0]
        output_height = output_img.size[1]

        # Draw the plot.
        img1 = self._plot_classification_history(output_width, output_height)
        img2 = Image.fromarray(self._plot_3D_pose(pose_landmarks))
        img1.thumbnail((int(output_width * self._plot_max_width),
                       int(output_height * self._plot_max_height)),
                      Image.ANTIALIAS)
        img2.thumbnail((int(output_width * self._plot_max_width),
                        int(output_height * self._plot_max_height)),
                       Image.ANTIALIAS)
        output_img.paste(img1,
                         (int(output_width * self._plot_location_x),
                          int(output_height * self._plot_location_y)))
        output_img.paste(img2,
                         (int(output_width * self._plot_location_x),
                          int(output_height * (0.6 - self._plot_location_y))))

        # Draw the count.
        output_img_draw = ImageDraw.Draw(output_img)
        if self._counter_font is None:
            font_size = int(output_height * self._counter_font_size)
            #       font_request = requests.get(self._counter_font_path, allow_redirects=True)
            self._counter_font = ImageFont.truetype('Roboto-Regular.ttf', size=font_size)
        output_img_draw.text((output_width * self._counter_location_x,
                              output_height * self._counter_location_y),
                             str(repetitions_count),
                             font=self._counter_font,
                             fill=self._counter_font_color)

        return output_img

    def _plot_classification_history(self, output_width, output_height):
        fig = plt.figure(figsize=self._plot_figsize)
        for class_name in self._class_name:
            for classification_history in [self._pose_classification_history,
                                           self._pose_classification_filtered_history]:
                y = []
                for classification in classification_history:
                    if classification is None:
                        y.append(None)
                    elif class_name in classification:
                        y.append(classification[class_name])
                    else:
                        y.append(0)
                plt.plot(y, linewidth=7)

        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Frame')
        plt.ylabel('Confidence')
        # plt.title('Classification history for `{}`'.format(self._class_name))
        # plt.legend(loc='upper right')

        if self._plot_y_max is not None:
            plt.ylim(top=self._plot_y_max)
        if self._plot_x_max is not None:
            plt.xlim(right=self._plot_x_max)

        # Convert plot to image.
        buf = io.BytesIO()
        dpi = min(
            output_width * self._plot_max_width / float(self._plot_figsize[0]),
            output_height * self._plot_max_height / float(self._plot_figsize[1]))
        fig.savefig(buf, dpi=dpi)
        buf.seek(0)
        img = Image.open(buf)
        plt.close()

        return img


    def _plot_3D_pose(self, pose_landmarks):
        if pose_landmarks == None:
            return self.result_img
        result_img = np.array(mp_drawing.plot_landmarks(pose_landmarks, mp_pose.POSE_CONNECTIONS))
        white_region_w = int(result_img.shape[1] * 0.15)
        white_region_h = int(result_img.shape[0] * 0.2)
        result_img = result_img[white_region_h:result_img.shape[0] - white_region_h,
                     white_region_w:result_img.shape[1] - white_region_w]
        self.result_img = result_img
        return result_img
