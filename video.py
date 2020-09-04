import imageio
import os
import numpy as np
import sys

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2

# Use Agg backend for canvas
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import utils

class VideoRecorder(object):
    def __init__(self, root_dir, height=256, width=256, camera_id=0, fps=30):
        self.save_dir = utils.make_dir(root_dir, 'video') if root_dir else None
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.fps = fps
        self.frames = []

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled

    def record(self, env):
        if self.enabled:
            frame = env.render(mode='rgb_array')
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.save_dir, file_name)
            imageio.mimsave(path, self.frames, fps=self.fps)

class GridVideoRecorder(VideoRecorder):
    def __init__(self, root_dir, height=256, width=256, camera_id=0, fps=30):
        self.save_dir = utils.make_dir(root_dir, 'video') if root_dir else None
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.fps = fps
        self.frames = []

    def save(self, file_name):
        path = os.path.join(self.save_dir, file_name)
        # create OpenCV video writer
        video = None

        # loop over your images
        if not self.frames:
            return
        for i in range(len(self.frames)):
            fig = plt.figure()
            plt.imshow(self.frames[i])

            # put pixel buffer in numpy array
            canvas = FigureCanvas(fig)
            canvas.draw()
            mat = np.array(canvas.renderer._renderer)
            mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
            if video is None:
                video = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 5, (mat.shape[1], mat.shape[0]))
            # write frame to video
            video.write(mat)

        # close video writer
        #cv2.destroyAllWindows()
        video.release()

