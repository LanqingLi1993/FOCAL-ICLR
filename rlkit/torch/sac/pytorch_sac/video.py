import imageio
import os
import numpy as np
import sys

from . import utils

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

    def record(self, env, mujoco):
        if self.enabled:
            if mujoco:
                frame = env.render(mode='rgb_array',
                                   height=self.height, # for mujoco env
                                   width=self.width, # for mujoco env
                                   camera_id=self.camera_id) # for mujoco env
                self.frames.append(frame)
            else:
                env.render()


    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.save_dir, file_name)
            imageio.mimsave(path, self.frames, fps=self.fps)
