# Copyright (c) Facebook, Inc. and its affiliates.

"""
Renders mesh using OpenDr / Pytorch-3D for visualization.
"""

import sys
import numpy as np
import cv2
import pdb

g_valid_visualize = False
try:
    from .od_renderer import OpendrRenderer
    g_valid_visualize = True
except ImportError:
    print("Cannot import OpendrRenderer")

try:
    from .p3d_renderer import Pytorch3dRenderer
    g_valid_visualize = True
except ImportError:
    print("Cannot import Pytorch3dRenderer")

assert g_valid_visualize, "You should import either OpenDR or Pytorch3D"

from renderer.image_utils import draw_raw_bbox, draw_hand_bbox, draw_body_bbox, draw_arm_pose


colors = {
    # colorbline/print/copy safe:
    'light_gray':  [0.9, 0.9, 0.9],
    'light_purple':  [0.8, 0.53, 0.53],
    'light_green': [166/255.0, 178/255.0, 30/255.0],
    'light_blue': [0.65098039, 0.74117647, 0.85882353],
}


class Visualizer(object):

    def __init__(self, renderer_backend):
        self.input_size = 1920

        # set-up renderer
        assert renderer_backend in ['opendr', 'pytorch3d']
        if renderer_backend == 'opendr':
            self.renderer = OpendrRenderer(
                img_size=self.input_size, 
                mesh_color=colors['light_purple'])
        else:
            self.renderer = Pytorch3dRenderer(
                img_size=self.input_size, 
                mesh_color=colors['light_purple'])


    def __render_pred_verts(self, img_original, pred_mesh_list_body, pred_mesh_list_garment, texture_rgb):
        assert max(img_original.shape) <= self.input_size, \
            f"Currently, we donlt support images size larger than:{self.input_size}"
        
        mesh_body = pred_mesh_list_body[0]
        mesh_garment = pred_mesh_list_garment[0]

        pred_mesh_list = [mesh_body, mesh_garment]

        res_img = img_original.copy()
        rend_img = np.ones((self.input_size, self.input_size, 3))
        mask = np.ones((self.input_size, self.input_size, 3))
        h, w = img_original.shape[:2]
        rend_img[:h, :w, :] = img_original
        mask[:h, :w, :] = img_original

        verts_body = mesh_body['vertices']
        faces_body = mesh_body['faces']
        verts_garment = mesh_garment['vertices']
        faces_garment = mesh_garment['faces']

        rend_img = self.renderer.render(verts_body, faces_body, verts_garment, faces_garment, rend_img, texture_rgb)
        
        #self.renderer = Pytorch3dRenderer(
        #    img_size=self.input_size, 
        #    mesh_color=colors['light_gray'],
        #    ambient=True)
        #mask = self.renderer.render(verts_garment, faces_garment, rend_img)

        #self.renderer = Pytorch3dRenderer(
        #    img_size=self.input_size, 
        #    mesh_color=colors['light_purple'],
        #    ambient=True)
        #mask = self.renderer.render(verts_body, faces_body, mask)

        res_img = rend_img[:h, :w, :]
        #res_img = mask[:h, :w, :]
        return res_img


    def visualize(self, 
        input_img, 
        hand_bbox_list = None, 
        body_bbox_list = None,
        body_pose_list = None,
        raw_hand_bboxes = None,
        pred_mesh_list_body = None,
        pred_mesh_list_garment = None,
        vis_raw_hand_bbox = True,
        vis_body_pose = True,
        vis_hand_bbox = True,
        texture_rgb = None,
    ):
        # init
        res_img = input_img.copy()

        # draw raw hand bboxes
        if raw_hand_bboxes is not None and vis_raw_hand_bbox:
            res_img = draw_raw_bbox(input_img, raw_hand_bboxes)
            # res_img = np.concatenate((res_img, raw_bbox_img), axis=1)

        # draw 2D Pose
        if body_pose_list is not None and vis_body_pose:
            res_img = draw_arm_pose(res_img, body_pose_list)

        # draw body bbox
        if body_bbox_list is not None:
            body_bbox_img = draw_body_bbox(input_img, body_bbox_list)
            res_img = body_bbox_img

        # draw hand bbox
        if hand_bbox_list is not None:
            res_img = draw_hand_bbox(res_img, hand_bbox_list)
        
        # render predicted meshes
        if (pred_mesh_list_body is not None) and (pred_mesh_list_garment is not None):
            rend_img = self.__render_pred_verts(input_img, pred_mesh_list_body, pred_mesh_list_garment, texture_rgb)
            res_img = np.concatenate((res_img, rend_img), axis=1)
            # res_img = rend_img
        
        return res_img