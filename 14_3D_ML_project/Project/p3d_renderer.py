# Copyright (c) Facebook, Inc. and its affiliates.

# Part of code is modified from https://github.com/facebookresearch/pytorch3d

import cv2
import os
import sys
import torch
import numpy as np

from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import Textures
from pytorch3d.renderer import (
    PerspectiveCameras,
    FoVOrthographicCameras,
    PointLights,
    RasterizationSettings, 
    MeshRenderer, 
    BlendParams,
    MeshRasterizer,  
    SoftPhongShader,
)
from pytorch3d.renderer.utils import TensorProperties
from typing import Union
Device = Union[str, torch.device]

class AmbientLights(TensorProperties):
    """
    A light object representing the same color of light everywhere.
    By default, this is white, which effectively means lighting is
    not used in rendering.
    """

    def __init__(self, *, ambient_color=None, device: Device = "cpu") -> None:
        """
        If ambient_color is provided, it should be a sequence of
        triples of floats.
        Args:
            ambient_color: RGB color
            device: Device (as str or torch.device) on which the tensors should be located
        The ambient_color if provided, should be
            - 3 element tuple/list or list of lists
            - torch tensor of shape (1, 3)
            - torch tensor of shape (N, 3)
        """
        if ambient_color is None:
            ambient_color = ((1.0, 1.0, 1.0),)
        super().__init__(ambient_color=ambient_color, device=device)

    def clone(self):
        other = self.__class__(device=self.device)
        return super().clone(other)

    def diffuse(self, normals, points) -> torch.Tensor:
        return torch.zeros_like(points)

    def specular(self, normals, points, camera_position, shininess) -> torch.Tensor:
        return torch.zeros_like(points)
        

class Pytorch3dRenderer(object):

    def __init__(self, img_size, mesh_color, ambient=False):
        self.device = torch.device("cuda:0")
        # self.render_size = 1920

        self.img_size = img_size

        # mesh color
        mesh_color = np.array(mesh_color)[::-1]
        self.mesh_color = torch.from_numpy(
            mesh_color.copy()).view(1, 1, 3).float().to(self.device)

        # renderer for large objects, such as whole body.
        self.render_size_large = 700
        if ambient == False:
            lights = PointLights(
                ambient_color = [[1.0, 1.0, 1.0],],
                diffuse_color = [[1.0, 1.0, 1.0],],
                device=self.device, location=[[1.0, 1.0, -30]])
        else:
            lights = AmbientLights(
                ambient_color = [[1.0, 1.0, 1.0],],
                device=self.device
            )
        self.renderer_large = self.__get_renderer(self.render_size_large, lights)

        # renderer for small objects, such as whole body.
        self.render_size_medium = 400
        if ambient == False:
            lights = PointLights(
                ambient_color = [[0.5, 0.5, 0.5],],
                diffuse_color = [[0.5, 0.5, 0.5],],
                device=self.device, location=[[1.0, 1.0, -30]])
        else:
            lights = AmbientLights(
                ambient_color = [[0.5, 0.5, 0.5],],
                device=self.device
            )
        self.renderer_medium = self.__get_renderer(self.render_size_medium, lights)

        # renderer for small objects, such as whole body.
        self.render_size_small = 200
        if ambient == False:
            lights = PointLights(
                ambient_color = [[0.5, 0.5, 0.5],],
                diffuse_color = [[0.5, 0.5, 0.5],],
                device=self.device, location=[[1.0, 1.0, -30]])
        else:
            lights = AmbientLights(
                ambient_color = [[0.5, 0.5, 0.5], ],
                device=self.device
            )
        self.renderer_small = self.__get_renderer(self.render_size_small, lights)

    def __get_renderer(self, render_size, lights):

        cameras = FoVOrthographicCameras(
            device = self.device,
            znear=0.1,
            zfar=10.0,
            max_y=1.0,
            min_y=-1.0,
            max_x=1.0,
            min_x=-1.0,
            scale_xyz=((1.0, 1.0, 1.0),),  # (1, 3)
        )

        raster_settings = RasterizationSettings(
            image_size = render_size,
            blur_radius = 0,
            faces_per_pixel = 1,
        )
        blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color = (0,0,0))

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device, 
                cameras=cameras,
                lights=lights,
                blend_params=blend_params
            )
        )

        return renderer

    def render(self, verts_body, faces_body, verts_garment, faces_garment, bg_img, texture_rgb=None):
        verts_body = verts_body.copy()
        faces_body = faces_body.copy()
        verts_garment = verts_garment.copy()
        faces_garment = faces_garment.copy()

        # bbox for verts
        x0 = int(np.min(verts_body[:, 0]))
        x1 = int(np.max(verts_body[:, 0]))
        y0 = int(np.min(verts_body[:, 1]))
        y1 = int(np.max(verts_body[:, 1]))
        width = x1 - x0
        height = y1 - y0

        bbox_size = max(height, width)
        if bbox_size <= self.render_size_small:
            print("Using small size renderer")
            render_size = self.render_size_small
            renderer = self.renderer_small
        else:
            if bbox_size <= self.render_size_medium:
                print("Using medium size renderer")
                render_size = self.render_size_medium
                renderer = self.renderer_medium
            else:
                print("Using large size renderer")
                render_size = self.render_size_large
                renderer = self.renderer_large
        
        # padding the tight bbox
        margin = int(max(width, height) * 0.1)
        x0 = max(0, x0-margin)
        y0 = max(0, y0-margin)
        x1 = min(self.img_size, x1+margin)
        y1 = min(self.img_size, y1+margin)

        # move verts to be in the bbox
        verts_body[:, 0] -= x0
        verts_body[:, 1] -= y0
        
        verts_garment[:, 0] -= x0
        verts_garment[:, 1] -= y0

        # normalize verts to (-1, 1)
        bbox_size = max(y1-y0, x1-x0)
        half_size = bbox_size / 2
        verts_body[:, 0] = (verts_body[:, 0] - half_size) / half_size
        verts_body[:, 1] = (verts_body[:, 1] - half_size) / half_size
        
        verts_garment[:, 0] = (verts_garment[:, 0] - half_size) / half_size
        verts_garment[:, 1] = (verts_garment[:, 1] - half_size) / half_size

        # the coords of pytorch-3d is (1, 1) for upper-left and (-1, -1) for lower-right
        # so need to multiple minus for vertices
        verts_body[:, :2] *= -1
        verts_garment[:, :2] *= -1

        # shift verts along the z-axis
        verts_body[:, 2] /= 112
        verts_body[:, 2] += 5
        
        verts_garment[:, 2] /= 112
        verts_garment[:, 2] += 5

        verts_tensor_b = torch.from_numpy(verts_body).float().unsqueeze(0).cuda()
        faces_tensor_b = torch.from_numpy(faces_body.copy()).long().unsqueeze(0).cuda()
        
        verts_tensor_g = torch.from_numpy(verts_garment).float().unsqueeze(0).cuda()
        faces_tensor_g = torch.from_numpy(faces_garment.copy()).long().unsqueeze(0).cuda()

        # set color
        mesh_color = self.mesh_color.repeat(1, verts_body.shape[0], 1)
        textures_b = Textures(verts_rgb=mesh_color)
        textures_g = Textures(verts_rgb = torch.from_numpy(texture_rgb).float().to(self.device))

        # rendering
        mesh_body = Meshes(verts=verts_tensor_b, faces=faces_tensor_b, textures=textures_b)
        mesh_garment = Meshes(verts=verts_tensor_g, faces=faces_tensor_g, textures=textures_g)

        # create z-mask
        fragments_body = renderer.rasterizer(mesh_body)
        fragments_garment = renderer.rasterizer(mesh_garment)

        z_body = fragments_body.zbuf.squeeze(3).squeeze(0)
        z_garment = fragments_garment.zbuf.squeeze(3).squeeze(0)

        eps = 0.1
        mask = (z_garment < z_body+eps).cpu().numpy()
        mask = np.repeat(mask[:, :, np.newaxis], 4, axis=2)
        # print('mask:', mask.shape)
        
        # blending rendered mesh with background image
        rend_img = renderer(mesh_garment)
        # print('rend_img:', rend_img.shape)
        rend_img = rend_img[0].cpu().numpy()

        rend_img = rend_img * mask

        scale_ratio = render_size / bbox_size
        img_size_new = int(self.img_size * scale_ratio)
        # print('bg_img:', bg_img.shape)
        bg_img_new = cv2.resize(bg_img, (img_size_new, img_size_new))
        # print('bg_img_new:', bg_img_new.shape)

        x0 = max(int(x0 * scale_ratio), 0)
        y0 = max(int(y0 * scale_ratio), 0)
        x1 = min(int(x1 * scale_ratio), img_size_new)
        y1 = min(int(y1 * scale_ratio), img_size_new)

        h0 = min(y1-y0, render_size)
        w0 = min(x1-x0, render_size)

        y1 = y0 + h0
        x1 = x0 + w0

        rend_img_new = np.zeros((img_size_new, img_size_new, 4))
        rend_img_new[y0:y1, x0:x1, :] = rend_img[:h0, :w0, :]
        rend_img = rend_img_new

        alpha = rend_img[:, :, 3:4]
        alpha[alpha>0] = 1.0


        rend_img = rend_img[:, :, :3]
        maxColor = rend_img.max()
        rend_img *= 255 /maxColor #Make sure <1.0
        rend_img = rend_img[:, :, ::-1]

        res_img = alpha * rend_img + (1.0 - alpha) * bg_img_new
        # print('res_img:', res_img.shape)
        res_img = cv2.resize(res_img, (self.img_size, self.img_size))

        return res_img
