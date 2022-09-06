import torch
from torch.nn import functional as F
from pyntcloud import PyntCloud
from pyntcloud.plot import common as pyntcloud_common
import matplotlib.pyplot as plt
import pandas as pd
import util
import data_util
import geometry

import numpy as np
import os



def get_query_cam(input):
    query_dict = input['query']
    # pdb.set_trace()
    pose = util.flatten_first_two(query_dict["cam2world"])
    intrinsics = util.flatten_first_two(query_dict["intrinsics"])
    uv = util.flatten_first_two(query_dict["uv"].float())
    return pose, intrinsics, uv


def create_depth(dataloader, model):
    depths = []
    points = []
    colors = []
    pose_ids = []
    rgb_ids = []
    images = []
    for inp,_ in dataloader:
        model_input = util.dict_to_gpu(inp)

        model_output = model(model_input)
        lightfield_function = model_output['lf_function']

        ## for each image we need to know the corresponding pose... such that we can compare with img..
        
        
        b, n_ctxt = model_input['query']["uv"].shape[:2]
        n_qry, n_pix = model_input['query']["uv"].shape[1:3]
        query_pose, query_intrinsics, query_uv = get_query_cam(model_input)
        light_field_coords = geometry.plucker_embedding(query_pose, query_uv, query_intrinsics)
        ray_origin = query_pose[:, :3, 3][:, None, :]
        ray_dir = geometry.get_ray_directions(query_uv, query_pose, query_intrinsics)
        intsec_1, intsec_2 = geometry.ray_sphere_intersect(ray_origin, ray_dir, radius=100)
        intsec_1 = F.normalize(intsec_1, dim=-1)
        intsec_2 = F.normalize(intsec_2, dim=-1)
        light_field_coords = torch.cat((intsec_1, intsec_2), dim=-1)
        
        dpm = util.light_field_depth_map(light_field_coords, query_pose, model_output['lf_function'] )
        depth = dpm['depth']
        depth = depth.view(b, n_qry, n_pix, 1)
        depths.append(depth)
        points.append(dpm['points'])
        colors.append(model_input['query']['rgb'] *255)
        
        pose_id = model_input['query']['pose_key'][0][0]
        rgb_id = model_input['query']['rgb_key'][0][0]
        pose_ids.append(pose_id)
        rgb_ids.append(rgb_id)
        outp = util.convert_image( model_output['rgb'].detach().cpu(),'rgb')
        images.append(outp)

    points = torch.cat(points, dim=1).squeeze()
    colors = torch.cat(colors, dim=2).squeeze()
    images = np.asarray(images)

    return depths,points,colors, pose_ids, rgb_ids,images


def plot_with_matplotlib(cloud, output_dir, filename, **kwargs):
    plt.ioff()

    colors = pyntcloud_common.get_colors(cloud, kwargs["use_as_color"], kwargs["cmap"])

    ptp = cloud.xyz.ptp()

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    ax.view_init(elev=kwargs["elev"], azim=kwargs["azim"])

    ax.scatter(
        cloud.xyz[:, 0],
        cloud.xyz[:, 1],
        cloud.xyz[:, 2],
        marker="D",
        facecolors=colors / 255,
        zdir="z",
        depthshade=True,
        s=kwargs["initial_point_size"] or ptp / 10)
    ax.set_axis_off()
    util.set_proper_aspect_ratio(ax)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{output_dir}{filename}.png', bbox_inches='tight')
    
    cloud.to_file(f'{output_dir}{filename}.ply')
    plt.close(fig)

def construct_point_cloud(points, colors, elev,azim, output_dir, filename):
    
    point_cloud_data = torch.cat((points,colors),dim=1).detach().cpu().numpy()
    point_cloud_df = pd.DataFrame(point_cloud_data, columns=['x','y','z','red','green','blue'])
    cloud = PyntCloud(point_cloud_df)
    plot_with_matplotlib(cloud, 
                        use_as_color=['red','green','blue'], 
                        cmap='RGB', elev=elev, 
                        azim=azim,
                        output_dir=output_dir,
                        filename=filename,
                        initial_point_size=None)

def plot_depth_maps(depths, output_dir,filename):
    
    for i,depth in enumerate(depths):
        depth_image = util.convert_image(depth,'depth').squeeze()
        vmin = depth_image.min()
        vmax = depth_image.max()
        fig, ax = plt.subplots()
        fig.set_facecolor("white")
        
        ax.set_axis_off()
        plt.imsave(fname=f'{output_dir}{filename}_{i}.png', arr=depth_image, format='png')

