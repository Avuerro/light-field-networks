import torch
import util 
import data_util

import numpy as np
import glob
import os
def pose_predictor_information(query, data_dir):
    instance_name = query['instance_name'][0][0]
    instance_location = os.path.join(data_dir,instance_name) 

    def _get_poses():
        print(f'location {instance_location}')

        all_poses = glob.glob(f'{instance_location}/pose/*.txt')
        if len(all_poses) == 0:
            return []
        specific_poses = np.random.choice(all_poses, 10)
        return specific_poses
    
    def _get_poses_gts(poses, instance_name):
        imgs_ids = [x.split('/')[-1].split('.')[0]+'.png' for x in poses]
        imgs_ids = [os.path.join(f'{instance_location}/rgb/', x) for x in imgs_ids]
        return imgs_ids
    
    poses = _get_poses()
    pose_img_ids = _get_poses_gts(poses,instance_name)
    
    
    return poses, pose_img_ids

def pose_predictor(model,input_dict,poses,pose_img_ids,output_dir):
    instance_name = input_dict['query']['instance_name'][0][0]
    location = os.path.join(output_dir,instance_name)
    if not os.path.isdir(location):
        print(location)
        os.makedirs(location)
#         os.mkdir(os.path.join(location, ''))
    input_image = input_dict['query']['rgb'].detach().cpu()
    util.save_img(util.convert_image(input_image,'rgb'), location, 'input_image.png' )
    util.write_text(poses, location,'poses')
    util.write_text(pose_img_ids, location,'images')
    for i,pose in enumerate(poses):
        loaded_pose = data_util.load_pose(pose)
#         pdb.set_trace()
        input_dict['query']['cam2world'] = torch.tensor(loaded_pose).reshape(1,1,4,4)
        
        model_input = util.dict_to_gpu(input_dict)
        model_output = model(model_input)
        
        
        out_dict = {}
        out_dict['rgb'] = model_output['rgb']
        outp = util.convert_image(out_dict['rgb'].detach().cpu(),'rgb')
        actual_gt = data_util.load_rgb(pose_img_ids[i])
        util.save_img(outp/255., location, f'prediction_pose_{i}.png')
        util.save_img(actual_gt, location, f'actual_pose_gt_{i}.png')
    