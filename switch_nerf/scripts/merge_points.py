from plyfile import PlyData, PlyElement
import os
import argparse
import random
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser(description='Merge points.')
parser.add_argument('--data_path', type=str, 
                    help='root data path')
parser.add_argument('--image_ids', type=str, nargs='+', default=None, 
                    help='image ids for process')
parser.add_argument("--merge_all", action='store_true', default=False, 
                    help='''merge all the data in data_path''')
parser.add_argument("--image_num", type=int, default=0, 
                    help='''image number used for merge all''')
parser.add_argument("--expert_num", type=int, default=8, 
                    help='''expert or submodel number''')
parser.add_argument("--model_type", type=str, 
                    help='''mega or switch or nerf''')
parser.add_argument("--data_type", type=str, default="coarse", 
                    help='''coarse or fine, only support coarse''')
parser.add_argument("--topk", type=int, default=0, 
                    help='''topk for expert''')
parser.add_argument("-r", "--sample_ratio", type=float, default=1.0, 
                    help='''topk for expert''')

args = parser.parse_args()

data_path = args.data_path
merge_all = args.merge_all
expert_num = args.expert_num
model_type = args.model_type
data_type = args.data_type
topk = args.topk
sample_ratio = args.sample_ratio

if model_type == "nerf":
    if merge_all:
        data_path_1 = Path(data_path)
        plys = [i.name for i in data_path_1.glob('**/*') if i.suffix == ".ply"]
        image_ids = [i.split("_")[0] for i in plys if i.split("_")[0].isdigit()]
        image_ids = list(set(image_ids))
    else:
        image_ids = args.image_ids
else:
    if merge_all:
        image_ids = [str(i) for i in range(args.image_num)]
    else:
        image_ids = args.image_ids

print("image_ids", image_ids)
if expert_num > 0:
    for expert_id in range(expert_num):
        out_ply_name = '{}_pts_rgba_exp_{}.ply'.format(data_type, expert_id)
        out_ply_path = os.path.join(data_path, out_ply_name)
        sample_datas = []
        for image_id in image_ids:
            if model_type == "mega":
                ply_name = '{:03d}_{}_pts_rgba_exp_{}.ply'.format(int(image_id), data_type, expert_id)
            elif model_type == "switch" or model_type == "nerf":
                ply_name = '{:03d}_{}_pts_rgba_top_{:01d}_exp_{}.ply'.format(int(image_id), data_type, topk, expert_id)

            ply_path = os.path.join(data_path, image_id, ply_name)
            ply_data = PlyData.read(ply_path)
            pts_data = ply_data.elements[0].data

            pts_num = ply_data.elements[0].count
            sample_num = int(pts_num * sample_ratio)
            if sample_num == 0:
                continue
            else:
                sample_ids = random.sample(range(pts_num), sample_num)
                sample_data = pts_data[sample_ids]
                sample_datas.append(sample_data)

        sample_data = np.concatenate(sample_datas)
        el = PlyElement.describe(sample_data, 'vertex')
        PlyData([el]).write(out_ply_path)
        pass
else:
    # no moe or clusters
    out_ply_name = '{}_pts_rgba.ply'.format(data_type)
    out_ply_path = os.path.join(data_path, out_ply_name)
    sample_datas = []
    for image_id in image_ids:
        ply_name = '{:03d}_{}_pts_rgba.ply'.format(int(image_id), data_type)

        ply_path = os.path.join(data_path, image_id, ply_name)
        ply_data = PlyData.read(ply_path)
        pts_data = ply_data.elements[0].data

        pts_num = ply_data.elements[0].count
        sample_num = int(pts_num * sample_ratio)
        if sample_num == 0:
            continue
        else:
            sample_ids = random.sample(range(pts_num), sample_num)
            sample_data = pts_data[sample_ids]
            sample_datas.append(sample_data)

    sample_data = np.concatenate(sample_datas)
    el = PlyElement.describe(sample_data, 'vertex')
    PlyData([el]).write(out_ply_path)
    pass



