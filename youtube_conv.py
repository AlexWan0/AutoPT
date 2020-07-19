import os
import glob
from pose_api import PoseAPI
import json
from tqdm import tqdm

pa = PoseAPI()

data_all = []

def files_to_database(out_fp, folder='raw_videos'):
	for fp in tqdm(glob.glob(os.path.join(folder, '*.mp4'))):
		data_vid = {}
		
		data_vid['file_path'] = fp
		data_vid['vid_name'] = '-'.join(os.path.split(fp)[-1].split('.mp4')[0].split('-')[:-1])
		
		#print(data_vid)

		poses, key_idx, sum_diff = pa.get_poses_from_video(fp, extracted_dir='converted')

		data_vid['pose_data'] = poses
		data_vid['pose_path'] = os.path.join('converted', os.path.split(fp)[-1].split('.')[0] + '.mp4')
		data_vid['key_idx'] = key_idx
		data_vid['sum_diff'] = sum_diff

		data_all.append(data_vid)

		with open(out_fp, 'w') as file_out:
			json.dump(data_all, file_out)

if __name__ == '__main__':
	files_to_database('db.json')