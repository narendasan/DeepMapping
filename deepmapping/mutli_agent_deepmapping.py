import os
import argparse

from matplotlib.pyplot import grid
from models import DeepMapping2D
import utils
import numpy as np
import torch
from torch.utils.data import DataLoader
from loss import bce_ch as bce_ch_loss
import json
from typing import Dict, List
import random

import open3d as o3d
import cv2
import pycpd

from dataset_loader import SimulatedPointCloud

torch.backends.cudnn.deterministic = True
torch.manual_seed(999)

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="v0", help="The enviornment being mapped")
parser.add_argument("--num-agents", type=int, default=4, help="Number of agents mapping the enviornment")
parser.add_argument("--data-dir", type=str, default="data/multi_agent", help="Root directory for the dataset")
parser.add_argument("--results-dir", type=str, default="results/multi_agent", help="Root directory for results")
parser.add_argument("--local-lr", type=float, default=0.001, help="Learning rate for local deepmapping optimization")
parser.add_argument("--local-samples", type=int, default=16, help="Number of unsampled points along rays for local pc registration")
parser.add_argument("--local-epochs", type=int, default=500, help="Number of epochs to refine local pose estimation")
parser.add_argument("--local-batch-size", type=int, default=32, help="Batch size for local training of point cloud registration")
parser.add_argument("--save-interval", type=int, default=100, help="How often to store a checkpoint")
parser.add_argument("--restart", action="store_true", help="Restart using last checkpoints")
parser.add_argument("--skip-training", action="store_true", help="Skip training local dm models")
args = parser.parse_args()

NUM_AGENTS = args.num_agents
ENV = args.env
DATA_DIR = args.data_dir
RESULTS_DIR = args.results_dir

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mkdirp(d: str) -> None:
	if not os.path.exists(d):
		os.makedirs(d)

def icp_initial_refinement(pcs: List, valid_ids: List, result_dir: str, metrics="plane", n_iter=100) -> np.array:
	n_pc = len(pcs)
	pose_est = np.zeros((n_pc, 3), dtype=np.float32)

	for idx in range(n_pc-1):
		dst = pcs[idx]
		valid_dst = valid_ids[idx]
		src = pcs[idx+1]
		valid_src = valid_ids[idx+1]

		dst = dst[valid_dst,:].numpy()
		src = src[valid_src,:].numpy()

		_, R0, t0 = utils.icp(src, dst, n_iter=n_iter, metrics=metrics)
		if idx == 0:
			R_cum = R0
			t_cum = t0
		else:
			R_cum = np.matmul(R_cum, R0)
			t_cum = np.matmul(R_cum, t0) + t_cum

		pose_est[idx+1,:2] = t_cum.T
		pose_est[idx+1,2] = np.arctan2(R_cum[1,0], R_cum[0,0])

	icp_pose_est_file = os.path.join(result_dir, "icp_pose_est.npy")
	np.save(icp_pose_est_file, pose_est)

	return pose_est

def icp_refinement(pcs, valid_ids, poses, result_dir: str, metrics="plane", n_iter=100) -> np.array:
	n_pc = len(pcs)
	pose_est = np.zeros((n_pc, 3), dtype=np.float32)

	global_pcs = utils.transform_to_global_2D(torch.from_numpy(poses), pcs)

	for idx in range(n_pc-1):
		dst = global_pcs[idx]
		valid_dst = valid_ids[idx]
		src = global_pcs[idx+1]
		valid_src = valid_ids[idx+1]

		dst = dst[valid_dst,:].numpy()
		src = src[valid_src,:].numpy()

		_, R0, t0 = utils.icp(src, dst, n_iter=n_iter, metrics=metrics)
		if idx == 0:
			R_cum = R0
			t_cum = t0
		else:
			R_cum = np.matmul(R_cum, R0)
			t_cum = np.matmul(R_cum, t0) + t_cum

		pose_est[idx+1,:2] = t_cum.T
		pose_est[idx+1,2] = np.arctan2(R_cum[1,0], R_cum[0,0])

	icp_pose_est_file = os.path.join(result_dir, "icp_global_pose_est.npy")
	np.save(icp_pose_est_file, pose_est)

	return pose_est

def dm_pose_est(icp_pose_est: np.array,
								model: torch.nn.Module,
								obvs: SimulatedPointCloud,
								loader: torch.utils.data.DataLoader,
								ckpt_dir: str) -> np.array:
		obs_global_est = []
		pose_est = []
		with torch.no_grad():
			model.eval()
			for i, (obs_batch, valid_pts, pose) in enumerate(loader):
				obs_batch = obs_batch.to(device)
				valid_pts = valid_pts.to(device)
				pose = pose.to(device)

				model(obs_batch, valid_pts, pose)

				obs_global_est.append(model.obs_global_est.cpu().detach().numpy())
				pose_est.append(model.pose_est.cpu().detach().numpy())

			pose_est = np.concatenate(pose_est)
			pose_est = utils.cat_pose_2D(icp_pose_est, pose_est)

			obs_global_est = np.concatenate(obs_global_est)
			valid_pts = obvs.valid_points.cpu().detach().numpy()
			utils.plot_global_point_cloud(obs_global_est, pose_est, valid_pts, ckpt_dir, stage="eval_local_dm")

		return pose_est

def local_pc_registration(init_pose: np.array, obvs_dir: str, ckpt_dir: str) -> np.array:
	# In practice its not really great to train in pipeline
	# It is better to train per agent using the train_2d script and then use --skip-training to
	# perform inference in pipeline
	init_pose_tensor = torch.from_numpy(init_pose)
	obvs = SimulatedPointCloud(obvs_dir, init_pose_tensor)
	loader = DataLoader(obvs, batch_size=args.local_batch_size, shuffle=False)
	loss_fn = bce_ch_loss
	model = DeepMapping2D(loss_fn=loss_fn, n_obs=obvs.n_obs, n_samples=args.local_samples).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=args.local_lr)

	start_epoch = 0
	if args.restart:
		ckpt_file = os.path.join(ckpt_dir, "model_best.pth")
		if os.path.exists(ckpt_file):
			utils.load_checkpoint(ckpt_file, model, optimizer)
			with open(os.path.join(ckpt_dir, "model_info.json"), "r") as f:
				data = json.load(f)
				start_epoch = data["epoch"]

	best_pose_est = None
	if not args.skip_training:
		for epoch in range(start_epoch, args.local_epochs):
			tot_loss = 0.0
			model.train()

			for i, (obs_batch, valid_pts, pose) in enumerate(loader):
				obs_batch = obs_batch.to(device)
				valid_pts = valid_pts.to(device)
				pose = pose.to(device)

				loss = model(obs_batch, valid_pts, pose)
				loss.backward()
				optimizer.step()

				tot_loss += loss.item()

			avg_loss = tot_loss/len(loader)

			if ( epoch + 1 ) % args.save_interval == 0:
				print("[{}/{}] avg training loss: {:.4}".format(epoch+1,args.local_epochs,avg_loss))

				obs_global_est = []
				pose_est = []
				with torch.no_grad():
					model.eval()
					for i, (obs_batch, valid_pts, pose) in enumerate(loader):
						obs_batch = obs_batch.to(device)
						valid_pts = valid_pts.to(device)
						pose = pose.to(device)

						model(obs_batch, valid_pts, pose)

						obs_global_est.append(model.obs_global_est.cpu().detach().numpy())
						pose_est.append(model.pose_est.cpu().detach().numpy())

					pose_est = np.concatenate(pose_est)
					pose_est = utils.cat_pose_2D(init_pose, pose_est)

					ckpt_file = os.path.join(ckpt_dir, "model_best.pth")
					utils.save_checkpoint(ckpt_file, model, optimizer)
					with open(os.path.join(ckpt_dir, "model_info.json"), "w") as f:
						json.dump({"epoch": epoch}, f)


					obs_global_est = np.concatenate(obs_global_est)
					valid_pts = obvs.valid_points.cpu().detach().numpy()
					utils.plot_global_point_cloud(obs_global_est, pose_est, valid_pts, ckpt_dir, epoch=epoch+1)

					dm_pose_est_file = os.path.join(ckpt_dir, "local_dm_pose_est.npy")
					np.save(dm_pose_est_file, pose_est)

	best_pose_est = dm_pose_est(init_pose, model, obvs, loader, ckpt_dir)
	return best_pose_est

def plot_global_pc(obvs: SimulatedPointCloud, pose_est: np.array, result_dir: str, stage: str) -> None:
	pose_est_tensor = torch.from_numpy(pose_est)
	local_pcs, valid_ids, _ = obvs[:]
	global_pc = utils.transform_to_global_2D(pose_est_tensor, local_pcs)
	print(global_pc.shape)
	print(pose_est_tensor.shape)
	utils.plot_global_point_cloud(global_pc, pose_est_tensor, valid_ids, result_dir, stage=stage)
	return

def naive_global_fusion(agent_trajectory_poses: Dict, pose_src: str, env_result_dir: str, stage: str):
	poses = [a[pose_src] for _, a in agent_trajectory_poses.items()]
	pcs = [a["obvs"][:][0] for _, a in agent_trajectory_poses.items()]
	valid_ids = [a["obvs"][:][1] for _, a in agent_trajectory_poses.items()]

	poses = np.concatenate(poses)
	pcs = torch.cat(pcs)
	valid_ids = torch.cat(valid_ids)

	poses_tensor = torch.from_numpy(poses)
	unified_global_pc = utils.transform_to_global_2D(poses_tensor, pcs)
	utils.plot_global_point_cloud(unified_global_pc, poses_tensor, valid_ids, env_result_dir, plot_pose=True, stage=stage)

def attempt1(agent_trajectory_poses, env_result_dir):
	global_pcs = []
	valid_ids = []
	global_poses = []
	for agent in agent_trajectory_poses.keys():
		poses = agent_trajectory_poses[agent]["dm_poses"]
		pcs,v_ids,_ = agent_trajectory_poses[agent]["obvs"][:]
		for pose, pc, v_id in zip(poses, pcs, v_ids):
			pose = np.expand_dims(pose, axis=0)
			pc = pc.unsqueeze(0)
			v_id = v_id.unsqueeze(0)

			g_pc = utils.transform_to_global_2D(torch.from_numpy(pose), pc)
			global_pcs.append(g_pc)
			valid_ids.append(v_id)
			global_poses.append(pose)

	global_poses_tensor = torch.from_numpy(np.concatenate(global_poses))
	global_pcs_tensor = torch.cat(global_pcs)
	valid_ids_tensor = torch.cat(valid_ids)

	global_pc = utils.transform_to_global_2D(global_poses_tensor, global_pcs_tensor)
	utils.plot_global_point_cloud(global_pc, global_poses_tensor, valid_ids_tensor, env_result_dir, stage="global_pose_transform")

	global_icp(global_pcs, valid_ids, env_result_dir)

def global_icp(pcs, valid_ids, result_dir):
	global_pose_est = icp_initial_refinement(pcs, valid_ids, result_dir)
	global_pose_est_tensor = torch.from_numpy(global_pose_est)
	global_pcs = torch.cat(pcs)
	valid_ids = torch.cat(valid_ids)

	print("Render fusion using ICP to refine alignment")
	global_pc = utils.transform_to_global_2D(global_pose_est_tensor, global_pcs)
	utils.plot_global_point_cloud(global_pc, global_pose_est_tensor, valid_ids, result_dir, stage="global_icp_refinement")
	return

def pc_downsampling(pc, valid_ids, grid_size=0.005, bounds=(-3.0, 3.0)):
	"""
	Inspired by voxel downsampling, down sample the point cloud by imposing a grid over the space
	and pooling points
	"""

	print(pc.shape)

	num_buckets_per_dim = int((bounds[1] - bounds[0]) // grid_size)

	p_to_i = lambda p: int((p.item() - bounds[0]) // grid_size)
	pt_to_idx = lambda pt: (p_to_i(pt[0]) if (p_to_i(pt[0]) > 0 and p_to_i(pt[0]) < num_buckets_per_dim) else None,
													p_to_i(pt[1]) if (p_to_i(pt[1]) > 0 and p_to_i(pt[1]) < num_buckets_per_dim) else None)

	i_to_p = lambda i: ((i * grid_size) + bounds[0]) + (grid_size / 2) # center in bucket
	idx_to_pt = lambda idx: (i_to_p(idx[0]), i_to_p(idx[1]))

	grid = np.zeros((num_buckets_per_dim, num_buckets_per_dim), dtype=bool)
	uint_img = np.array(grid.astype(int)*255).astype('uint8')

	for i, p in enumerate(pc):
		if valid_ids[i]:
			if all(pt_to_idx(p)):
				grid[pt_to_idx(p)] = True

	new_pc_points = []
	test = []
	for i in np.argwhere(grid):
		test.append((*idx_to_pt(i), 0))
		new_pc_points.append(idx_to_pt(i))

	downsampled_pc = np.array(new_pc_points)
	ids = np.full(downsampled_pc.shape[0], True)

	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(test)
	#o3d.visualization.draw_geometries([pcd])

	return downsampled_pc, ids

def pad_pc(pc, ids, target_len):
	diff = target_len - ids.shape[1]
	if diff > 0:
		pc_pad = torch.zeros((diff, 2))
		ids_pad = torch.zeros((diff), dtype=bool)
		padded_pc = torch.cat([pc[0], pc_pad]).unsqueeze(0)
		padded_ids = torch.cat([ids[0], ids_pad]).unsqueeze(0)
		return padded_pc, padded_ids
	else:
		return pc, ids

def iterative_cpd_affine_alignment(src_pcs, valid_ids, pad_size, result_dir):

	aligned_pcs = [src_pcs[0]]
	aligned_ids = [valid_ids[0]]
	aligned_poses = [np.expand_dims(np.array((0,0,0)), axis=0)]

	n_pcs = len(src_pcs)
	for i in range(n_pcs - 1):
		reg = pycpd.AffineRegistration(X=src_pcs[i].numpy()[valid_ids[i].numpy()],
																	 Y=src_pcs[i+1].numpy()[valid_ids[i+1].numpy()],
																	 max_iterations=1000,
																	 tolarance=0.0001)
		new_pc,_ = reg.register()
		new_ids = np.full(new_pc.shape[0], True)

		new_pc_tensor = torch.from_numpy(new_pc).unsqueeze(0)
		new_ids_tensor = torch.from_numpy(new_ids).unsqueeze(0)

		new_pc_tensor, new_ids_tensor = pad_pc(new_pc_tensor, new_ids_tensor, pad_size)

		aligned_pcs.append(new_pc_tensor)
		aligned_ids.append(new_ids_tensor)
		aligned_poses.append(np.expand_dims(np.array((0,0,0)), axis=0))

	pcs_tensor = torch.cat(aligned_pcs)
	valid_ids_tensor = torch.cat(aligned_ids)
	poses = np.concatenate(aligned_poses, axis=0)

	utils.plot_global_point_cloud(pcs_tensor, torch.from_numpy(poses), valid_ids_tensor, result_dir, plot_pose=False, stage="global_iterative_affine_cpd")

def root_view_cpd_affine_alignment(src_pcs, valid_ids, pad_size, result_dir):

	n_pcs = len(src_pcs)
	order = list(range(n_pcs))
	#random.shuffle(order)

	src_pcs = [src_pcs[i] for i in order]
	valid_ids = [valid_ids[i] for i in order]

	aligned_pcs = [src_pcs[0]]
	aligned_ids = [valid_ids[0]]
	aligned_poses = [np.expand_dims(np.array((0,0,0)), axis=0)]

	transforms = []

	for i in range(1, n_pcs):
		reg = pycpd.AffineRegistration(X=src_pcs[0].numpy()[valid_ids[0].numpy()],
																	 Y=src_pcs[i].numpy()[valid_ids[i].numpy()],
																	 max_iterations=1000,
																	 tolarance=0.0001)
		new_pc, transform = reg.register()
		new_ids = np.full(new_pc.shape[0], True)

		new_pc_tensor = torch.from_numpy(new_pc).unsqueeze(0)
		new_ids_tensor = torch.from_numpy(new_ids).unsqueeze(0)

		new_pc_tensor, new_ids_tensor = pad_pc(new_pc_tensor, new_ids_tensor, pad_size)

		aligned_pcs.append(new_pc_tensor)
		aligned_ids.append(new_ids_tensor)
		aligned_poses.append(np.expand_dims(np.array((0,0,0)), axis=0))
		transforms.append(transform)

	pcs_tensor = torch.cat(aligned_pcs)
	valid_ids_tensor = torch.cat(aligned_ids)
	poses = np.concatenate(aligned_poses, axis=0)

	utils.plot_global_point_cloud(pcs_tensor, torch.from_numpy(poses), valid_ids_tensor, result_dir, plot_pose=False, stage="global_rooted_affine_cpd")

def iterative_cpd_rigid_alignment(src_pcs, valid_ids, pad_size, result_dir):

	aligned_pcs = [src_pcs[0]]
	aligned_ids = [valid_ids[0]]
	aligned_poses = [np.expand_dims(np.array((0,0,0)), axis=0)]

	n_pcs = len(src_pcs)
	for i in range(n_pcs - 1):
		reg = pycpd.RigidRegistration(X=src_pcs[i].numpy()[valid_ids[i].numpy()],
																	 Y=src_pcs[i+1].numpy()[valid_ids[i+1].numpy()],
																	 max_iterations=1000,
																	 tolarance=0.0001)
		new_pc,_ = reg.register()
		new_ids = np.full(new_pc.shape[0], True)

		new_pc_tensor = torch.from_numpy(new_pc).unsqueeze(0)
		new_ids_tensor = torch.from_numpy(new_ids).unsqueeze(0)

		new_pc_tensor, new_ids_tensor = pad_pc(new_pc_tensor, new_ids_tensor, pad_size)

		aligned_pcs.append(new_pc_tensor)
		aligned_ids.append(new_ids_tensor)
		aligned_poses.append(np.expand_dims(np.array((0,0,0)), axis=0))

	pcs_tensor = torch.cat(aligned_pcs)
	valid_ids_tensor = torch.cat(aligned_ids)
	poses = np.concatenate(aligned_poses, axis=0)

	utils.plot_global_point_cloud(pcs_tensor, torch.from_numpy(poses), valid_ids_tensor, result_dir, plot_pose=False, stage="global_iterative_rigid_cpd")

def root_view_cpd_rigid_alignment(src_pcs, valid_ids, pad_size, result_dir):

	n_pcs = len(src_pcs)
	order = list(range(n_pcs))
	#random.shuffle(order)

	src_pcs = [src_pcs[i] for i in order]
	valid_ids = [valid_ids[i] for i in order]

	aligned_pcs = [src_pcs[0]]
	aligned_ids = [valid_ids[0]]
	aligned_poses = [np.expand_dims(np.array((0,0,0)), axis=0)]

	transforms = []

	inital_prior = [0,-1.5,0,1.5,-1.50]
	Rs = [utils.ang2mat(t) for t in inital_prior]

	for i in range(1, n_pcs):
		reg = pycpd.RigidRegistration(X=src_pcs[0].numpy()[valid_ids[0].numpy()],
																	 Y=src_pcs[i].numpy()[valid_ids[i].numpy()],
																	 max_iterations=10000,
																	 tolarance=0.00001,
																	 R=Rs[i])
		new_pc, transform = reg.register()
		new_ids = np.full(new_pc.shape[0], True)

		new_pc_tensor = torch.from_numpy(new_pc).unsqueeze(0)
		new_ids_tensor = torch.from_numpy(new_ids).unsqueeze(0)

		new_pc_tensor, new_ids_tensor = pad_pc(new_pc_tensor, new_ids_tensor, pad_size)

		aligned_pcs.append(new_pc_tensor)
		aligned_ids.append(new_ids_tensor)
		aligned_poses.append(np.expand_dims(np.array((0,0,0)), axis=0))
		transforms.append(transform)

	pcs_tensor = torch.cat(aligned_pcs)
	valid_ids_tensor = torch.cat(aligned_ids)
	poses = np.concatenate(aligned_poses, axis=0)

	utils.plot_global_point_cloud(pcs_tensor, torch.from_numpy(poses), valid_ids_tensor, result_dir, plot_pose=False, stage="global_rooted_rigid_cpd")
	return transforms

def main() -> None:

	env_result_dir = "{}/{}".format(RESULTS_DIR, ENV)

	mkdirp(env_result_dir)
	utils.save_opt(env_result_dir, args)

	agent_trajectory_data = lambda env, agent: "{}/{}/{}_pose{}".format(DATA_DIR, env, env, agent)
	agent_trajectory_results = lambda env, agent: "{}/{}/agent{}".format(RESULTS_DIR, env, agent)

	agent_trajectory_poses = {}

	for a in range(NUM_AGENTS):
		# Conduct local mapping
		agent_obvs_dir = agent_trajectory_data(ENV, a)
		agent_result_dir = agent_trajectory_results(ENV, a)

		print("Agent {} observation dir: {}".format(a, agent_obvs_dir))
		print("Agent {} results directory: {}".format(a, agent_result_dir))

		mkdirp(agent_result_dir)

		# 1. ICP local refinement
		agent_obvs = SimulatedPointCloud(agent_obvs_dir)

		print("Running initial ICP refinement for Agent {} on map {}".format(a, ENV))
		raw_pcs, valid_ids, _ = agent_obvs[:]
		icp_pose_est = icp_initial_refinement(raw_pcs, valid_ids, agent_result_dir)
		plot_global_pc(agent_obvs, icp_pose_est, agent_result_dir, "icp")

		# 2.Refinement using DeepMapping
		print("Running local point cloud registration for Agent {} on map {}".format(a, ENV))
		dm_pose_est = local_pc_registration(icp_pose_est, agent_obvs_dir, agent_result_dir)
		plot_global_pc(agent_obvs, dm_pose_est, agent_result_dir, "local_dm")

		agent_trajectory_poses[a] = {
			"icp_poses": icp_pose_est,
			"dm_poses" : dm_pose_est,
			"obvs" : SimulatedPointCloud(agent_obvs_dir)
		}

	# Fuse maps into a single result
	naive_global_fusion(agent_trajectory_poses, "icp_poses", env_result_dir, "non_oriented_naive_global_fusion_icp")
	naive_global_fusion(agent_trajectory_poses, "dm_poses", env_result_dir, "non_oriented_naive_global_fusion_dm")

	# Simplify into N maps and then run ICP
	max_size = 0
	for a in range(NUM_AGENTS):
		poses = agent_trajectory_poses[a]["dm_poses"]
		pcs = agent_trajectory_poses[a]["obvs"][:][0]
		valid_ids = agent_trajectory_poses[a]["obvs"][:][1]

		poses_tensor = torch.from_numpy(poses)
		unified_global_pc = utils.transform_to_global_2D(poses_tensor, pcs)
		unified_global_pc = unified_global_pc.reshape((-1, 2))
		valid_ids = valid_ids.reshape((-1))

		print(unified_global_pc.shape, valid_ids.shape, type(unified_global_pc), type(valid_ids))

		agent_trajectory_poses[a]["global_pc"] = unified_global_pc.unsqueeze(0)
		agent_trajectory_poses[a]["valid_ids"] = valid_ids.unsqueeze(0)

		downsampled_pc, downsampled_ids = pc_downsampling(unified_global_pc, valid_ids, grid_size=0.05)
		if downsampled_ids.shape[0] > max_size:
			max_size = downsampled_pc.shape[0]
		agent_trajectory_poses[a]["downsampled_pc"] = torch.from_numpy(downsampled_pc).unsqueeze(0)
		agent_trajectory_poses[a]["downsampled_ids"] = torch.from_numpy(downsampled_ids).unsqueeze(0)

		print(downsampled_pc.shape, downsampled_ids.shape)


	print(max_size)
	for a in range(NUM_AGENTS):
		pc = agent_trajectory_poses[a]["downsampled_pc"]
		ids = agent_trajectory_poses[a]["downsampled_ids"]

		padded_pc, padded_ids = pad_pc(pc, ids, max_size)

		agent_trajectory_poses[a]["padded_downsampled_pc"] = padded_pc
		agent_trajectory_poses[a]["padded_downsampled_ids"] = padded_ids

	pcs = [a["padded_downsampled_pc"] for a in agent_trajectory_poses.values()]
	valid_ids = [a["padded_downsampled_ids"] for a in agent_trajectory_poses.values()]

	poses = icp_initial_refinement(pcs, valid_ids, env_result_dir, metrics="point", n_iter=100000)
	print(poses)

	pcs_tensor = torch.cat(pcs)
	valid_ids_tensor = torch.cat(valid_ids)
	print(poses.shape, pcs_tensor.shape, valid_ids_tensor.shape)

	utils.plot_global_point_cloud(pcs_tensor, torch.from_numpy(poses), valid_ids_tensor, env_result_dir, plot_pose=False, stage="global_icp_alignment")

	#iterative_cpd_affine_alignment(pcs, valid_ids, max_size, env_result_dir)
	#root_view_cpd_affine_alignment(pcs, valid_ids, max_size, env_result_dir)
	#iterative_cpd_rigid_alignment(pcs, valid_ids, max_size, env_result_dir)
	transforms = root_view_cpd_rigid_alignment(pcs, valid_ids, max_size, env_result_dir)
	transforms.insert(0, (np.array(1), np.zeros((2,2)), np.zeros(2)))
	for a in range(NUM_AGENTS):
		agent_trajectory_poses[a]["cpd_transform"] = transforms[a]
		s, R, t = transforms[a]
		theta = np.arctan2(R[1,0],R[0,0]).astype(float)
		print(theta)
		cpd_pose = np.array((s * t[0], s*t[1], -theta)).astype(np.float32) #np.array((t[0], t[1], theta)).astype(float) #* s
		dm_poses = agent_trajectory_poses[a]["dm_poses"]
		aligned_poses = np.concatenate([utils.cat_pose_2D(np.expand_dims(p, axis=0), np.expand_dims(cpd_pose, axis=0)) for p in dm_poses]).astype(np.float32)


		agent_trajectory_poses[a]["cpd_poses"] = aligned_poses #dm_poses + cpd_pose

	naive_global_fusion(agent_trajectory_poses, "cpd_poses", env_result_dir, "non_oriented_naive_global_fusion_cpd")

	pcs = []
	valid_ids = []
	icp_poses = []
	dm_poses = []
	cpd_poses = []
	for a in range(NUM_AGENTS):
		p = agent_trajectory_poses[a]["obvs"][:][0]
		v = agent_trajectory_poses[a]["obvs"][:][1]
		i = agent_trajectory_poses[a]["icp_poses"]
		d = agent_trajectory_poses[a]["dm_poses"]
		c = agent_trajectory_poses[a]["cpd_poses"]
		pcs.append(p)
		valid_ids.append(v)
		icp_poses.append(i)
		dm_poses.append(d)
		cpd_poses.append(c)

	pcs_tensor = torch.cat(pcs)
	valid_ids_tensor = torch.cat(valid_ids)
	icp_poses_np = np.concatenate(icp_poses, axis=0)
	dm_poses_np = np.concatenate(dm_poses, axis=0)
	cpd_poses_np = np.concatenate(cpd_poses, axis=0)

	np.save("{}/{}".format(env_result_dir, "cpd_poses.npy"), cpd_poses_np)

	icp_dm_poses = icp_refinement(pcs_tensor, valid_ids_tensor, dm_poses_np, env_result_dir)
	icp_dm_poses = utils.cat_pose_2D(dm_poses_np, icp_dm_poses)
	icp_cpd_poses = icp_refinement(pcs_tensor, valid_ids_tensor, cpd_poses_np, env_result_dir)
	icp_cpd_poses = utils.cat_pose_2D(cpd_poses_np, icp_cpd_poses)

	utils.plot_global_point_cloud(pcs_tensor, torch.from_numpy(icp_dm_poses), valid_ids_tensor, env_result_dir, plot_pose=True, stage="global_icp_after_dm_alignment")
	utils.plot_global_point_cloud(pcs_tensor, torch.from_numpy(icp_cpd_poses), valid_ids_tensor, env_result_dir, plot_pose=True, stage="global_icp_after_cpm_alignment")

	return

if __name__ == "__main__":
	main()