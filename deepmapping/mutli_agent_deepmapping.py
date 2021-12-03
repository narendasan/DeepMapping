import os
import argparse
from models import DeepMapping2D
import utils
import numpy as np
import torch
from torch.utils.data import DataLoader
from loss import bce_ch as bce_ch_loss
import json

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
args = parser.parse_args()

NUM_AGENTS = args.num_agents
ENV = args.env
DATA_DIR = args.data_dir
RESULTS_DIR = args.results_dir

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mkdirp(d: str) -> None:
	if not os.path.exists(d):
		os.makedirs(d)

def icp_intial_refinement(obvs: np.array, result_dir: str) -> np.array:
	n_pc = len(obvs)
	pose_est = np.zeros((n_pc, 3), dtype=np.float32)

	for idx in range(n_pc-1):
		dst,valid_dst,_ = obvs[idx]
		src,valid_src,_ = obvs[idx+1]

		dst = dst[valid_dst,:].numpy()
		src = src[valid_src,:].numpy()

		_, R0, t0 = utils.icp(src, dst, metrics="plane")
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

def local_pc_registration(init_pose: np.array, obvs_dir: str, ckpt_dir: str) -> np.array:
	init_pose = torch.from_numpy(init_pose)
	obvs = SimulatedPointCloud(obvs_dir, init_pose)
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
				best_pose_est = pose_est

	return best_pose_est

def plot_global_pc(obvs: SimulatedPointCloud, pose_est: np.array, result_dir: str, stage: str,) -> None:
	pose_est_tensor = torch.from_numpy(pose_est)
	local_pcs, valid_ids, _ = obvs[:]
	global_icp_pc = utils.transform_to_global_2D(pose_est_tensor, local_pcs)
	utils.plot_global_point_cloud(global_icp_pc, pose_est_tensor, valid_ids, result_dir, stage=stage)
	return

def main() -> None:
	mkdirp("{}/{}".format(RESULTS_DIR, ENV))
	utils.save_opt("{}/{}".format(RESULTS_DIR, ENV), args)

	agent_trajectory_data = lambda env, agent: "{}/{}/{}_pose{}".format(DATA_DIR, env, env, agent)
	agent_trajectory_results = lambda env, agent: "{}/{}/agent{}".format(RESULTS_DIR, env, agent)

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
		icp_pose_est = icp_intial_refinement(agent_obvs, agent_result_dir)
		plot_global_pc(agent_obvs, icp_pose_est, agent_result_dir, "icp")

		# 2.Refinement using DeepMapping
		print("Running local point cloud registration for Agent {} on map {}".format(a, ENV))
		dm_pose_est = local_pc_registration(icp_pose_est, agent_obvs_dir, agent_result_dir)
		plot_global_pc(agent_obvs, dm_pose_est, agent_result_dir, "local_dm")

	# Fuse maps into a single result

if __name__ == "__main__":
	main()