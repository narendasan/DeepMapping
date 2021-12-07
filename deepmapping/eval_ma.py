import set_path
import os
import argparse
import functools
print = functools.partial(print,flush=True)

import numpy as np
import scipy.io as sio

import utils

AGENT0_GT = "/home/naren/Developer/py/DeepMapping/data/multi_agent/v1/v1_pose0/gt_pose.mat"
AGENT1_GT = "/home/naren/Developer/py/DeepMapping/data/multi_agent/v1/v1_pose1/gt_pose.mat"

MA_RESULTS = "/home/naren/Developer/py/DeepMapping/results/multi_agent/v1/cpd_poses.npy"

gt_pose0 = sio.loadmat(AGENT0_GT)
gt_pose0 = gt_pose0['pose']
gt_location0 = gt_pose0[:,:2]

print(gt_pose0.shape, gt_location0.shape)

gt_pose1 = sio.loadmat(AGENT1_GT)
gt_pose1 = gt_pose1['pose']
gt_location1 = gt_pose1[:,:2]

print(gt_pose1.shape, gt_location1.shape)

pred_pose = np.load(MA_RESULTS)
pred_location = pred_pose[:,:2] * 512 # denormalization, tbd

agent0_pose = pred_pose[:256]
agent1_pose = pred_pose[256:]
agent0_location = pred_location[:256]
agent1_location = pred_location[256:]

print(agent0_location.shape)
print(agent1_location.shape)

ate,aligned_location = utils.compute_ate(agent0_location,gt_location0)
print('{}, ate: {}'.format("Agent0",ate))

ate,aligned_location = utils.compute_ate(agent1_location,gt_location1)
print('{}, ate: {}'.format("Agent1",ate))
