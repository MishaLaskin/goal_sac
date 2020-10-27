N='2'
R='Sparse'
O='Dictstate'
IM_SIZE='84'
ST='True'
CASE='Pickandplace'
ENV_NAME=FetchBlockHRL_${N}Blocks_${R}Reward_${O}Obs_${IM_SIZE}Rendersize_${ST}Stackonly_${CASE}Case-v1

#FetchBlockHRL_4Blocks_SparseReward_DictstateObs_84Rendersize_TrueStackonly_PickandplaceCase-v1


#CUDA_VISIBLE_DEVICES=0 python train.py env=${ENV_NAME} her_iters=0 her_strat=future experiment=fetch_pick_her0_futureA save_video=true eval_frequency=10000 &
#CUDA_VISIBLE_DEVICES=1 python train.py env=${ENV_NAME} her_iters=0 her_strat=future experiment=fetch_pick_her0_futureB save_video=true eval_frequency=25000 &
CUDA_VISIBLE_DEVICES=2 taskset 0,1,2,3,4,5,6,7 python train.py env=${ENV_NAME} her_iters=4 her_strat=future experiment=fetch_pick_her4_2block save_video=true eval_frequency=10000
#CUDA_VISIBLE_DEVICES=3 python train.py env=${ENV_NAME} her_iters=4 her_strat=future experiment=fetch_pick_her4_futureB save_video=true eval_frequency=25000 &
#CUDA_VISIBLE_DEVICES=4 python train.py env=${ENV_NAME} her_iters=4 her_strat=future experiment=fetch_pick_her4_futureC save_video=true eval_frequency=25000 &
#CUDA_VISIBLE_DEVICES=6 python train.py env=FetchPickAndPlace-v1 her_iters=10 her_strat=future experiment=fetch_pick_her10_future &
#CUDA_VISIBLE_DEVICES=7 python train.py env=FetchPickAndPlace-v1 her_iters=12 her_strat=future experiment=fetch_pick_her12_future &
#CUDA_VISIBLE_DEVICES=7 python train.py env=FetchReach-v1 her_iters=4 her_strat=future experiment=fetch_reach 
CUDA_VISIBLE_DEVICES=0 taskset 0,1,2,3,4,5,6,7 python train.py env=FetchBlockHRL_3Blocks_SparseReward_DictstateObs_84Rendersize_TrueStackonly_PutdownCase-v1 her_iters=4 her_strat=future experiment=fetch_putdown_3block_randomreset_lrsched save_video=true eval_frequency=10000
CUDA_VISIBLE_DEVICES=1 taskset 0,1,2,3,4,5,6,7 python train.py env=FetchBlockHRL_2Blocks_SparseReward_DictstateObs_84Rendersize_TrueStackonly_ReachblockCase-v1 her_iters=4 her_strat=future experiment=0.2_fetch_reachblock_2 save_video=true eval_frequency=10000
CUDA_VISIBLE_DEVICES=1 taskset 0,1,2,3,4,5,6,7 python train.py env=FetchBlockHRL_3Blocks_SparseReward_DictimageObs_84Rendersize_TrueStackonly_PutdownCase-v1 her_iters=0 her_strat=future experiment=fetch_image_pickup_2block save_video=true eval_frequency=10000