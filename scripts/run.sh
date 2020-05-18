N='3'
R='Sparse'
O='Dictstate'
IM_SIZE='84'
ST='True'
CASE='Verticalpick'
ENV_NAME=FetchBlockHRL_${N}Blocks_${R}Reward_${O}Obs_${IM_SIZE}Rendersize_${ST}Stackonly_${CASE}Case-v1


CUDA_VISIBLE_DEVICES=0 python train.py env=${ENV_NAME} her_iters=0 her_strat=future experiment=fetch_pick_her0_futureA save_video=true eval_frequency=10000 &
#CUDA_VISIBLE_DEVICES=1 python train.py env=${ENV_NAME} her_iters=0 her_strat=future experiment=fetch_pick_her0_futureB save_video=true eval_frequency=25000 &
CUDA_VISIBLE_DEVICES=2 python train.py env=${ENV_NAME} her_iters=4 her_strat=future experiment=fetch_pick_her4_futureA save_video=true eval_frequency=10000 &
#CUDA_VISIBLE_DEVICES=3 python train.py env=${ENV_NAME} her_iters=4 her_strat=future experiment=fetch_pick_her4_futureB save_video=true eval_frequency=25000 &
#CUDA_VISIBLE_DEVICES=4 python train.py env=${ENV_NAME} her_iters=4 her_strat=future experiment=fetch_pick_her4_futureC save_video=true eval_frequency=25000 &
#CUDA_VISIBLE_DEVICES=6 python train.py env=FetchPickAndPlace-v1 her_iters=10 her_strat=future experiment=fetch_pick_her10_future &
#CUDA_VISIBLE_DEVICES=7 python train.py env=FetchPickAndPlace-v1 her_iters=12 her_strat=future experiment=fetch_pick_her12_future &

#CUDA_VISIBLE_DEVICES=7 python train.py env=FetchReach-v1 her_iters=4 her_strat=future experiment=fetch_reach 