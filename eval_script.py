all_rews = []
n_itr = 100

for _ in range(n_itr):
	obs = self.env.reset()
	first_obs = obs.copy()
	#print("obs", first_obs, "\n")
	self.agent.reset()
	done = False
	r = 0
	while not done: 
		action = self.agent.act(obs['observation'], obs['desired_goal'], sample=True)
		obs, reward, done, _ = self.env.step(action)
		r += reward
	#print(self.env.env.block_ind, self.env.env.goal_pos, r)
	all_rews.append(r)

reward = -25
seed = 1
while True:
r = 0
done = False
self.env.env.seed(19)
obs = self.env.reset()
print("des goal", obs['desired_goal'])
with torch.no_grad():
	while not done: 
		action = self.agent.act(obs['observation'], obs['desired_goal'], sample=True)
		print(action, self.env.block_ind, obs['observation'])
		obs, reward, done, _ = self.env.step(action)
		r += reward
	if r == -25:
		break
	seed += 1

print(self.env.env.block_ind, self.env.env.goal_pos, r)
