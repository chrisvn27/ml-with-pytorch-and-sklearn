import gymnasium as gym

### Brief intro  to OpenAI  Gym
env = gym.make('CartPole-v1')
print(env.observation_space)
print(env.action_space)
print(env.reset())
print(env.step(action=0))
#  outputs for  step  are  now:  obs, reward, terminated, truncated, info = env.step(action)
print(env.step(action=1))
###