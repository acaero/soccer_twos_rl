import soccer_twos

env = soccer_twos.make(render=True, time_scale=1)
print("Observation Space: ", env.observation_space.shape)
print("Action Space: ", env.action_space.shape)

team0_reward = 0
team1_reward = 0



i = 0
while True:
    i+=1
    obs, reward, done, info = env.step(
        {
            0: env.action_space.sample(),
            1: env.action_space.sample(),
            2: env.action_space.sample(),
            3: env.action_space.sample(),
        }
    )

    # if i % 1000 == 0:
    #     print(
    #     f"obs: {obs}",
    #     f"reward: {reward}",
    #     f"done: {done}",
    #     f"info: {info}",
    #     sep="\n\n")

    team0_reward += reward[0] + reward[1]
    team1_reward += reward[2] + reward[3]
    if done["__all__"]:
        print("Total Reward: ", team0_reward, " x ", team1_reward)
        team0_reward = 0
        team1_reward = 0
        env.reset()