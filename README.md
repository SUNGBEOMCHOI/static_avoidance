# static_avoidance

### first training
[reward]
success_reward = 1
collision_penalty = -0.025
discomfort_dist = 0.1
discomfort_penalty_factor = 0.02


[sim]
square_width = 3
circle_radius = 1
grid_size = 0.2

robot.initialpy = -2 robot.gy = 1
model : rl_model.pth

### second training
[reward]
success_reward = 1
collision_penalty = -0.025
discomfort_dist = 0.05
discomfort_penalty_factor = 0.02


[sim]
square_width = 8
circle_radius = 4
grid_size = 0.2

robot.initialpy = -4 robot.gy = 4
model : resumed_rl_model.pth
