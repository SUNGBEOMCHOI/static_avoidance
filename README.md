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

[robot]
robot.initialpy = -2 robot.gy = 1
robot.radius = 0.5

[obstacle]
num = 5
radius = round(random.choice(np.linspace(self.grid_size*2, self.grid_size * 5, num=6)), 2)
x_position = round(random.choice(np.linspace(-self.square_width//2, self.square_width//2, num=int(self.square_width))), 2)
y_position = round(random.choice(np.linspace(-self.circle_radius + 2, self.square_width//2, num=int(self.square_width))), 2)

model : model.pth

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

[robot]
robot.initialpy = -4 robot.gy = 4
robot.radius = 0.5

[obstacle]
num = 5
radius = round(random.choice(np.linspace(self.grid_size*2, self.grid_size * 5, num=6)), 2)
x_position = round(random.choice(np.linspace(-self.square_width//2, self.square_width//2, num=int(self.square_width))), 2)
y_position = round(random.choice(np.linspace(-self.circle_radius + 2, self.square_width//2, num=int(self.square_width))), 2)

model : resumed_rl_model.pth

### third training
변경점
1. time_step * 2 이후의 value를 예상함.
2. time_step * 2 이후의 충돌이 예상되면 그 action을 취하지 않음
3. Lr 0.01 -> 0.001
4. epsilon_start = 0.5 -> 0.3

[reward]
success_reward = 1
collision_penalty = -0.04
discomfort_dist = 0.05
discomfort_penalty_factor = 0.02


[sim]
square_width = 8
circle_radius = 4
grid_size = 0.2

[robot]
robot.initialpy = -4 robot.gy = 4
robot.radius = 0.5

[obstacle]
num = 5
radius = round(random.choice(np.linspace(self.grid_size*2, self.grid_size * 5, num=6)), 2)
x_position = round(random.choice(np.linspace(-self.square_width//2, self.square_width//2, num=int(self.square_width))), 2)
y_position = round(random.choice(np.linspace(-self.circle_radius + 2, self.square_width//2, num=int(self.square_width))), 2)

model : resumed_rl_model2.pth
