import math
import torch
import time
from collections import OrderedDict
import scipy.io
import os
# CUDA support
if torch.cuda.is_available():
    print('cuda')
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    print('cpu')

class NN(torch.nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            output_size,
            depth,
            act,
    ):
        super(NN, self).__init__()

        layers = [('input', torch.nn.Linear(input_size, hidden_size))]
        layers.append(('input_activation', act()))
        for i in range(depth):
            layers.append(
                ('hidden_%d' % i, torch.nn.Linear(hidden_size, hidden_size))
            )
            layers.append(('activation_%d' % i, act()))
        layers.append(('output', torch.nn.Linear(hidden_size, output_size)))

        layerDict = OrderedDict(layers)
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out

model_dict = {}
u = {}
u_x = {}
u_xx = {}
R_u = {}
T_u = {}
num_lambda = 7
num_epoch = 70000
model_parameters = []
boundary_point_number = 2
boundary_point = torch.tensor([[0], [torch.pi]])

integral_point_number = 500
h = torch.pi/integral_point_number
integral_point = torch.arange(h/2, torch.pi, h).reshape(integral_point_number, 1)

boundary_point = boundary_point.to(device)
integral_point = integral_point.to(device)
start_time = time.time()
min_loss = float('inf')
integral_point.requires_grad = True

best_lambda = torch.zeros(num_lambda+1, dtype=torch.float32).to(device)

for i in range(1, num_lambda + 1):
    model_dict[i] = NN(input_size=1, hidden_size=32, output_size=1, depth=4, act=torch.nn.Tanh).to(device)
    model_parameters += list(model_dict[i].parameters())

optimizer = torch.optim.Adam(model_parameters, lr=0.01, weight_decay=1e-8)
scheduler_lr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)

for epoch in range(num_epoch):

    optimizer.zero_grad()
    loss = torch.tensor([0.0], dtype=torch.float32).to(device)

    for i in range(1, num_lambda + 1):
        u[i] = model_dict[i](integral_point)
        u_x[i] = torch.autograd.grad(outputs=u[i], inputs=integral_point, grad_outputs=torch.ones_like(u[i]),
                                 retain_graph=True, create_graph=True)[0]
        u_xx[i] = torch.autograd.grad(outputs=u_x[i], inputs=integral_point, grad_outputs=torch.ones_like(u_x[i]),
                                  retain_graph=True, create_graph=True)[0]
        R_u[i] = - torch.sum(u_xx[i] * u[i]) / torch.sum(u[i] ** 2)

        T_u[i] = u_xx[i] + R_u[i] * u[i]

        loss_l2 = torch.mean(T_u[i] ** 2)
        loss_max = torch.max(torch.abs(T_u[i]))
        loss_boundary = torch.mean(model_dict[i](boundary_point) ** 2)
        loss_integral = (torch.sum(model_dict[i](integral_point) ** 2) * h - 1) ** 2
        loss_R = R_u[i] ** 2
        loss_orth = torch.tensor([0.0], dtype=torch.float32).to(device)

        for j in range(1, i):
            orth = torch.sum(model_dict[j](integral_point) * model_dict[i](integral_point)) * h
            loss_orth += orth ** 2

        loss += 0.1 * loss_l2 + 0.1 * loss_max + 0.5 * loss_boundary + 1.5 * loss_integral + 2 * loss_orth \
                + 1/i * loss_R / 1000

    loss.backward()
    optimizer.step()
    scheduler_lr.step()

    if loss.item() < min_loss:
        best_epoch = epoch
        min_loss = loss.item()
        best_loss = min_loss
        for i in range(1, num_lambda + 1):
            best_lambda[i] = R_u[i].item()

    if epoch % 1000 == 0:

        print("epoch:", epoch, "  loss:%.6f" % loss.item(), "  loss_l2:%.6f" % loss_l2.item(),
              "  loss_max:%.6f" % loss_max.item(), "  loss_boundary:%.6f" % loss_boundary.item(),
              "  loss_integral:%.6f" % loss_integral.item(),
              "  lambda:", ", ".join(["%.6f" % R_u[i] for i in range(1, num_lambda + 1)]))
end_time = time.time()
print("epoch:", best_epoch, "  best_loss:%.6f" % best_loss,"  best_lambda:",
      ", ".join(["%.6f" % best_lambda[i] for i in range(1, num_lambda + 1)]))
print('Took %f second' % (end_time - start_time))

with open('ICLR_Dirichlet.txt', 'w') as file:
    # 写入第一行内容
    file.write("epoch: %d  best_loss:%.6f  best_lambda: " % (best_epoch, best_loss))
    file.write(", ".join(["%.6f" % best_lambda[i] for i in range(1, num_lambda + 1)]) + '\n')

    # 写入第二行内容
    file.write('Took %f second\n' % (end_time - start_time))
