# 用例3
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import time
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

def p(x):
    return 1 + 0.0 * torch.sin(x)

MSE = torch.nn.MSELoss()
start_time = time.time()
model_dict = {}
best_lambda_c = 0.0
num_lambda = 10
alpha = 1.0
delta = 1.0

with open('best_results_periodic.txt', 'w') as file:
    pass


for i in range(1, num_lambda + 1):
    #for j in range(1, i):
    #    model_dict[j] = torch.load(f'Vmodel_{j}.pt')

    patience = 2000
    min_loss = float('inf')
    no_improvement_count = 0

    num_epoch = 20000
    loss_list = []
    iteration_list = []
    model = NN(input_size=1, hidden_size=32, output_size=1, depth=4, act=torch.nn.SiLU).to(device)
    lambda_c = torch.tensor([best_lambda_c], requires_grad=True).to(device)
    lambda_c = torch.nn.Parameter(lambda_c)
    model.register_parameter('lambda_c', lambda_c)

    #for name, parameters in model.named_parameters():
    #   print(name, ':', parameters.size())

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler_lr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.9)
    for epoch in range(num_epoch):

        optimizer.zero_grad()

        boundary_point_number = 2
        boundary_point = torch.tensor([[0], [torch.pi]])

        inner_point_number = 1000
        inner_point = torch.linspace(0, np.pi, inner_point_number).unsqueeze(1)
        #inner_point = torch.empty(inner_point_number, 1).uniform_(0.0, torch.pi)

        integral_point_number = 200
        h = torch.pi / integral_point_number
        integral_point = torch.arange(h/2, torch.pi, h).reshape(integral_point_number, 1)

        inner_point = inner_point.to(device)
        boundary_point = boundary_point.to(device)
        integral_point = integral_point.to(device)
        inner_point.requires_grad = True
        boundary_point.requires_grad = True

        p_inner = p(inner_point)

        u_inner = model(inner_point)
        du_dx = torch.autograd.grad(outputs=u_inner, inputs=inner_point,
                                    grad_outputs=torch.ones_like(u_inner), retain_graph=True, create_graph=True)[0]

        dpu_xdx = torch.autograd.grad(outputs=du_dx*p_inner, inputs=inner_point,
                                     grad_outputs=torch.ones_like(du_dx), retain_graph=True, create_graph=True)[0]

        f = dpu_xdx + p_inner * lambda_c * u_inner
        loss_pde = torch.mean(f ** 2)

        u_boundary = model(boundary_point)

        du_boundary_dx = torch.autograd.grad(outputs=u_boundary, inputs=boundary_point,
          grad_outputs=torch.ones_like(u_boundary), retain_graph=True, create_graph=True)[0]

        loss_boundary = (u_boundary[0] - u_boundary[1]) ** 2 + (du_boundary_dx[0] - du_boundary_dx[1]) ** 2

        u_integral = torch.sum(model(integral_point) * model(integral_point) * p(integral_point)) * h

        loss_integral = (u_integral - 1) ** 2
        loss_orth = torch.tensor([[0.0]]).to(device)
        for j in range(1, i):
            orth = torch.sum(model(integral_point) * model_dict[j](integral_point) * p(integral_point)) * h
            loss_orth += orth ** 2

        loss = alpha * loss_pde + loss_boundary + loss_integral/u_integral + 10 * delta * loss_orth
        loss.backward()
        optimizer.step()
        scheduler_lr.step()

        # 检查是否需要早停
        if loss.item() < min_loss:
            #print(f"Loss decreased from {min_loss:.6f} to {loss.item():.6f}. Saving model...")
            min_loss = loss.item()
            no_improvement_count = 0  # 重置计数器
            model_dict[i] = model
            #torch.save(model, f'Vmodel_{i}.pt')  # 保存当前最佳模型
            best_loss = loss.item()
            best_loss_pde = loss_pde.item()
            best_loss_boundary = loss_boundary.item()
            best_loss_orth = loss_orth.item()
            best_epoch = epoch
            best_lambda_c = lambda_c.item()

        else:
            no_improvement_count += 1
            #print(f"No improvement for {no_improvement_count} epochs.")

        if no_improvement_count >= patience:
            #print(f"Performance did not improve for {patience} epochs. Training stopped.")
            break  # 应用早停策略，跳出循环

        if epoch % 1000 == 0:
            print("iter:", epoch, "  loss_pde:%.6f" % loss_pde.item(), "  loss_boundary:%.6f" % loss_boundary.item(),
                  "  loss_integral:%.6f" % loss_integral.item(), "  loss_orth:%.6f" % loss_orth.item(),
                  "  lambda_%d:%.6f" % (i, lambda_c.item()), "   delta:%.6f" % delta)


            iteration_list.append(epoch)
            loss_list.append(loss_pde)
        if epoch > np.round(num_epoch/2):
            alpha = 1.0
            delta = 1.0
    print("The best iter:", best_epoch, "  loss_pde:%.6f" % best_loss_pde, "  loss_boundary:%.6f" % best_loss_boundary,
        "  loss_orth:%.6f" % best_loss_orth, "  lambda_%d:%.6f" % (i, best_lambda_c))
    # 打开一个文件以追加模式（'a'），这样每次写入都不会覆盖之前的内容
    with open('best_results_periodic.txt', 'a') as file:
        # 使用和print语句相似的格式化字符串，但是写入到文件中
        file.write("The best iter: {}  loss_pde:{:.6f}  loss_boundary:{:.6f}"
                   "  loss_orth:{:.6f}  lambda_{}:{:.6f}\n".format(
            best_epoch, best_loss_pde, best_loss_boundary, best_loss_orth, i, best_lambda_c
        ))
    alpha = 1.0
    if i==1:
        delta = 1.0
    else:
        delta = (best_lambda_c+1)*10
    end_time = time.time()
    print('Took %f second' % (end_time - start_time))


