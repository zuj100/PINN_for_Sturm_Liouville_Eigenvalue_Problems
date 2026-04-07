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



num_lambda = 3
num_epoch = 10000
patience = 10000
p_num = 400  # inner point number
loop = 1
width = 32

filename = f"loop_{loop}_epoch_{num_epoch}_point_{p_num}_width_{width}.txt"
with open(filename, 'w') as file:
    pass
for m in range(1, loop+1):
    best_lambda_c = 0.0
    alpha = 1.0
    delta = 1.0
    start_time = time.time()
    model_dict = {}
    for i in range(1, num_lambda + 1):
        #for j in range(1, i):
        #    model_dict[j] = torch.load(f'Vmodel_{j}.pt')

        min_loss = float('inf')
        no_improvement_count = 0
        loss_list = []
        iteration_list = []
        model = NN(input_size=1, hidden_size=width, output_size=1, depth=4, act=torch.nn.SiLU).to(device)

        lambda_c = torch.tensor([best_lambda_c], requires_grad=True).to(device)
        lambda_c = torch.nn.Parameter(lambda_c)
        model.register_parameter('lambda_c', lambda_c)

        #for name, parameters in model.named_parameters():
        #   print(name, ':', parameters.size())

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler_lr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
        for epoch in range(num_epoch):

            optimizer.zero_grad()

            boundary_point_number = 2
            boundary_point = torch.tensor([[0], [torch.pi]])

            inner_point_number = p_num
            #inner_point = torch.linspace(0, np.pi, inner_point_number).unsqueeze(1)
            inner_point = torch.empty(inner_point_number, 1).uniform_(0.0, torch.pi)

            integral_point_number = p_num
            h = torch.pi / integral_point_number
            integral_point = torch.arange(h/2, torch.pi, h).reshape(integral_point_number, 1)
            #integral_point = inner_point.clone()

            inner_point = inner_point.to(device)
            boundary_point = boundary_point.to(device)
            integral_point = integral_point.to(device)
            inner_point.requires_grad = True

            p_inner = p(inner_point)
            #dp_dx = torch.autograd.grad(outputs=p_inner, inputs=inner_point,
                                      #  grad_outputs=torch.ones_like(p_inner), retain_graph=True, create_graph=True)[0]

            u_inner = model(inner_point)
            du_dx = torch.autograd.grad(outputs=u_inner, inputs=inner_point,
                                        grad_outputs=torch.ones_like(u_inner), retain_graph=True, create_graph=True)[0]

            dpu_xdx = torch.autograd.grad(outputs=du_dx*p_inner, inputs=inner_point,
                                         grad_outputs=torch.ones_like(du_dx), retain_graph=True, create_graph=True)[0]

            f = dpu_xdx + p_inner * lambda_c * u_inner
            loss_pde = torch.mean(f ** 2)

            boundary_predict = model(boundary_point)
            loss_boundary = torch.mean(boundary_predict ** 2)

            u_integral = torch.sum(model(integral_point) * model(integral_point) * p(integral_point)) * h
            #u_integral = torch.mean(model(integral_point) * model(integral_point) * p(integral_point)) * torch.pi

            loss_integral = (u_integral - 1) ** 2
            loss_orth = torch.tensor([[0.0]]).to(device)
            for j in range(1, i):
                #orth = torch.sum(model(integral_point) * model_dict[j](integral_point) * p(integral_point)) * h
                orth = torch.mean(model(integral_point) * model_dict[j](integral_point) * p(integral_point)) * torch.pi
                loss_orth += orth ** 2

            loss = alpha * loss_pde + loss_boundary + loss_integral/u_integral + delta * loss_orth
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
            #if epoch > np.round(num_epoch/2):
            #    alpha = 1.0
            #    delta = 1
        print("The best iter:", best_epoch, "  loss_pde:%.6f" % best_loss_pde, "  loss_boundary:%.6f" % best_loss_boundary,
            "  loss_orth:%.6f" % best_loss_orth, "  lambda_%d:%.6f" % (i, best_lambda_c))
        # 打开一个文件以追加模式（'a'），这样每次写入都不会覆盖之前的内容
        duration_time = time.time() - start_time
        print('Took %f second' % duration_time)
        #with open(filename, 'a') as file:
        #    # 使用和print语句相似的格式化字符串，但是写入到文件中
        #    file.write("The best iter: {}  loss_pde:{:.6f}  loss_boundary:{:.6f}"
        #               "  loss_orth:{:.6f}  lambda_{}:{:.6f} duration_time: {:.1f}\n".format(
        #        best_epoch, best_loss_pde, best_loss_boundary, best_loss_orth, i, best_lambda_c, duration_time
        #    ))
        alpha = 1.0
        delta = (best_lambda_c+1)*2

        # test error
        test_inner_point = torch.empty(100, 1).uniform_(0.0, torch.pi)
        test_inner_point = test_inner_point.to(device)
        test_inner_point.requires_grad = True
        u_test = model_dict[i](test_inner_point)
        du_test_dx = torch.autograd.grad(outputs=u_test, inputs=test_inner_point,
                                    grad_outputs=torch.ones_like(u_test), retain_graph=True, create_graph=True)[0]
        p_test = p(test_inner_point)
        dpu_test_xdx = torch.autograd.grad(outputs=du_test_dx * p_test, inputs=test_inner_point,
                                      grad_outputs=torch.ones_like(du_test_dx), retain_graph=True, create_graph=True)[0]

        f_test = dpu_test_xdx + p_test * best_lambda_c * u_test
        loss_pde_test = torch.mean(f_test ** 2)
        #print("The best iter:", best_epoch, "  loss_pde_test:%.6f" % loss_pde_test)
        with open(filename, 'a') as file:
            # 使用和print语句相似的格式化字符串，但是写入到文件中
            file.write("lambda_{}:{:.8f}  train:{:.8f}  test:{:.8f} ".format(i, best_lambda_c, best_loss_pde, loss_pde_test))

    with open(filename, 'a') as file:
        # 使用和print语句相似的格式化字符串，但是写入到文件中
        file.write("\n")











# 在你的代码末尾添加以下绘制代码

import matplotlib.pyplot as plt
import numpy as np

# 绘制所有特征函数
fig, axes = plt.subplots(3, 3, figsize=(12, 10))
axes = axes.flatten()

# 生成用于绘图的点
x_plot = torch.linspace(0, torch.pi, 1000).reshape(-1, 1).to(device)

for i in range(1, min(num_lambda + 1, 10)):  # 最多绘制9个
    if i <= len(model_dict):
        # 获取模型并计算特征函数
        model = model_dict[i]
        model.eval()
        with torch.no_grad():
            y_plot = model(x_plot).cpu().numpy()

        x_plot_np = x_plot.cpu().numpy()

        # 在子图中绘制
        ax = axes[i - 1]
        ax.plot(x_plot_np, y_plot, 'b-', linewidth=2, label='PINN')

        # 添加理论解作为对比（对于p(x)=1的情况）
        # 理论特征函数: sin(n*x)，归一化后为 sqrt(2/pi)*sin(n*x)
        y_theory = np.sqrt(2 / np.pi) * np.sin(i * x_plot_np)
        ax.plot(x_plot_np, y_theory, 'r--', linewidth=1, alpha=0.7, label='Theory')

        # 设置子图标题和标签
        if i <= len(model_dict):
            # 获取对应的特征值
            lambda_value = 0.0
            for param_name, param in model.named_parameters():
                if 'lambda_c' in param_name:
                    lambda_value = param.item()
                    break
            ax.set_title(f'n={i}, λ={lambda_value:.4f} (Theory: {i ** 2})', fontsize=10)

        ax.set_xlabel('x')
        ax.set_ylabel(f'u_{i}(x)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlim([0, np.pi])

# 隐藏多余的子图
for i in range(len(model_dict), 9):
    axes[i].set_visible(False)

plt.suptitle('Eigenfunctions of Sturm-Liouville Problem', fontsize=14, y=1.02)
plt.tight_layout()

# 保存图片
plt.savefig(f'eigenfunctions_loop_{loop}_epoch_{num_epoch}_point_{p_num}_width_{width}.png', dpi=150,
            bbox_inches='tight')
plt.show()

# 另外绘制一个图：所有特征函数在同一张图上
plt.figure(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0, 0.9, num_lambda))

for i in range(1, min(num_lambda + 1, 8)):  # 最多绘制前7个
    if i <= len(model_dict):
        model = model_dict[i]
        model.eval()
        with torch.no_grad():
            y_plot = model(x_plot).cpu().numpy()

        x_plot_np = x_plot.cpu().numpy()

        # 获取特征值
        lambda_value = 0.0
        for param_name, param in model.named_parameters():
            if 'lambda_c' in param_name:
                lambda_value = param.item()
                break

        plt.plot(x_plot_np, y_plot, color=colors[i - 1], linewidth=2,
                 label=f'u_{i}(x), λ={lambda_value:.3f}')

plt.xlabel('x', fontsize=12)
plt.ylabel('u(x)', fontsize=12)
plt.title('All Eigenfunctions', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
plt.xlim([0, np.pi])

# 保存图片
plt.savefig(f'all_eigenfunctions_loop_{loop}_epoch_{num_epoch}_point_{p_num}_width_{width}.png', dpi=150,
            bbox_inches='tight')
plt.show()

print("\nPlots saved successfully!")