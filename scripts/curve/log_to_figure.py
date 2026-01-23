import re
import matplotlib.pyplot as plt
import numpy as np

def extract_test_losses(log_file_path):
    """
    从日志文件中提取Rel. Test L2 Loss数据
    
    参数:
    log_file_path (str): 日志文件路径
    
    返回:
    tuple: (default_scaled, two_circles_scaled) 或 None（如果提取失败）
    """    
    try:
        # 读取文件的最后一行
        with open(log_file_path, 'r') as file:
            lines = file.readlines()
            if not lines:
                return None
            
            last_line = lines[-1].strip()
        
        
        # 直接提取数值的简化方法
        # 匹配格式: Rel. Test L2 Loss : {'Default': 0.000741..., 'Two Circles': 0.000980...}
        values_pattern = r"Rel\. Test L2 Loss\s*:\s*\{'Default'\s*:\s*([\d\.Ee+-]+)\s*,\s*'Two Circles'\s*:\s*([\d\.Ee+-]+)\}"
        values_match = re.search(values_pattern, last_line)
        
        if not values_match:
            return None
        
        # 提取两个数值并转换为浮点数
        # 使用float()可以处理科学计数法如1.23e-4
        default_value = float(values_match.group(1))
        two_circles_value = float(values_match.group(2))
        
        # 乘以100并保留4位小数
        default_scaled = round(default_value * 100, 4)
        two_circles_scaled = round(two_circles_value * 100, 4)
        
        return default_scaled, two_circles_scaled
        
    except FileNotFoundError:
        print(f"文件未找到: {log_file_path}")
        return None
    except Exception as e:
        print(f"处理文件时出错: {e}")
        return None
    
def extract_test_losses_with_epochcheck(log_file_path,epochs = 500):
    """
    从日志文件中提取Rel. Test L2 Loss数据
    
    参数:
    log_file_path (str): 日志文件路径
    
    返回:
    tuple: (default_scaled, two_circles_scaled) 或 None（如果提取失败）
    """    
    try:
        # 读取文件的最后一行
        with open(log_file_path, 'r') as file:
            lines = file.readlines()
            if not lines:
                return None
            
            last_line = lines[-1].strip()
        
        # 首先检查Epoch是否为499
        epoch_pattern = r"Epoch\s*:\s*(\d+)"
        epoch_match = re.search(epoch_pattern, last_line)
        
        # 如果Epoch信息存在且不是499，返回None表示失败
        if epoch_match:
            epoch_value = int(epoch_match.group(1))
            if epoch_value != epochs-1:
                print(f"警告: 最后一行的Epoch={epoch_value}, 不是499, 文件: {log_file_path}")
                return None
        else:
            # 如果没有找到Epoch信息，也视为异常
            print(f"警告: 未找到Epoch信息, 文件: {log_file_path}")
            return None
        
        # 提取数值的简化方法
        # 匹配格式: Rel. Test L2 Loss : {'Default': 0.000741..., 'Two Circles': 0.000980...}
        values_pattern = r"Rel\. Test L2 Loss\s*:\s*\{'Default'\s*:\s*([\d\.Ee+-]+)\s*,\s*'Two Circles'\s*:\s*([\d\.Ee+-]+)\}"
        values_match = re.search(values_pattern, last_line)
        
        if not values_match:
            print(f"警告: 未找到测试损失数据, 文件: {log_file_path}")
            return None
        
        # 提取两个数值并转换为浮点数
        # 使用float()可以处理科学计数法如1.23e-4
        default_value = float(values_match.group(1))
        two_circles_value = float(values_match.group(2))
        
        # 乘以100并保留4位小数
        default_scaled = round(default_value * 100, 4)
        two_circles_scaled = round(two_circles_value * 100, 4)
        
        return default_scaled, two_circles_scaled
        
    except FileNotFoundError:
        print(f"文件未找到: {log_file_path}")
        return None
    except Exception as e:
        print(f"处理文件时出错: {e}")
        return None
    
def get_the_loss_from_log(kernel_type, k_max, n_train, layers, act, prefix = "1_1_5_2d", grad="True", geo="True", geoint="True",
                          n_test=1000, n_two_circles_test=1000,
                          to_divide_factor=20.0, batch_size=8):
    """
    构建日志文件路径并提取损失数据
    
    参数:
    prefix (str): 前缀
    kernel_type (str): 核类型
    grad (str): 是否使用梯度
    geo (str): 是否使用几何信息
    geoint (str): 是否使用几何积分
    k_max (int): 最大k值
    n_train (int): 训练样本数
    n_test (int): 测试样本数
    n_two_circles_test (int): 双圆测试样本数
    to_divide_factor (float): 除数因子
    batch_size (int): 批量大小
    layers (str): 网络层配置
    act (str): 激活函数类型
    normal_prod (str): 是否使用法向量乘积
    num_grad (int): 梯度数量（1或3）
    
    返回:
    tuple: (default_scaled, two_circles_scaled) 或 None（如果提取失败）
    """
    log_dir = f"log/log_new/{prefix}_{kernel_type}/{layers}_{act}/"
    log_file_path = (f"{log_dir}N{n_train}_Ntest{n_test},{n_two_circles_test}_k{k_max}_L10_"
                     f"bsz{batch_size}_factor{to_divide_factor}_grad{grad}_geo{geo}_geoint{geoint}.log")
    
    return extract_test_losses_with_epochcheck(log_file_path)

def generate_combined_figure():
    plt.rc('xtick', labelsize=16)  # X轴刻度标签字号
    plt.rc('ytick', labelsize=16)  # Y轴刻度标签字号

    kernel_types = ["sp_laplace","dp_laplace","modified_dp_laplace","adjoint_dp_laplace","stokes"]
    formal_names = {"sp_laplace":"Laplacian single\n layer potential",
                    "dp_laplace":"Laplacian double\n layer potential",
                    "modified_dp_laplace":"Modified Laplacian double\n layer potential",
                    "adjoint_dp_laplace":" Adjoint Laplacian\n double layer potential",
                    "stokes":"Stokeslet"}
    
    # Figure 1 参数
    k_max_values = [8,16,32,64]
    n_train_fig1 = 8000
    
    # Figure 2 参数
    k_max_fig2 = 32
    n_values_list = [1000,2000,4000,8000]
    
    # 创建2行5列的子图
    fig, axes = plt.subplots(2, 5, figsize=(22, 11), sharey=True)
    
    # 定义线条样式
    styles_fig1 = {
        "1L-S": {"color": "blue", "ls": "-", "marker": "o", "label": "1L-S"},
        "1L-D": {"color": "blue", "ls": "--", "marker": "s", "label": "1L-D"},
        "5L-S": {"color": "red", "ls": "-", "marker": "o", "label": "5L-S"},
        "5L-D": {"color": "red", "ls": "--", "marker": "s", "label": "5L-D"},
    }
    
    styles_fig2 = {
        "5L-S": {"color": "red", "ls": "-", "marker": "o", "label": "5L-S"},
        "5L-D": {"color": "red", "ls": "--", "marker": "s", "label": "5L-D"},
    }
    
    configs_fig1 = [
        {"layers": "64,64", "act": "none", "name": "1L"},
        {"layers": "64,64,64,64,64,64", "act": "gelu", "name": "5L"}
    ]
    
    configs_fig2 = [
        {"layers": "64,64,64,64,64,64", "act": "gelu", "name": "5L"}
    ]
    
    from matplotlib.lines import Line2D
    
    # 创建第一个图的图例元素（放在顶部）
    legend_elements_fig1 = [
        Line2D([0], [0], color='blue', lw=2, linestyle='-'),
        Line2D([0], [0], color='blue', lw=2, linestyle='--'),
        Line2D([0], [0], color='red', lw=2, linestyle='-'),
        Line2D([0], [0], color='red', lw=2, linestyle='--'),
    ]
    
    legend_labels_fig1 = ['Single-layer linear model\n(Single-curve test)', 'Single-layer linear model\n(Two-curve test)', '5-layer M-PCNO\n(Single-curve test)', '5-layer M-PCNO\n(Two-curve test)']
    
    # 第一行：Figure 1 (Varying p)
    for col_idx, kernel in enumerate(kernel_types):
        ax = axes[0, col_idx]
        
        # 收集所有配置的数据
        data = {}
        for config in configs_fig1:
            layers = config["layers"]
            act = config["act"]
            config_name = config["name"]
            
            singles, doubles = [], []
            for k_max in k_max_values:
                result = get_the_loss_from_log(
                    kernel_type=kernel, 
                    k_max=k_max, 
                    layers=layers, 
                    act=act, 
                    n_train=n_train_fig1
                )
                if result:
                    single, double = result
                    singles.append(single)
                    doubles.append(double)
                else:
                    singles.append(None)
                    doubles.append(None)
            
            data[f"{config_name}-S"] = singles
            data[f"{config_name}-D"] = doubles
        
        # 绘制四条线
        for line_name, line_style in styles_fig1.items():
            values = data.get(line_name, [])
            if any(values):
                ax.plot([k for k in k_max_values],
                        [v * 1e-2 if v else np.nan for v in values], 
                        color=line_style["color"],
                        linestyle=line_style["ls"],
                        linewidth=2,
                        marker=line_style["marker"],
                        markersize=6)
        
        # 设置log-log坐标
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(min(k_max_values) * 0.8, max(k_max_values) * 1.2)
        ax.set_xticks(k_max_values, k_max_values)
        
        # 理论参考线 y = 1/p
        ax.plot(k_max_values,
                [1/k for k in k_max_values],
                color='green',
                linestyle=':',
                linewidth=2,
                alpha=0.7)
        
        # 添加文本标注
        x_mid = np.exp((np.log(k_max_values[0]) + np.log(k_max_values[-1])) / 2)
        y_mid = 1/x_mid
        ax.text(x_mid, y_mid,
                r'$1/p$',
                fontsize=16,
                color='green',
                alpha=0.8,
                verticalalignment='bottom',
                horizontalalignment='left')
        
        # 设置标题和标签
        ax.set_title(formal_names[kernel], fontsize=20, pad=8, fontweight='bold')
        if col_idx == 0:
            ax.set_ylabel('Test Error', fontsize=18)
        else:
            ax.set_ylabel('')
            ax.tick_params(axis='y', labelleft=False)
        
        ax.set_xlabel('p', fontsize=18)
        
        # 添加网格
        ax.grid(True, linestyle=':', alpha=0.3)
    
    # 第二行：Figure 2 (Varying N)
    for col_idx, kernel in enumerate(kernel_types):
        ax = axes[1, col_idx]
        
        # 收集所有配置的数据
        data = {}
        for config in configs_fig2:
            layers = config["layers"]
            act = config["act"]
            config_name = config["name"]
            
            singles, doubles = [], []
            for n_train in n_values_list:
                result = get_the_loss_from_log(
                    kernel_type=kernel, 
                    k_max=k_max_fig2, 
                    layers=layers, 
                    act=act, 
                    n_train=n_train
                )
                if result:
                    single, double = result
                    singles.append(single)
                    doubles.append(double)
                else:
                    singles.append(None)
                    doubles.append(None)
            
            data[f"{config_name}-S"] = singles
            data[f"{config_name}-D"] = doubles
        
        # 绘制两条线
        for line_name, line_style in styles_fig2.items():
            values = data.get(line_name, [])
            if any(values):
                ax.plot(n_values_list,
                        [v * 1e-2 if v else np.nan for v in values],
                        color=line_style["color"],
                        linestyle=line_style["ls"],
                        linewidth=2,
                        marker=line_style["marker"],
                        markersize=6)
        
        # 设置log-log坐标
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(min(n_values_list) * 0.8, max(n_values_list) * 1.2)
        ax.set_xticks(n_values_list, n_values_list)
        
        # 理论参考线 y = 1/√N
        ax.plot(n_values_list,
                [1/np.sqrt(N) for N in n_values_list],
                color='green',
                linestyle=':',
                linewidth=2,
                alpha=0.7)
        
        # 添加文本标注
        x_mid = np.exp((np.log(n_values_list[0]) + np.log(n_values_list[-1])) / 2)
        y_mid = 1/np.sqrt(x_mid)
        if col_idx != 1:
            ax.text(x_mid, y_mid,
                r'$1/\sqrt{n}$',
                fontsize=16,
                color='green',
                alpha=0.8,
                verticalalignment='bottom',
                horizontalalignment='left')
        else:
            ax.text(x_mid, y_mid,
                r'$1/\sqrt{n}$',
                fontsize=16,
                color='green',
                alpha=0.8,
                verticalalignment='top',
                horizontalalignment='right')
        
        # 设置标题和标签
        if col_idx == 0:
            ax.set_ylabel('Test Error', fontsize=18)
        else:
            ax.set_ylabel('')
            ax.tick_params(axis='y', labelleft=False)
        
        ax.set_xlabel('n', fontsize=18)
        
        # 添加网格
        ax.grid(True, linestyle=':', alpha=0.3)
    
    # 在整个图形顶部添加统一图例（只显示第一个图的图例）
    fig.legend(legend_elements_fig1, legend_labels_fig1,
               loc='upper center',
               ncol=len(legend_elements_fig1),
               fontsize=20,
               bbox_to_anchor=(0.55, 1.05),
               frameon=False,
               fancybox=False,
               shadow=False,
               borderpad=1,
               handletextpad=0.5,
               columnspacing=1.5)
    
    # 添加行标题
    axes[0, 0].text(-0.32, 0.5, 'Varying truncated mode number p',
                    fontsize=18, fontweight='bold',
                    rotation=90, verticalalignment='center',
                    transform=axes[0, 0].transAxes)
    
    axes[1, 0].text(-0.32, 0.5, 'Varying training data size N',
                    fontsize=18, fontweight='bold',
                    rotation=90, verticalalignment='center',
                    transform=axes[1, 0].transAxes)
    
    # 调整布局
    plt.tight_layout(rect=[0.05, 0, 1, 0.92])  # 为顶部图例留出空间
    
    # 保存图像
    plt.savefig('kernel_integral_combined_figure.png', dpi=300, bbox_inches='tight')
    plt.show()




def generate_exterior_laplacian_neumann_figure(fontsize=22):
    plt.rc('xtick', labelsize=fontsize)  # X轴刻度标签字号
    plt.rc('ytick', labelsize=fontsize)  # Y轴刻度标签字号

    kernel_type = "exterior_laplacian_neumann"
    
    # Figure 1 参数
    k_max_values = [8,16,32,64]
    n_values = [1000,2000,4000,8000]
    

    single_curve = np.array([[0.034427267342805865    , 0.02299725789576769    , 0.016628885842859743    , 0.01232832932844758],
			[0.027866664484143256    , 0.017473292864859106    , 0.012365857612341642   , 0.009288566693663597],
			[0.03596678347885609     , 0.02144157275557518    , 0.01210837160050869    , 0.0092167739123106],
			[0.045980724170804024    , 0.029172634124755858    , 0.0178871094584465  , 0.010938586924225092]])
    two_curve = np.array([[0.13570469135046007  ,   0.1355136509537697  ,   0.13061414629220963  ,  0.1302403035759926],
			[0.10094071254134178   ,   0.0889949080646038   ,   0.08630543220043183   ,   0.08103378015756607],
			[0.09332142081856727   ,   0.07394308692216874  ,   0.0576402183175087   ,   0.049038773134350774],
			[0.10741320264339448   ,   0.09519950917363167   ,   0.07916956254839898  ,   0.06170262950658798]])
    
    # 创建1行2列的子图
    fig, axe = plt.subplots(1,1, figsize=(7, 6), sharey=True)
    # 第一个图变k_max， 第二个图变n
    axe.loglog(k_max_values, single_curve[:,2], linestyle = "--", marker = "s", color="C0", label=r"$\substack{n=4000 \\ \text{(single-curve test)}}$")
    axe.loglog(k_max_values, two_curve[:,2], color="C0", marker = "o", label=r"$\substack{n=4000 \\ \text{(two-curve test)}}$")
    axe.loglog(k_max_values, single_curve[:,3], linestyle = "--", marker = "s", color="C1", label=r"$\substack{n=8000 \\ \text{(single-curve test)}}$")
    axe.loglog(k_max_values, two_curve[:,3], color="C1", marker = "o", label=r"$\substack{n=8000 \\ \text{(two-curve test)}}$")
    axe.set_xlabel(r"$p$",fontsize=fontsize)
    # axe.set_title("Two-curve test")
    # 理论参考线 y = 1/p
    axe.plot(k_max_values,
                [1/k for k in k_max_values],
                color='green',
                linestyle=':',
                linewidth=2,
                alpha=0.7)
    x_mid = np.exp((np.log(k_max_values[0]) + np.log(k_max_values[-1])) / 2)
    y_mid = 1/x_mid
    axe.text(x_mid, y_mid,
        r'$1/p$',
        fontsize=fontsize,
        color='green',
        alpha=0.8,
        verticalalignment='bottom',
        horizontalalignment='left')
    
    axe.set_ylabel('Rel. test error', fontsize=fontsize)
    axe.set_xscale('log')
    axe.set_yscale('log')
    axe.set_xlim(min(k_max_values) * 0.8, max(k_max_values) * 1.2)
    axe.set_xticks(k_max_values, k_max_values)
    axe.set_yticks([1e-2,5e-2,1e-1], [1e-2,5e-2,1e-1])
    axe.grid(True, linestyle=':', alpha=0.3)
    axe.legend(loc='center right',
            fontsize=fontsize,
            bbox_to_anchor=(1.75, 0.5),
            frameon=False,
            fancybox=False,
            shadow=False,
            borderpad=1,
            handletextpad=0.5,
            columnspacing=1.5)
    
    # 调整布局
    plt.tight_layout(rect=[0.05, 0.05, 0.4, 0.0])  # 为顶部图例留出空间
    
    # 保存图像
    plt.savefig('exterior_laplace_neumann_p_figure.png', dpi=300, bbox_inches='tight')
    plt.show()




     # 创建1行2列的子图
    fig, axe = plt.subplots(1,1, figsize=(7, 6), sharey=True)
    # 第一个图变k_max， 第二个图变n
    axe.loglog(n_values, single_curve[1,:], linestyle = "--", marker = "s", color="C0", label=r"$\substack{p=16 \\ \text{(single-curve test)}}$")
    axe.loglog(n_values, two_curve[1,:], color="C0", marker = "o", label=r"$\substack{p=16 \\ \text{(two-curve test)}}$")
    axe.loglog(n_values, single_curve[2,:], linestyle = "--", marker = "s", color="C1", label=r"$\substack{p=32 \\ \text{(single-curve test)}}$")
    axe.loglog(n_values, two_curve[2,:], color="C1", marker = "o", label=r"$\substack{p=32 \\ \text{(two-curve test)}}$")
    axe.set_xlabel(r"$n$",fontsize=fontsize)
    
    # 理论参考线 y = 1/√N
    axe.plot(n_values,
                [2/np.sqrt(n) for n in n_values],
                color='green',
                linestyle=':',
                linewidth=2,
                alpha=0.7)
    x_mid = np.exp((np.log(n_values[0]) + np.log(n_values[-1])) / 2)
    y_mid = 2/np.sqrt(x_mid)
    axe.text(x_mid, y_mid,
        r'$1/\sqrt{n}$',
        fontsize=fontsize,
        color='green',
        alpha=0.8,
        verticalalignment='bottom',
        horizontalalignment='left')
    
    axe.set_ylabel('Rel. test error', fontsize=fontsize)
    axe.set_xscale('log')
    axe.set_yscale('log')
    axe.set_xlim(min(n_values) * 0.8, max(n_values) * 1.2)
    axe.set_xticks(n_values, n_values)
    axe.set_yticks([1e-2,5e-2,1e-1], [1e-2,5e-2,1e-1])
    axe.grid(True, linestyle=':', alpha=0.3)
    axe.legend(loc='center right',
            fontsize=fontsize,
            bbox_to_anchor=(1.75, 0.5),
            frameon=False,
            fancybox=False,
            shadow=False,
            borderpad=1,
            handletextpad=0.5,
            columnspacing=1.5)
    



    # 调整布局
    plt.tight_layout(rect=[0.05, 0.05, 0.4, 0.0])  # 为顶部图例留出空间
    
    # 保存图像
    plt.savefig('exterior_laplace_neumann_n_figure.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    
if __name__ == "__main__":
    generate_combined_figure()
    generate_exterior_laplacian_neumann_figure()