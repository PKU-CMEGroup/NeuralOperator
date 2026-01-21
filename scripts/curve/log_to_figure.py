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
    log_dir = f"log/log_old/{prefix}_{kernel_type}/{layers}_{act}/"
    log_file_path = (f"{log_dir}N{n_train}_Ntest{n_test},{n_two_circles_test}_k{k_max}_L10_"
                     f"bsz{batch_size}_factor{to_divide_factor}_grad{grad}_geo{geo}_geoint{geoint}.log")
    
    return extract_test_losses_with_epochcheck(log_file_path)

def Figure1_generate():
    kernel_types = ["sp_laplace","dp_laplace","modified_dp_laplace","adjoint_dp_laplace","stokes"]
    formal_names = {"sp_laplace":"Laplacian single layer potential",
                    "dp_laplace":"Laplacian double layer potential",
                    "modified_dp_laplace":"Modified Laplacian double layer potential",
                    "adjoint_dp_laplace":"Adjoint Laplacian double layer potential",
                    "stokes":"Stokeslet"}
    k_max_values = [8,16,32,64]
    n_train = 8000
    
    # 创建1行5列的子图
    fig, axes = plt.subplots(1, 5, figsize=(22, 5.5))
    
    # 定义线条样式
    styles = {
        "1L-S": {"color": "blue", "ls": "-", "marker": "o", "label": "1L-S"},
        "1L-D": {"color": "blue", "ls": "--", "marker": "s", "label": "1L-D"},
        "5L-S": {"color": "red", "ls": "-", "marker": "o", "label": "5L-S"},
        "5L-D": {"color": "red", "ls": "--", "marker": "s", "label": "5L-D"},
    }
    configs = [
        {"layers": "64,64", "act": "none", "name": "1L"},
        {"layers": "64,64,64,64,64,64", "act": "gelu", "name": "5L"}
    ]
    
    # 创建自定义图例句柄 - 只显示线条，不带标记
    from matplotlib.lines import Line2D
    
    # 预先创建图例句柄（只包含线条，不带标记）
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, linestyle='-'),  # 蓝色实线
        Line2D([0], [0], color='blue', lw=2, linestyle='--'), # 蓝色虚线
        Line2D([0], [0], color='red', lw=2, linestyle='-'),   # 红色实线
        Line2D([0], [0], color='red', lw=2, linestyle='--'),  # 红色虚线
    ]
    
    # 图例标签
    legend_labels = ['1L-S', '1L-D', '5L-S', '5L-D']
    
    for ax_idx, kernel in enumerate(kernel_types):
        ax = axes[ax_idx]
        
        # 收集所有配置的数据
        data = {}
        for config in configs:
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
        
        # 绘制四条线（仍然保留标记，只是图例中不显示）
        for line_name, line_style in styles.items():
            values = data.get(line_name, [])
            if any(values):
                ax.plot([k for k in k_max_values],
                        [v if v else np.nan for v in values], color=line_style["color"],
                        linestyle=line_style["ls"],
                        linewidth=2,
                        marker=line_style["marker"],
                        markersize=6)

                # 设置log-log坐标
                ax.set_xscale('log')
                ax.set_yscale('log')

                # 设置横坐标范围（关键步骤！）
                ax.set_xlim(min(k_max_values) * 0.8, max(k_max_values) * 1.2)

                # 设置横坐标刻度（现在只会显示我们指定的刻度）
                ax.set_xticks(k_max_values, k_max_values)
        
        # 理论参考线 y = 5 - x
        ax.plot(k_max_values,
                [100/k for k in k_max_values],
                color='green',
                linestyle=':',
                linewidth=2,
                alpha=0.7)
        
        # 在每个子图的参考线旁边添加"e^5/p"标注
        x_mid = np.exp((np.log(k_max_values[0]) + np.log(k_max_values[-1])) / 2)
        y_mid = 100/x_mid
        
        # 添加文本标注
        ax.text(x_mid, y_mid,
                r'$1/p$',
                fontsize=9,
                color='green',
                alpha=0.8,
                verticalalignment='bottom',
                horizontalalignment='left')
        
        ax.set_xlabel('')
        ax.tick_params(axis='x', labelleft=False)
        
        # 设置子图标题和坐标轴
        ax.set_title(formal_names[kernel], fontsize=8, pad=8, fontweight='bold')
        ax.set_xlabel('p', fontsize=11)
        
        # 只在第一个子图显示完整的y轴
        if ax_idx == 0:
            ax.set_ylabel('log(Error × 10²)', fontsize=11)
        else:
            ax.set_ylabel('')
            ax.tick_params(axis='y', labelleft=False)
        
        # 添加网格
        ax.grid(True, linestyle=':', alpha=0.3)
    
    # 在整个图形顶部添加统一图例（只显示线条）
    fig.legend(legend_elements, legend_labels,
               loc='upper center',
               ncol=len(legend_elements),
               fontsize=11,
               bbox_to_anchor=(0.5, 0.95),
               frameon=True,
               fancybox=True,
               shadow=False,
               borderpad=1,
               handletextpad=0.5,
               columnspacing=1.5)
    
    # 添加总标题
    plt.suptitle('Error Convergence with Varying Truncation Order p', 
                 fontsize=14, y=0.98)
    
    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    
    # 保存图像
    plt.savefig('figure1.png', dpi=300, bbox_inches='tight')
    plt.show()

def Figure2_generate():
    kernel_types = ["sp_laplace","dp_laplace","modified_dp_laplace","adjoint_dp_laplace","stokes"]
    formal_names = {"sp_laplace":"Laplacian single layer potential",
                    "dp_laplace":"Laplacian double layer potential",
                    "modified_dp_laplace":"Modified Laplacian double layer potential",
                    "adjoint_dp_laplace":"Adjoint Laplacian double layer potential",
                    "stokes":"Stokeslet"}
    k_max = 32
    n_values_list = [1000,2000,4000,8000]
    
    # 创建1行5列的子图
    fig, axes = plt.subplots(1, 5, figsize=(22, 5.5))
    
    # 定义线条样式
    styles = {
        "5L-S": {"color": "red", "ls": "-", "marker": "o", "label": "5L-S"},
        "5L-D": {"color": "red", "ls": "--", "marker": "s", "label": "5L-D"},
    }
    configs = [
        {"layers": "64,64,64,64,64,64", "act": "gelu", "name": "5L"}
    ]
    
    # 创建自定义图例句柄 - 只显示线条，不带标记
    from matplotlib.lines import Line2D
    
    # 预先创建图例句柄（只包含线条，不带标记）
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, linestyle='-'),   # 红色实线
        Line2D([0], [0], color='red', lw=2, linestyle='--'),  # 红色虚线
    ]
    
    # 图例标签
    legend_labels = ['5L-S', '5L-D']
    
    for ax_idx, kernel in enumerate(kernel_types):
        ax = axes[ax_idx]
        
        # 收集所有配置的数据
        data = {}
        for config in configs:
            layers = config["layers"]
            act = config["act"]
            config_name = config["name"]
            
            singles, doubles = [], []
            for n_train in n_values_list:
                result = get_the_loss_from_log(
                    kernel_type=kernel, 
                    k_max=k_max, 
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
        
        # 绘制四条线（仍然保留标记，只是图例中不显示）
        for line_name, line_style in styles.items():
            values = data.get(line_name, [])
            if any(values):
                ax.plot(n_values_list,
                        values,
                        color=line_style["color"],
                        linestyle=line_style["ls"],
                        linewidth=2,
                        marker=line_style["marker"],
                        markersize=6)
                # 设置log-log坐标
                ax.set_xscale('log')
                ax.set_yscale('log')

                # 设置横坐标范围（关键步骤！）
                ax.set_xlim(min(n_values_list) * 0.8, max(n_values_list) * 1.2)

                # 设置横坐标刻度（现在只会显示我们指定的刻度）
                ax.set_xticks(n_values_list, n_values_list)
        
        # 理论参考线
        ax.plot(n_values_list,
                [100/np.sqrt(N) for N in n_values_list],
                color='green',
                linestyle=':',
                linewidth=2,
                alpha=0.7)
        
        # 在每个子图的参考线旁边添加"e^5/p"标注
        x_mid = np.exp((np.log(n_values_list[0]) + np.log(n_values_list[-1])) / 2)
        y_mid = 100/np.sqrt(x_mid)
        
        # 添加文本标注
        ax.text(x_mid, y_mid,
                r'$1/\sqrt{N}$',
                fontsize=9,
                color='green',
                alpha=0.8,
                verticalalignment='bottom',
                horizontalalignment='left')
        
        # 设置子图标题和坐标轴
        ax.set_title(formal_names[kernel], fontsize=8, pad=8, fontweight='bold')
        ax.set_xlabel('log(N)', fontsize=11)
        
        # 只在第一个子图显示完整的y轴
        if ax_idx == 0:
            ax.set_ylabel('log(Error × 10²)', fontsize=11)
        else:
            ax.set_ylabel('')
            ax.tick_params(axis='y', labelleft=False)
        
        # 添加网格
        ax.grid(True, linestyle=':', alpha=0.3)
    
    # 在整个图形顶部添加统一图例（只显示线条）
    fig.legend(legend_elements, legend_labels,
               loc='upper center',
               ncol=len(legend_elements),
               fontsize=11,
               bbox_to_anchor=(0.5, 0.95),
               frameon=True,
               fancybox=True,
               shadow=False,
               borderpad=1,
               handletextpad=0.5,
               columnspacing=1.5)
    
    # 添加总标题
    plt.suptitle('Error Convergence with Varying Data Amount N', 
                 fontsize=14, y=0.98)
    
    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    
    # 保存图像
    plt.savefig('figure2.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    Figure1_generate()
    Figure2_generate()