import re

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

def TABLE1_generate():
    # 表格1:固定layers=64,64,act=none,n_train=8000
    kernel_types = ["sp_laplace","dp_laplace","modified_dp_laplace","adjoint_dp_laplace","stokes"]
    formal_names = {"sp_laplace":"Laplacian single \\\\layer potential",
                    "dp_laplace":"Laplacian double \\\\layer potential",
                    "modified_dp_laplace":"Modified Laplacian \\\\double layer potential",
                    "adjoint_dp_laplace":"Adjoint Laplacian \\\\double layer potential",
                    "stokes":"Stokeslet"}
    k_max_values = [8,16,32,64]
    n_train = 8000
    layers = "64,64"
    act = "none"

    string0 = "\\begin{table}[htbp]\n\t\\begin{center}\n\t\t\\begin{tabular}{|c|c|c|c|c|}\n\t\t\t\\hline\n\t\t\t\\diagbox{Kernel}{$p$}  & $8 $ & $16$ & $32$ & $64$\\\\ \\hline\n"
    string_list = [string0]
    for kernel in kernel_types:
        row_string = f"\t\t\t\\makecell{{{formal_names[kernel]}}}  "
        for k_max in k_max_values:
            result = get_the_loss_from_log(kernel_type=kernel, k_max=k_max, layers=layers, act=act, n_train=n_train)
            if result:
                default_scaled, two_circles_scaled = result
                row_string += f" & {default_scaled}\\quad {two_circles_scaled}  "
            else:
                row_string += " & XXX\\quad XXX  "
        row_string += " \\\\ \\hline\n"
        string_list.append(row_string)
    string_list.append("\t\t\\end{tabular}\n\t\\end{center}\n")
    string_list.append(f"\t\\caption{{Linear train on 8000:  Relative $L_2$ error ($\\times 10^{{-2}}$) between the reference and predicted results, with single circle and two circles.}}\n")
    string_list.append("\\end{table}\n")

    return ''.join(string_list)

def TABLE2_generate():
    # 表格2:固定layers="64,64,64,64,64,64",act=gelu,n_train=8000
    kernel_types = ["sp_laplace","dp_laplace","modified_dp_laplace","adjoint_dp_laplace","stokes"]
    formal_names = {"sp_laplace":"Laplacian single \\\\layer potential",
                    "dp_laplace":"Laplacian double \\\\layer potential",
                    "modified_dp_laplace":"Modified Laplacian \\\\double layer potential",
                    "adjoint_dp_laplace":"Adjoint Laplacian \\\\double layer potential",
                    "stokes":"Stokeslet"}
    k_max_values = [8,16,32,64]
    n_train = 8000
    layers = "64,64,64,64,64,64"
    act = "gelu"

    string0 = "\\begin{table}[htbp]\n\t\\begin{center}\n\t\t\\begin{tabular}{|c|c|c|c|c|}\n\t\t\t\\hline\n\t\t\t\\diagbox{Kernel}{$p$}  & $8 $ & $16$ & $32$ & $64$\\\\ \\hline\n"
    string_list = [string0]
    for kernel in kernel_types:
        row_string = f"\t\t\t\\makecell{{{formal_names[kernel]}}}  "
        for k_max in k_max_values:
            result = get_the_loss_from_log(kernel_type=kernel, k_max=k_max, layers=layers, act=act, n_train=n_train)
            if result:
                default_scaled, two_circles_scaled = result
                row_string += f" & {default_scaled}\\quad {two_circles_scaled}  "
            else:
                row_string += " & XXX\\quad XXX  "
        row_string += " \\\\ \\hline\n"
        string_list.append(row_string)
    string_list.append("\t\t\\end{tabular}\n\t\\end{center}\n")
    string_list.append(f"\t\\caption{{5 layer train on 8000:  Relative $L_2$ error ($\\times 10^{{-2}}$) between the reference and predicted results, with single circle and two circles.}}\n")
    string_list.append("\\end{table}\n")

    return ''.join(string_list)

def TABLE3_generate():
    # 表格3:固定k_max=32,layers="64,64,64,64,64,64",act=gelu
    kernel_types = ["sp_laplace","dp_laplace","modified_dp_laplace","adjoint_dp_laplace","stokes"]
    formal_names = {"sp_laplace":"Laplacian single \\\\layer potential",
                    "dp_laplace":"Laplacian double \\\\layer potential",
                    "modified_dp_laplace":"Modified Laplacian \\\\double layer potential",
                    "adjoint_dp_laplace":"Adjoint Laplacian \\\\double layer potential",
                    "stokes":"Stokeslet"}
    n_train_values = [1000,2000,4000,8000]
    k_max = 32
    layers = "64,64,64,64,64,64"
    act = "gelu"
    string0 = "\\begin{table}[htbp]\n\t\\begin{center}\n\t\t\\begin{tabular}{|c|c|c|c|c|}\n\t\t\t\\hline\n\t\t\t\\diagbox{Kernel}{$N$}  & $1000$ & $2000$ & $4000$ & $8000$\\\\ \\hline\n"
    string_list = [string0]
    for kernel in kernel_types:
        row_string = f"\t\t\t\\makecell{{{formal_names[kernel]}}}  "
        for n_train in n_train_values:
            result = get_the_loss_from_log(kernel_type=kernel, k_max=k_max, layers=layers, act=act, n_train=n_train)
            if result:
                default_scaled, two_circles_scaled = result
                row_string += f" & {default_scaled}\\quad {two_circles_scaled}  "
            else:
                row_string += " & XXX\\quad XXX  "
        row_string += " \\\\ \\hline\n"
        string_list.append(row_string)
    string_list.append("\t\t\\end{tabular}\n\t\\end{center}\n")
    string_list.append(f"\t\\caption{{5 layer $p = 32$:  Relative $L_2$ error ($\\times 10^{{-2}}$) between the reference and predicted results, with single circle and two circles.}}\n")
    string_list.append("\\end{table}\n")
    return ''.join(string_list)


with open("table.txt", "w") as f:
    lines = TABLE1_generate()
    f.write(lines)
    f.write("\n\n")
    lines = TABLE2_generate()
    f.write(lines)
    f.write("\n\n")
    lines = TABLE3_generate()
    f.write(lines)
    f.write("\n\n")