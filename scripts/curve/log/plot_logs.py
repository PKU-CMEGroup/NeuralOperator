import os
import re
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.ticker as ticker

def parse_log_file_full(filepath):
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            for line in reversed(lines):
                if 'Rel. Test L2 Loss' in line:
                    # 格式示例: Epoch :  0  Time: ... Rel. Train L2 Loss : ... Rel. Test L2 Loss :  0.3370 ...
                    try:
                        # Extract Epoch
                        epoch_part = line.split('Epoch :')[1].split('Time:')[0].strip()
                        epoch = int(epoch_part)
                        
                        # Extract Train Loss
                        train_part = line.split('Rel. Train L2 Loss :')[1].split('Rel. Test L2 Loss :')[0].strip()
                        train_loss = float(train_part)
                        
                        # Extract Test Loss
                        test_part = line.split('Rel. Test L2 Loss :')[1].split('Test L2 Loss :')[0].strip()
                        test_loss = float(test_part)
                        
                        return epoch, train_loss, test_loss
                    except (ValueError, IndexError, AttributeError):
                        continue
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None
    return None

def save_results_to_md(log_dir, output_file, record_previous=False):
    print(f"Generating {output_file} from {log_dir}...")
    with open(output_file, 'w') as md:
        if not os.path.exists(log_dir):
            return

        for root, dirs, files in os.walk(log_dir):
            # Filter dirs in-place to prevent walking into them and ensure order
            if record_previous:
                dirs[:] = [d for d in dirs if not d.startswith('.')]
            else:
                dirs[:] = [d for d in dirs if not d.startswith('__') and not d.startswith('.')]
            dirs.sort()
            files.sort()
            
            # Filter log files
            log_files = [f for f in files if f.endswith('.log')]
            
            rel_path = os.path.relpath(root, log_dir)
            if rel_path == '.':
                level = 0
            else:
                level = len(rel_path.split(os.sep))
            
            # Write header if level > 0
            # Always write header to maintain directory structure context for subdirectories
            if level > 0:
                dir_name = os.path.basename(root)
                md.write(f"{'#' * level} {dir_name}\n")
            
            # Calculate max length for alignment
            file_infos = []
            max_len = 0
            for filename in log_files:
                filepath = os.path.join(root, filename)
                info = parse_log_file_full(filepath)
                if info:
                    file_infos.append((filename, info))
                    if len(filename) > max_len:
                        max_len = len(filename)
            
            for filename, (epoch, train_loss, test_loss) in file_infos:
                md.write(f"- {filename:<{max_len}}: ep {epoch:>4}, train {train_loss:.8f}, test {test_loss:.8f}\n")
            
            # Add newline only if we wrote something (header or files)
            if level > 0 or file_infos:
                md.write("\n")
    print('\n########################################')
    print(f"Finished generating {output_file}")
    print('########################################\n')

def load_results_from_md(md_file):
    data = {}
    path_stack = []
    if not os.path.exists(md_file): return data
        
    with open(md_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            if line.startswith('#'):
                level = len(line) - len(line.lstrip('#'))
                dirname = line[level:].strip()
                if level == 1: path_stack = []
                else: path_stack = path_stack[:level-1]
                path_stack.append(dirname)
                
                curr = data
                for key in path_stack:
                    curr = curr.setdefault(key, {})
                    
            elif line.startswith('- ') and path_stack:

                parts = line[2:].split(': ')
                filename = parts[0].strip()
                vals = [v.strip() for v in ': '.join(parts[1:]).split(',')]
                
                curr = data
                for key in path_stack: curr = curr[key]
                curr[filename] = (int(vals[0].split()[-1]), float(vals[1].split()[-1]), float(vals[2].split()[-1]))

    return data

def parse_model_info(filename, required_keywords=None, excluded_keywords=None):
    """
    通用的文件名解析:
    - required_keywords: 必须包含的关键词列表
    - excluded_keywords: 必须不包含的关键词列表
    返回 (n_data, model_type, k, other_infos) 或 None
    """
    if required_keywords is None:
        required_keywords = []
    if excluded_keywords is None:
        excluded_keywords = []

    n_data = None
    k = None
    other_infos = []
    flags = {'geo': False, 'geograd': False, 'grad': True}

    filename = filename.replace('.log', '')
    all_infos = filename.split('_')  
    
    # Check required keywords
    for keyword in required_keywords:
        if keyword not in all_infos:
            return None
            
    # Check excluded keywords
    for keyword in excluded_keywords:
        if keyword in all_infos:
            return None
        
    for info in all_infos:
        if info.startswith('n') and info[1:].isdigit():
            n_data = int(info[1:])
        elif info.startswith('k') and info[1:].isdigit():
            k = int(info[1:])
        elif info == 'nograd':
            flags['grad'] = False
        elif info == 'geograd':
            flags['geograd'] = True
        elif info in ['geo3wx', 'geo2wx']:
            flags['geo'] = True
        elif info in required_keywords:
            continue
        elif info.startswith('L'):
            # 暂时不考虑L的信息
            continue
        elif info == 'layer2' or info == 'noact':
            continue
        else:
            other_infos.append(info)

    prefix = ' + '.join(sorted(k for k, v in flags.items() if v)) or 'base'
    model_type = '_'.join([prefix] + other_infos)

    return n_data, model_type, k


def dominant_model_type(model_type):
    if model_type.startswith('geo + grad'):
        return 'geo + grad'
    elif model_type.startswith('grad_') or model_type == 'grad':
        return 'grad'
    elif model_type.startswith('geo_') or model_type == 'geo':
        return 'geo'
    elif model_type.startswith('base_') or model_type == 'base':
        return 'base'
    elif model_type.startswith('geograd_') or model_type == 'geograd':
        return 'geograd'
    elif model_type.startswith('geo + geograd'):
        return 'geo + geograd'
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def update_log_summary(log_dir=None, md_filename='collected_log_results.md', record_previous=False):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if log_dir is None:
        log_dir = script_dir
    
    md_file = os.path.join(script_dir, md_filename)
    save_results_to_md(log_dir, md_file, record_previous=record_previous)

def plot_curves(kernel_type, layer_type, normal_keywords, special_keywords, ignore_keywords, plot_normal, plot_special_with_normal, plot_special, plot_axis='n_train', md_filename='collected_log_results.md', output_dir=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    md_data = load_results_from_md(os.path.join(script_dir, md_filename))
    output_dir = output_dir or script_dir

    # Simplify data finding
    target_key = f'1_1_5_2d_{kernel_type}_panel'
    target_data = md_data.get(target_key, {}).get(layer_type)
    
    if not target_data:
        for k, v in md_data.items():
            if kernel_type in k:
                if isinstance(v, dict) and layer_type in v:
                    target_data = v[layer_type]; break
    
    if not target_data:
        print(f"No data found for {kernel_type}/{layer_type}"); return

    print(f"Processing {len(target_data)} files for {kernel_type}/{layer_type}...")

    def get_data(required_keys, excluded_keys, label):
        results = {}
        count = 0
        for filename, (_, _, loss) in target_data.items():
            if not filename.endswith('.log'): continue
            info = parse_model_info(filename, required_keys, excluded_keys)
            if not info: continue
            
            count += 1
            n_data, model_type, k_value = info
            group_key, x_val = (k_value, n_data) if plot_axis == 'n_train' else (n_data, k_value)
            
            if model_type not in results.setdefault(group_key, {}):
                results[group_key][model_type] = {}
            
            current_results = results[group_key][model_type]
            if x_val not in current_results or loss < current_results[x_val]:
                current_results[x_val] = loss
        print(f"Processed {count} {label} files (keywords: {required_keys}, ignored: {excluded_keys})")
        return results

    normal_results = get_data(normal_keywords, special_keywords + ignore_keywords, 'normal')
    special_results = {}
    if special_keywords:
        special_results = get_data(normal_keywords + special_keywords, ignore_keywords, 'special')

    styles = {
        'geo + grad': {'marker': 'o', 'color': 'red'},
        'grad': {'marker': 's', 'color': 'blue'},
        'geo': {'marker': '^', 'color': 'green'},
        'base': {'marker': 'x', 'color': 'orange'},
        'geograd': {'marker': 'D', 'color': 'purple'},
        'geo + geograd': {'marker': 'v', 'color': 'brown'}
    }

    def plot_group(data, overlay=None, suffix=''):
        for group_key, models in data.items():
            plt.figure(figsize=(10, 8))
            
            def draw_lines(model_dict, linestyle='-', alpha=1.0, label_suffix=''):
                # Sort by length to ensure base model comes first
                sorted_models = sorted(model_dict.items(), key=lambda x: (len(x[0]), x[0]))
                seen_types = {}

                for model_type, points in sorted_models:
                    dtype = dominant_model_type(model_type)
                    seen_types[dtype] = seen_types.get(dtype, 0) + 1
                    style = styles.get(dtype, {'marker': 'o', 'color': 'black'})
                    
                    ls = linestyle
                    if linestyle == '-':
                        # Cycle styles for variants: Solid -> Dash-dot -> Dotted
                        ls = ['-', '-.', ':'][ (seen_types[dtype]-1) % 3 ]

                    x_vals, y_vals = zip(*sorted(points.items()))
                    plt.plot(x_vals, y_vals, marker=style['marker'], color=style['color'], 
                             label=model_type + label_suffix, linestyle=ls, 
                             linewidth=1.5, markersize=4, alpha=alpha)

            draw_lines(models)
            if overlay and group_key in overlay:
                draw_lines(overlay[group_key], linestyle='--', alpha=0.7, label_suffix=f' ({"+".join(special_keywords)})')

            xlabel = 'Data Size (N)' if plot_axis == 'n_train' else 'k_max'
            group_label = 'k' if plot_axis == 'n_train' else 'N'
            
            plt.xlabel(xlabel)
            plt.ylabel('Rel. Test L2 Loss')
            plt.yscale('log')
            plt.title(f'Model Performance vs {xlabel} ({kernel_type}, {group_label}={group_key})')
            plt.legend()
            plt.grid(True, which="both", alpha=0.5)

            ax = plt.gca()
            ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10), numticks=20))
            formatter = ticker.ScalarFormatter()
            formatter.set_scientific(False)
            ax.yaxis.set_major_formatter(formatter)
            ax.yaxis.set_minor_formatter(formatter)
            
            output_filename = f"{kernel_type}_{layer_type}_{group_label}{group_key}{suffix}.png"
            plt.savefig(os.path.join(output_dir, output_filename), dpi=300)
            plt.close()
            print(f"Plot saved to {output_filename}")

    if plot_normal:
        normal_tag = "_".join(normal_keywords)
        suffix = f"_{normal_tag}"
        overlay = None
        if plot_special_with_normal and special_results:
            special_tag = "_".join(special_keywords)
            suffix += f"({special_tag})"
            overlay = special_results
        if ignore_keywords:
            suffix += f"_no_{'_'.join(ignore_keywords)}"
        plot_group(normal_results, overlay, suffix=suffix)

    if plot_special and special_results:
        suffix = f"_{'_'.join(normal_keywords + special_keywords)}"
        if ignore_keywords:
            suffix += f"_no_{'_'.join(ignore_keywords)}"
        plot_group(special_results, suffix=suffix)


if __name__ == '__main__':
    kernel_type = "stokes"  # dp_laplace or sp_laplace or stokes or modified_dp
    layer_type = 'layer2_gelu' # gelu or layer2_gelu or noact

    normal_keywords = ['deep']
    special_keywords = []
    ignore_keywords = []
    plot_normal = True
    plot_special_with_normal = True  
    plot_special = False  
    plot_axis = 'n_train' if layer_type in ['gelu','layer2_gelu'] else 'k_max' # n_train or k_max
    
    md_filename = 'collected_log_results.md'
    record_previous = False

    update_log_summary(md_filename=md_filename, record_previous=record_previous)
    plot_curves(kernel_type, layer_type,
                 normal_keywords, special_keywords, ignore_keywords,
                 plot_normal, plot_special_with_normal, plot_special, plot_axis,
                   md_filename=md_filename,
                   output_dir='figures/')