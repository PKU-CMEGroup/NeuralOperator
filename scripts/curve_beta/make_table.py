import os
import re
import ast

# Parameter definitions
kernel_names = ["dp_laplace", "sp_laplace"]
ranges = ["0.5-1.0", "0.5-10.5"]
layers_list = ["64,64_none","64,64_gelu", "64,64,64,64_gelu", "64,64,64,64,64,64_gelu"]
k_maxes = [8, 16, 32, 64]

# Layer display names for table rows
layer_display = {
    "64,64_none": "linear",
    "64,64_gelu": "2-layer",
    "64,64,64,64_gelu": "4-layer",
    "64,64,64,64,64,64_gelu": "6-layer",
}

def parse_last_line(filepath):
    """Parse the last line of a log file and return (val1, val2) rounded values.
    
    Supports two log formats:
      Format 1: Rel.Test L2: {'Single': ..., 'Two Curves': ...}
      Format 2: Rel. Test L2 Loss : {'Default': ..., 'Two Circles': ...}
    """
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, "r") as f:
        lines = f.readlines()
    
    if not lines:
        return None
    
    last_line = lines[-1].strip()
    
    # Check epoch is 499 — allow optional spaces around the colon
    epoch_match = re.search(r"Epoch\s*:\s*(\d+)", last_line)
    if not epoch_match or int(epoch_match.group(1)) != 499:
        return None
    
    # Match "Rel. Test L2 [Loss] :" with flexible spacing
    rel_test_match = re.search(
        r"Rel\.?\s*Test\s*L2(?:\s*Loss)?\s*:\s*(\{[^}]+\})",
        last_line
    )
    if not rel_test_match:
        return None
    
    try:
        rel_test_dict = ast.literal_eval(rel_test_match.group(1))
    except Exception:
        return None
    
    # Try both key naming conventions
    val1 = rel_test_dict.get("Single") or rel_test_dict.get("Default")
    val2 = rel_test_dict.get("Two Curves") or rel_test_dict.get("Two Circles")
    
    if val1 is None or val2 is None:
        return None
    
    return (round(val1 * 100, 3), round(val2 * 100, 3))


def format_cell(val):
    if val is None:
        return "N/A"
    return f"{val[0]:.3f},{val[1]:.3f}"


def build_table(kernel_name, rng):
    """Build a 3x4 markdown table for a given kernel_name and range."""
    # Header: k=8, k=16, k=32, k=64
    header = "| Layers | k=8 | k=16 | k=32 | k=64 |"
    separator = "|--------|-----|------|------|------|"
    rows = [header, separator]
    
    for layers in layers_list:
        cells = []
        for k in k_maxes:
            path = (
                f"log_lowrank/1_1_5_2d_{kernel_name}/beta{rng}/{layers}/"
                f"k{k}_L10_bsz8_factor20.0_gradTrue_geoTrue_geointTrue_beta{rng}.log"
            )
            val = parse_last_line(path)
            cells.append(format_cell(val))
        
        row = f"| {layer_display[layers]} | " + " | ".join(cells) + " |"
        rows.append(row)
    
    return "\n".join(rows)


def main():
    md_lines = ["# Log Results Summary\n",
                "Each cell shows `Single%, Two Curves%` (Rel.Test L2 × 100, rounded to 3 decimal places).\n"]
    
    for kernel_name in kernel_names:
        for rng in ranges:
            title = f"## kernel={kernel_name}, beta range={rng}"
            md_lines.append(title)
            md_lines.append("")
            table = build_table(kernel_name, rng)
            md_lines.append(table)
            md_lines.append("")
    
    output = "\n".join(md_lines)
    
    with open("results.md", "w") as f:
        f.write(output)
    
    print("Written to results.md")
    print()
    print(output)


if __name__ == "__main__":
    main()