import re

input_file = "colormap_raw.txt"
output_file = "colormap_converted.py"

with open(input_file, 'r') as f:
    lines = f.readlines()

colors = []
for line in lines:
    # 去掉注释和空白，只保留数字部分
    line = line.split('%')[0].strip()
    if not line:
        continue
    # 用正则提取3个数字
    match = re.findall(r'\d+', line)
    if len(match) == 3:
        r, g, b = map(int, match)
        colors.append([r/255, g/255, b/255])

with open(output_file, 'w') as f:
    f.write("import numpy as np\n\n")
    f.write("cMap = np.array([\n")
    for c in colors:
        f.write(f"    [{c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}],\n")
    f.write("])\n")

print(f"转换完成，结果保存到 {output_file}")