import json
import os

# 假设原始JSON文件名为 'original_data.json'
input_json_file = 'real_rcs_images_6000.json'

# 读取原始JSON文件
with open(input_json_file, 'r') as file:
    data = json.load(file)

# 确保读取的数据是一个列表
if not isinstance(data, list):
    raise ValueError("JSON文件应包含一个列表")

# 分割数据的逻辑
num_segments = 4 # 分成4个文件
segment_size = len(data) // num_segments
remaining_elements = len(data) % num_segments

# 分割数据并保存到新文件
for i in range(num_segments):
    start_index = i * segment_size
    end_index = start_index + segment_size
    segment= data[start_index:end_index]
    output_file_name = f'segment_{i + 1}.json'
    with open(output_file_name, 'w') as output_file:
        json.dump(segment, output_file, indent=4)

    print(f'文件 {output_file_name} 已保存。')

print("所有分割后的JSON文件已保存。")