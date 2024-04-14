import json

# 定义Cityscapes数据集中定义的类别名称列表
def get_cs_classes():
    return [
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
        'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
        'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
        'bicycle'
    ]

# 解析文件内容并构建JSON对象
def parse_file(input_file_path):
    with open(input_file_path, 'r') as file:
        lines = file.readlines()
    
    parsed_data = []
    for line in lines:
        parts = line.strip().split(',')
        file_name = parts[0].strip('"')
        class_index = int(parts[1])
        class_name = get_cs_classes()[class_index]
        parsed_data.append({
            "class": str(class_index),  # 类别索引转换为字符串
            "class_name": class_name,
            "file_name": file_name
        })
    
    return parsed_data

# 将JSON对象写入到输出文件
def write_to_file(output_file_path, json_data):
    with open(output_file_path, 'w') as file:
        json.dump(json_data, file, indent=4)

# 主函数
def main():
    input_file_path = '/mnt/DGdataset/gta6000_rcs1e-2/rcs_sample_gta_files/rcs_images_6000.json'  # 输入文件路径
    output_file_path = '/mnt/DGdataset/gta6000_rcs1e-2/rcs_sample_gta_files/real_rcs_images_6000.json'  # 输出文件路径
    
    parsed_data = parse_file(input_file_path)
    write_to_file(output_file_path, parsed_data)

if __name__ == "__main__":
    main()