import json
import os


def merge_json_files(file_paths, output_path):
    """
    合并多个JSON文件到一个文件

    参数:
    file_paths: 包含完整路径的文件名列表，如 ['/data/train.json', '/other/val.json', '/test.json']
    output_path: 合并后文件的输出路径
    """
    merged_data = {}

    for file_path in file_paths:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"警告: 文件 {file_path} 不存在，跳过")
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 合并数据
                for key, value in data.items():
                    # 如果key已存在，则合并问题列表
                    if key in merged_data:
                        merged_data[key].extend(value)
                    else:
                        merged_data[key] = value
                print(f"已加载 {file_path}，包含 {len(data)} 个图像ID")
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")

    # 保存合并后的数据
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)

    print(f"\n合并完成！共合并 {len(merged_data)} 个图像ID")
    print(f"输出文件已保存至: {os.path.abspath(output_path)}")


# 使用示例
if __name__ == "__main__":
    # 替换为你的实际文件路径
    files_to_merge = [
        "/home/jupyter-lpy/project/target_detection/CAD_VQA/allDate_vqa/qwen/output/train/llm_pred.json",  # 替换为train.json的实际路径
        "/home/jupyter-lpy/project/target_detection/CAD_VQA/allDate_vqa/qwen/output/val/llm_pred.json",  # 替换为val.json的实际路径
        "/home/jupyter-lpy/project/target_detection/CAD_VQA/allDate_vqa/qwen/output/test/llm_pred.json"  # 替换为test.json的实际路径
    ]

    output_file = '/home/jupyter-lpy/project/target_detection/CAD_VQA/allDate_vqa/qwen/merged.json'  # 替换为你想要的输出路径

    merge_json_files(files_to_merge, output_file)