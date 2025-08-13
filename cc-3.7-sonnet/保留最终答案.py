import re
import json
import os


def extract_last_content(text):
    """
    从文本中提取"最终答案："后的所有内容
    包括数字、文本、单位、短语等，并进行清理
    """
    if not isinstance(text, str):
        return ""

    match = re.search(r'最终答案：\s*([^\n]*(?:\n[^\n]*)*)', text)
    if match:
        result = match.group(1)
        # 清理结果：移除多余空格和换行
        result = re.sub(r'\s+', ' ', result).strip()
        # 特殊处理：移除可能存在的句号结尾
        result = re.sub(r'[。.]+$', '', result)
        return result
    return ""


def load_json(json_file):
    """加载JSON文件并返回解析后的数据"""
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"JSON文件不存在: {json_file}")

    with open(json_file, 'r', encoding='utf-8') as file:
        return json.load(file)


def save_json(data, output_file):
    """将数据保存为JSON文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def process_predictions(input_file, output_file):
    """
    处理预测结果：
    1. 加载原始预测数据
    2. 提取每个预测中的最终答案
    3. 保存处理后的结果
    """
    try:
        # 步骤1: 加载原始数据
        json_data = load_json(input_file)
        print(f"成功加载 {len(json_data)} 个图像的数据")

        # 步骤2: 处理每个图像的所有问题
        processed_count = 0
        for image_id, question_list in json_data.items():
            if not isinstance(question_list, list):
                continue

            for question in question_list:
                if isinstance(question, dict) and "pred" in question:
                    original_pred = question["pred"]
                    cleaned_pred = extract_last_content(original_pred)
                    question["pred"] = cleaned_pred
                    processed_count += 1

        # 步骤3: 保存结果
        save_json(json_data, output_file)
        print(f"处理完成，共更新 {processed_count} 条预测")
        print(f"结果已保存至: {output_file}")

    except Exception as e:
        print(f"处理过程中出错: {str(e)}")
        raise
if __name__ == "__main__":
    # 配置输入输出路径
    INPUT_JSON = "/home/jupyter-lpy/project/target_detection/CAD_VQA/cc-3.7-sonnet/output/p5_result/llm_pred.json"
    OUTPUT_JSON = "/home/jupyter-lpy/project/target_detection/CAD_VQA/cc-3.7-sonnet/output/p5_result/删除推理后的评估数据.json"

    # 执行处理流程
    process_predictions(INPUT_JSON, OUTPUT_JSON)