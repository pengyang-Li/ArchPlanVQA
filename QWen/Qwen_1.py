import os
import argparse
import json
from PIL import Image
import io
import base64
from openai import OpenAI
import random
from tqdm import tqdm  # 导入进度条库
# 导入必要的库
import time               # 用于控制请求间隔（避免速率限制）
def parse_args():
    '''
    Arguments
    '''
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=str, default="/home/jupyter-lpy/project/target_detection/SymPoint_main/dataset_FloorplanCAD/test/png/",
                        help='save the downloaded data')
    parser.add_argument('--cad_qa', type=str, default="/home/jupyter-lpy/project/target_detection/CAD_VQA/data/small_dataset.json",
                        help='save the downloaded data')
    parser.add_argument('--api_key', type=str, default="sk-ljbdphromcnxchfgbcvaxvlcslukcaasqmqafddompcgxeun",
                        help='Path to save the results')
    parser.add_argument('--output', type=str, default="/home/jupyter-lpy/project/target_detection/CAD_VQA/QWen/output/qwen_result1/",
                        help='Path to save the results')
    args = parser.parse_args()
    return args
def r_json(json_file):
    # 打开JSON文件并读取数据
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)  # 返回Python字典或列表
    return data
def convert_image_to_webp_base64(input_image_path):
    try:
        with Image.open(input_image_path) as img:
            byte_arr = io.BytesIO()
            img.save(byte_arr, format='webp')
            byte_arr = byte_arr.getvalue()
            base64_str = base64.b64encode(byte_arr).decode('utf-8')
            return base64_str
    except IOError:
        print(f"Error: Unable to open or convert the image {input_image_path}")
        return None


def get_full_response_from_image(client, system_prompt, base64_image, que):
    """
    获取图像处理后的完整响应。

    :param client: 用于调用模型的客户端对象。
    :param system_prompt: 系统角色提示信息。
    :param base64_image: 图像的 base64 编码字符串。
    :return: 完整的响应字符串。
    """
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-VL-72B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},  # 系统角色：设定任务规则
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/webp;base64,{base64_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": que
                    }
                ]
            }
        ],

        temperature=0.3,
        top_p=0.9,
        max_tokens=512,
        frequency_penalty=0.2

    )

    # 直接返回完整的响应内容
    return response.choices[0].message.content

if __name__=="__main__":
    args = parse_args()
    folder_path =args.dataset
    cad_qa = r_json(args.cad_qa)
    data_dict = {'number': {'real': [], 'pred': []},
            'bool': {'real': [], 'pred': []},
            'string': {'real': [], 'pred': []}
            }
    qwen_data = {}
    client = OpenAI(
        api_key=args.api_key,  # 替换为你的实际API密钥
        base_url="https://api.siliconflow.cn/v1"  # SiliconFlow的API端点
    )
    system_prompt = """
**角色提示**
你是一个专业的建筑平面CAD图纸审查员。
**任务描述**
你的任务是根据用户提供的建筑平面CAD图像及建筑领域专业知识回答问题。
**返回要求**
1.最终答案严格按照问题类型的回答要求进行回答，禁止添加标点及其他无关上下文（如：‘答案是’）
2.如果不知道问题答案，直接返回无法回答。

    """
    # 遍历文件夹
    image_id_list = []
    for image_id in cad_qa:
        image_id_list.append(image_id)
    for image_id in tqdm(image_id_list, desc="Processing images"):
        time.sleep(3)
        filename = f"{image_id}.png"  # Reconstruct filename
        image_path = os.path.join(folder_path, filename)
        base64_image = convert_image_to_webp_base64(image_path)
        qwen_data[image_id] = []
        ans_id = cad_qa[image_id]["ans_id"]
        ans_type = cad_qa[image_id]["ans_type"]
        que = cad_qa[image_id]["que"]
        ans = cad_qa[image_id]["ans"]
        data_dict[ans_type]["real"].append(ans)
        pred = get_full_response_from_image(client, system_prompt, base64_image, que)
        qwen_data[image_id].append({
            "question": que,
            "real": ans,
            "pred": pred,
            "type": ans_type
        })
        data_dict[ans_type]["pred"].append(pred)


    ev_path = os.path.join(args.output, "evaluate_data.json")
    pred_path = os.path.join(args.output, "llm_pred.json")
    with open(ev_path, 'w', encoding='utf-8') as json_file:
        json.dump(data_dict, json_file, ensure_ascii=False, indent=4)
    with open(pred_path, 'w', encoding='utf-8') as json_file:
        json.dump(qwen_data, json_file, ensure_ascii=False, indent=4)
    print(f"数据已成功写入 {ev_path}")
    print(f"预测答案已成功写入 {pred_path}")



