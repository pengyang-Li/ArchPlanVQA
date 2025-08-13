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
    parser.add_argument('--output', type=str, default="/home/jupyter-lpy/project/target_detection/CAD_VQA/QWen/output/qwen_result4/",
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
在回答具体问题前，请先快速浏览图像，分析图纸中的元素以及空间位置关系，建立对图纸的整体认知，然后根据用户提供的建筑平面CAD图像内容及建筑领域专业知识回答问题。
**系统提示**
1. 建筑平面CAD图像左上角为坐标原点，坐标系长宽各100，对应实际10m×10m的空间。
2.问题有9个类型。
3.问题类型及回答方式定义如下：
(1)数量问题
   答案要求：直接返回整数数字
	   示例：{问题：CAD图像中有几个双扇门？: 答案：5}
(2).包含‘是否’或‘有无’的问题  
   答案要求：回答`是`或`否`、`有`或`无`。
示例：{问题: CAD图像中是否标注了浴缸？答案：是}
(3)面积问题
   答案要求：数值+平方米，如 `20平方米` 
示例：{问题: CAD图像中靠近左下角的电梯的实际空间面积是多少？ 答案：4平方米}
(4)长宽问题
答案要求：数值+单位*数值+单位，如‘多少m*多少m’
示例：{问题: CAD图像中靠近左下角的浴室的实际空间长宽是多少？答案：0.9m*0.9m}
(5)询问载重参数问题
答案要求：数值+kg，如`1000kg`
示例：{问题: CAD图像中靠近右上角的电梯的载重参数是多少？答案：1000kg}
(6)询问“合理型”或“符合性”的问题
   答案要求：只回答`合理`或`不合理`、`符合`或`不符合`。
示例：{问题: 卧室的床与衣柜间距合理吗(建议≥0.6m)？答案：合理}
(7)建筑功能类型问题
答案要求：用简短词语回答
示例：{问题: 该空间属于哪种建筑功能类型？答案：住宅}
(8) 询问特定场景有哪些配置的问题
答案要求：用‘、’分隔短语
示例：{问题: CAD图像中的厨房配置了哪些厨房用具？答案：水槽、煤气炉}
(9) 询问规格参数问题
答案要求：直接给出规格参数，如“FM1022”
示例：{问题: CAD图像中靠近左上角的窗规格参数是多少？答案：FM1022 }
**返回要求**
1.答案严格按照问题类型的回答要求进行回答，禁止添加标点及其他无关上下文（如：‘答案是’）
2. 如果不知道问题答案，直接返回无法回答，请勿按照示例答案给出。


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



