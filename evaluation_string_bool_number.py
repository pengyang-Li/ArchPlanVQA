# "1002-0040": [
#         {
#             "question": "CAD图像中有几个双扇门？",
#             "real": 3,
#             "pred": "2",
#             "type":''
#         }
#     ]

import numpy as np
import re
import json
import argparse
from openai import OpenAI
def parse_args():
    '''
    Arguments
    '''
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--synonyms', type=str, default="/home/jupyter-lpy/project/target_detection/CAD_VQA/data/synonyms.json",
                        help='What is stored is a standard vocabulary mapping table.')
    parser.add_argument('--not_synonyms', type=str,
                        default="/home/jupyter-lpy/project/target_detection/CAD_VQA/data/not_synonyms.json",
                        help='What is stored are non-standard words.')
    parser.add_argument('--api_key', type=str,
                        default="sk-ljbdphromcnxchfgbcvaxvlcslukcaasqmqafddompcgxeun",
                        help='')
    parser.add_argument('--llm_pred', type=str, default="/home/jupyter-lpy/project/target_detection/CAD_VQA/allDate_vqa/qwen/删除推理后的评估数据.json",
                        help='What is stored are the prediction results of the model.')

    args = parser.parse_args()
    return args
def r_json(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        return json.load(file)


def remove_all_whitespace(text):
    """
    删除字符串中的所有空白字符（包括空格、制表符、换行符等）

    参数:
    text (str): 输入字符串

    返回:
    str: 清理后的字符串（无任何空白字符）
    """
    return re.sub(r'\s+', '', text)
def check_mapping(data, mapped):
    # 定义固定词汇列表
    vocabulary = {
        "冰箱", "水槽", "煤气炉", "蹲式厕所", "小便池", "坐便器",
        "浴室", "浴缸", "货梯", "客梯", "无障碍", "高区", "消防", "餐梯",
        "污梯",  "服务", "担架", "办公", "医疗", "手术专用",
        "公共", "观光", "非机动车提升"
    }
    # 找出在data中存在的词汇
    found = [word for word in vocabulary if word in data]
    # 判断found和mapped是否包含相同的元素（不考虑顺序）
    found_mapped = [word for i in mapped for word in vocabulary if word in i]

    return sorted(found) == sorted(found_mapped)
def map_to_standard_terms(raw_answer: str):
    global synonym_map
    global not_synonym
    global new_add_synonym
    global new_add_not_synonym
    keywords = [word.strip() for word in raw_answer.split('、')]
    mapped_terms = []
    for keyword in keywords:
        # 优先检查映射表（强制覆盖）
        if keyword in synonym_map:
            mapped_terms.append(synonym_map[keyword])
            continue
        # 检查该词汇是否在非标准词汇表中
        if keyword in not_synonym["not_synonyms"]:
            mapped_terms.append(keyword)
            continue
        # 语义匹配：找到最接近的标准术语

        print(f"第1次调用大模型")
        user_query = f"请从以下词汇列表中找出与'{keyword}'同义的词语：{candidate_str}。如果找到直接返回该词语，未找到请返回0（禁止任何解释）"
        response = client.chat.completions.create(
            model="Qwen/Qwen3-32B",
            messages=[
                {"role": "system", "content": system_prompt},  # 系统角色：设定任务规则
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_query
                        }
                    ]
                }
            ],

            temperature=0,  # 关键参数：确定性输出
            max_tokens=5,  # 限制输出长度
            # enable_thinking = False#禁止输出思考过程

        )
        raw_output = response.choices[0].message.content.strip()

        print(f"模型输出结果：{raw_output}")
        # 正则提取数字结果
        import re
        match = re.match(r'^(\d)$', raw_output)
        if match and match.group(1) == '0':
            result_word =  keyword
            not_synonym["not_synonyms"].append(keyword) # 总表
            new_add_not_synonym["not_synonym"].append(keyword)# 用于查看本次运行时添加的
        else:
            result_word =  raw_output
            synonym_map[keyword] = result_word# 总表
            new_add_synonym[keyword] = result_word# 用于查看本次运行时添加的
        mapped_terms.append(result_word)

    return mapped_terms
def calculate_accuracy(true_values, predicted_values):
    """
    计算预测正确率

    参数:
    true_values -- 真实值列表
    predicted_values -- 预测值列表

    返回:
    正确率 (float)
    """
    if len(true_values) != len(predicted_values):
        raise ValueError("两个列表的长度必须相同")

    correct = 0
    for true, pred in zip(true_values, predicted_values):
        if true == pred:
            correct += 1


    return correct,len(true_values)
def compare_dimensions(real, pred):
    def parse_dimensions(s):
        # 使用正则表达式匹配尺寸模式（例如：2.5m×2.0m 或 3m*4m）
        pattern = r'(\d+\.?\d*m[×*xX]\d+\.?\d*m)'
        matches = re.findall(pattern, s)

        if not matches:
            print("输入尺寸格式不对")
            print(f"尺寸信息为{s}")
            return None  # 未找到尺寸

        # 提取最后一个匹配的尺寸（假设最新预测在文本末尾）
        dim_str = matches[-1]

        # 移除所有'm'字符并按'×'或'*'分割
        s_clean = dim_str.replace('m', '')
        parts = re.split('[×*]', s_clean)

        if len(parts) != 2:
            return None  # 格式错误

        try:
            return [float(parts[0]), float(parts[1])]
        except ValueError:
            return None  # 无法转换为数字

    # 解析真实值和预测值
    real_nums = parse_dimensions(real)
    pred_nums = parse_dimensions(pred)
    if "无法回答" in data["pred"] or data["pred"] == "":
        return False
    # 检查解析是否成功
    if real_nums is None or pred_nums is None:
        print("长宽问题解析失败！！")
        print(f"预测值：{pred}")
        return False

    # 新增：对长宽进行排序（升序）
    real_sorted = sorted(real_nums)
    pred_sorted = sorted(pred_nums)

    # 赋值：大值为length，小值为width
    real_length, real_width = real_sorted[1], real_sorted[0]
    pred_length, pred_width = pred_sorted[1], pred_sorted[0]

    # 分别比较长和宽
    length_diff = abs(real_length - pred_length)
    width_diff = abs(real_width - pred_width)

    return length_diff < 0.01 and width_diff < 0.01


def split_number_unit(s):
    match = re.match(r'^(\d+\.?\d*)([a-zA-Z%]+)$', s)  # 匹配数字(含小数)和单位
    if match:
        return float(match.group(1)), match.group(2)  # 返回数字和单位
    return None, None  # 匹配失败时返回空值
def split_number_unit_area(s):
    match = re.match(r'^([+-]?\d+\.?\d*|\.\d+)(平方米)$', s)  # 匹配数字(含小数)和单位
    if match:
        return float(match.group(1)), match.group(2)  # 返回数字和单位
    return None, None  # 匹配失败时返回空值
def decide_string_TF(data):
    data["pred"] = remove_all_whitespace(data["pred"])
    question = data["question"]
    if "规格参数" in question:
        if data["real"] == data["pred"]:
            return True
        else:
            return False
    elif "载重参数" in question:
        real_num,real_unit = split_number_unit(data["real"])
        pred_num, pred_unit = split_number_unit(data["pred"])
        if real_num and real_unit and pred_num and pred_unit:
            if pred_unit in ["t","T"]:
                pred_num = pred_num*1000
            if real_unit in["t","T"]:
                real_num = real_num*1000
            tolerance = 1.0 # 容错值
            if abs(pred_num-real_num)<tolerance:
                return True
            return False
        else:
            if "无法回答" in data["pred"] or data["pred"] == "":
                return False
            else:
                print("载重参数的数据分割错误！！")
                shuchu = data["pred"]
                print(f"预测值：{shuchu}")
                return False
    elif "尺寸" in question:
        if "无法回答" in data["pred"] or data["pred"] == "":
            return False
        if compare_dimensions(data["real"],data["pred"]):
            return True
        else:
            return False
    elif "面积" in question:
        try:
            if "无法回答" in data["pred"] or data["pred"] == "":
                return False
            real_num, real_unit = split_number_unit_area(data["real"])
            pred_num, pred_unit = split_number_unit_area(data["pred"])
            if pred_num is None or pred_unit is None:
                return False
            return abs(real_num - pred_num) < 0.01
        except Exception as e:
            # 输出原始数据值和异常信息
            print(f"Error occurred when processing area question. Raw data:")
            print(f"data['real']: {data.get('real', 'N/A')}")
            print(f"data['pred']: {data.get('pred', 'N/A')}")
            print(f"Exception details: {str(e)}")
            return False


    else:
        mapped = map_to_standard_terms(data["pred"])
        if "建筑功能" in question:
            if mapped[0] == data["real"]:
                return True
            else:
                return False
        else:
            return check_mapping(data["real"],mapped)
if __name__ == "__main__":

    args = parse_args()
    # 数据加载
    candidate_str = ["停车场", "住宅", "学校", "医院", "冰箱", "水槽", "煤气炉", "蹲式厕所", "小便池", "坐便器",
                     "浴室", "浴缸", "货梯", "客梯", "无障碍电梯", "高区电梯", "消防电梯", "餐梯",
                     "污梯",  "服务电梯", "担架电梯", "办公电梯", "医疗电梯", "手术专用电梯",
                     "公共电梯", "观光电梯", "非机动车提升电梯"]
    client = OpenAI(
        api_key=args.api_key,  # 替换为你的实际API密钥
        base_url="https://api.siliconflow.cn/v1"  # SiliconFlow的API端点
    )
    system_prompt = """
        你是一个专业的同义词判断助手，必须严格遵循以下规则：
            1. 仅分析词语的核心语义，忽略用法差异和词性变化
            2. 如果给定的列表中存在同义的词语则返回同义词，否则返回0（禁止任何解释）
            4. 处理多义词时取最常用含义进行判断

            """
    # 加载术语映射表（仅包含需强制覆盖的词汇）
    synonym_map = r_json(args.synonyms)
    # 加载术非标准词汇表
    not_synonym = r_json(args.not_synonyms)

    json_data = r_json(args.llm_pred)
    new_add_synonym = {}
    new_add_not_synonym = {"not_synonym":[]}
    # 判断数据类型，进行划分。
    numbers_real = []
    numbers_pred = []
    bool_real = []
    bool_pred = []
    string_num = 0
    string_true_num = 0
    for image_id,datas in json_data .items():
        for data in datas:
            if data["ans_type"] == "number":
                numbers_real.append(data["real"])
                try:
                    i_x = int(remove_all_whitespace(data["pred"]))
                    numbers_pred.append(i_x)
                except (ValueError, TypeError):
                    numbers_pred.append(0)

            elif data["ans_type"] == "bool":
                bool_real.append(data["real"])
                bool_pred.append(remove_all_whitespace(data["pred"]))
            else:
                string_num +=1
                if decide_string_TF(data):
                    string_true_num +=1

    #计算number的准确率


    num_true,num_sum= calculate_accuracy(numbers_real,numbers_pred)
    accuracy_num = num_true/num_sum
    #计算bool的准确率
    bool_true, bool_sum= calculate_accuracy(bool_real,bool_pred)
    accuracy_bool = bool_true/bool_sum
    #计算string的准确率
    print(f"正确数量：{string_true_num}，总数量：{string_num}")
    accuracy_string = string_true_num/string_num
    accuracy_sum = (string_true_num+bool_true+num_true)/(string_num+bool_sum+num_sum)
    print("Evaluation Results:")
    print(f"number正确率: {accuracy_num:.4f}")  # 输出: 正确率: 1.00
    print(f"bool正确率: {accuracy_bool:.4f}")  # 输出: 正确率: 1.00
    print(f"string正确率: {accuracy_string:.4f}")
    print(f"综合准确率：{accuracy_sum:.4f}")
    #---------------------------存储映射文件--------------------------------------
    # 保存词汇强制映射表sysnonyms
    with open(args.synonyms, 'w', encoding='utf-8') as f:  # 注意这里使用'w'模式覆盖原文件
        json.dump(synonym_map, f, ensure_ascii=False,indent=4)  # indent=4 用于美化格式

    # 保存非标准词汇表not_sysnonyms
    with open(args.not_synonyms, 'w', encoding='utf-8') as f:  # 注意这里使用'w'模式覆盖原文件
        json.dump(not_synonym, f, ensure_ascii=False,indent=4)  # indent=4 用于美化格式

# 保存本次代码添加的词汇强制映射表
    with open("/home/jupyter-lpy/project/target_detection/CAD_VQA/data/new_add_synonym.json", 'w', encoding='utf-8') as f:  # 注意这里使用'w'模式覆盖原文件
        json.dump(new_add_synonym, f, ensure_ascii=False,indent=4)  # indent=4 用于美化格式

# 保存本次代码添加的词汇强制映射表
    with open("/home/jupyter-lpy/project/target_detection/CAD_VQA/data/new_add_not_synonym.json", 'w', encoding='utf-8') as f:  # 注意这里使用'w'模式覆盖原文件
        json.dump(new_add_not_synonym, f, ensure_ascii=False,indent=4)  # indent=4 用于美化格式
