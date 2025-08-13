import os
import argparse
import json
from PIL import Image
import io
import base64
from openai import OpenAI
from tqdm import tqdm
import time
import traceback
import logging
import sys


# 配置日志 - 修改为同时输出到控制台和文件
def setup_logging(output_dir):
    """配置日志系统，同时输出到控制台和文件"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 创建日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # 创建文件处理器 - 使用时间戳命名日志文件
    log_filename = os.path.join(output_dir, f"vqa_log_{time.strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(formatter)

    # 添加处理器到日志器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # 设置openai库的日志级别为WARNING，避免过多日志
    logging.getLogger("openai").setLevel(logging.WARNING)

    return log_filename  # 返回日志文件路径，便于后续引用


class APIRetryError(Exception):
    """API重试失败异常"""
    pass


def parse_args():
    '''
    Arguments
    '''
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=str,
                        default="/home/jupyter-lpy/project/target_detection/SymPoint_main/dataset_FloorplanCAD/test/png/",
                        help='')
    parser.add_argument('--cad_qa', type=str,
                        default="/home/jupyter-lpy/project/target_detection/SymPoint_main/dataset_FloorplanCAD/test/cad_qa_test.json",
                        help='')
    parser.add_argument('--api_key', type=str, default="sk-ljbdphromcnxchfgbcvaxvlcslukcaasqmqafddompcgxeun",
                        help='')
    parser.add_argument('--output', type=str,
                        default="/home/jupyter-lpy/project/target_detection/CAD_VQA/allDate_vqa/qwen/output/test/",
                        help='Path to save the results')
    parser.add_argument('--checkpoint_interval', type=int, default=5,
                        help='Save checkpoint every N images (default: 5)')
    args = parser.parse_args()
    return args


def r_json(json_file):
    # 打开JSON文件并读取数据
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)  # 返回Python字典或列表
    return data


def save_json(data, file_path):
    """安全写入JSON文件（原子操作）"""
    temp_path = file_path + ".tmp"
    try:
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        os.replace(temp_path, file_path)
    except Exception as e:
        logging.error(f"Failed to save JSON to {file_path}: {str(e)}")
        if os.path.exists(temp_path):
            os.remove(temp_path)


def load_checkpoint(output_dir):
    """加载检查点数据"""
    checkpoint_path = os.path.join(output_dir, "checkpoint.json")
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {str(e)}. Starting from scratch.")
    return {
        "completed_images": [],
        "vlm_data": {}
    }


def save_checkpoint(output_dir, checkpoint_data):
    """保存检查点数据"""
    checkpoint_path = os.path.join(output_dir, "checkpoint.json")
    save_json(checkpoint_data, checkpoint_path)


def convert_image_to_png_base64(input_image_path):
    try:
        with Image.open(input_image_path) as img:
            byte_arr = io.BytesIO()
            img.save(byte_arr, format='PNG')  # 明确使用 PNG
            base64_str = base64.b64encode(byte_arr.getvalue()).decode('utf-8')
            return base64_str
    except Exception as e:
        logging.error(f"Error converting image {input_image_path}: {str(e)}")
        return None


def get_full_response_from_image(client, system_prompt, base64_image, que, max_retries=3, retry_delay=5):
    """
    获取图像处理后的完整响应（带重试机制）

    :param max_retries: 最大重试次数
    :param retry_delay: 重试之间的基础等待时间（秒）
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="Qwen/Qwen2.5-VL-72B-Instruct",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
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
            return response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)  # 指数退避
                logging.warning(
                    f"API error (attempt {attempt + 1}/{max_retries}) for question: '{que}'. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                # 重试全部失败后抛出自定义异常
                raise APIRetryError(f"API failed after {max_retries} attempts for question: '{que}'. Error: {str(e)}")


if __name__ == "__main__":
    args = parse_args()

    # 确保输出目录存在
    os.makedirs(args.output, exist_ok=True)

    # 设置日志系统 - 同时输出到控制台和文件
    log_file = setup_logging(args.output)
    logging.info(f"Logging initialized. Log file: {log_file}")

    # 加载或初始化检查点
    checkpoint = load_checkpoint(args.output)
    completed_images = set(checkpoint.get("completed_images", []))
    vlm_data = checkpoint.get("vlm_data", {})

    # 加载数据集
    cad_qa = r_json(args.cad_qa)

    # 获取需要处理的图像ID列表（过滤已完成的）
    image_id_list = [
        image_id for image_id in cad_qa["indexes"]["image_to_questions"].keys()
        if image_id not in completed_images
    ]

    # 如果没有需要处理的图像，直接退出
    if not image_id_list:
        logging.info("All images already processed. Exiting.")
        exit(0)

    # 初始化客户端
    client = OpenAI(
        api_key=args.api_key,
        base_url="https://api.siliconflow.cn/v1"
    )

    system_prompt = """
**角色提示**
你是一个专业的建筑平面CAD图纸审查员。
**任务描述**
在回答具体问题前，请先快速浏览图像，分析图纸中的元素以及空间位置关系，建立对图纸的整体认知，然后根据用户提供的建筑平面CAD图像内容及建筑领域专业知识回答问题。
**系统提示**
1. 建筑平面CAD图像左上角为坐标原点，坐标系长宽各100，对应实际10m×10m的空间。
2.问题有9个类型。
3.问题类型及最终答案定义如下：
(1)数量问题
   最终答案：直接返回整数数字
	   示例：{问题：CAD图像中有几个双扇门？: 答案：5}
(2)包含‘是否’或‘有无’的问题  
   最终答案：回答`是`或`否`、`有`或`无`。
示例：{问题: CAD图像中是否存在浴缸？答案：是}
(3)面积问题
   最终答案：数值+平方米，如 `20平方米` 
示例：{问题: CAD图像中位于（65.84，23.26）的电梯的实际面积是多少？ 最终答案：4平方米}
(4)尺寸问题
最终答案：数值+单位×数值+单位，如‘多少m×多少m’
示例：{问题: CAD图像中位于（18.66，55.97）的浴室的实际尺寸多大？最终答案：0.9m×0.9m}
(5)询问载重参数问题
最终答案：数值+kg，如`1000kg`
示例：{问题: CAD图像中位于（33.66，38.97）的电梯的载重参数是多少？最终答案：1000kg}
(6)询问“符合性”问题
   最终答案：只回答`符合`或`不符合`。
示例：{问题: 当前CAD图像中卫生间的使用面积符合《住宅设计规范GB 50096-2011》吗？
最终答案：符合}
(7)建筑功能类型问题
最终答案：用简短词语回答
示例：{问题: 该空间属于哪种建筑功能类型？最终答案：住宅}
(8) 询问特定场景有哪些配置的问题
最终答案：用‘、’分隔短语
示例：{问题: CAD图像中的厨房配置了哪些厨房用具？最终答案：水槽、煤气炉}
(9) 询问规格参数问题
最终答案：直接给出规格参数
示例：{问题: CAD图像中位于（29.36，54.20）的窗规格参数是多少？最终答案：FM1022 }
**返回要求**
1.返回内容的格式：推理过程：xxxx; 最终答案：xxxxx
2.最终答案部分严格按照问题类型的回答要求进行回答，禁止添加标点及其他无关上下文（如：‘答案是’）
3.如果不知道问题答案，则返回：推理过程：xxxx; 最终答案：无法回答 
        """

    # 处理每个图像
    total_images = len(image_id_list)
    processed_count = 0

    # 当前正在处理的图像ID（用于错误处理）
    current_image_id = None

    try:
        for idx, image_id in enumerate(tqdm(image_id_list, desc="Processing images")):
            current_image_id = image_id  # 记录当前图像ID
            try:
                # 准备图像路径
                filename = f"{image_id}.png"
                image_path = os.path.join(args.dataset, filename)

                # 转换图像
                base64_image = convert_image_to_png_base64(image_path)
                if not base64_image:
                    logging.error(f"Skipping {image_id} due to image conversion error")
                    continue

                # 初始化该图像的VLM数据（临时存储，不直接修改主数据结构）
                temp_image_data = []

                # 处理该图像的所有问题
                for q_id in cad_qa["indexes"]["image_to_questions"][image_id]:
                    que = cad_qa["questions"][q_id]["question"]
                    que_type = cad_qa["questions"][q_id]["question_type"]
                    ans_id = cad_qa["questions"][q_id]["answer_ref"]
                    ans = cad_qa["answers"][ans_id]["value"]  # 修正了这里的访问方式
                    ans_type = cad_qa["answers"][ans_id]["type"]

                    # 获取模型响应
                    time.sleep(1)  # 基础延迟以避免速率限制

                    # 尝试获取响应，如果重试失败会抛出APIRetryError
                    pred = get_full_response_from_image(client, system_prompt, base64_image, que)
                    print(f"预测结果：{pred}")
                    # 保存结果到临时列表
                    temp_image_data.append({
                        "question": que,
                        "real": ans,
                        "pred": pred,
                        "que_id": q_id,
                        "ans_type": ans_type,
                        "que_type": que_type
                    })

                # 所有问题处理成功，更新主数据结构
                vlm_data[image_id] = temp_image_data
                completed_images.add(image_id)
                processed_count += 1
                logging.info(f"Successfully processed image {image_id}")

                # 定期保存检查点
                if (idx + 1) % args.checkpoint_interval == 0 or (idx + 1) == total_images:
                    checkpoint_data = {
                        "completed_images": list(completed_images),
                        "vlm_data": vlm_data
                    }
                    save_checkpoint(args.output, checkpoint_data)
                    logging.info(f"Checkpoint saved. Processed {idx + 1}/{total_images} images.")

            except APIRetryError as e:
                # API重试失败，记录错误并退出程序
                logging.error(f"Critical API failure for image {image_id}: {str(e)}")
                logging.info("API retries exhausted. Saving progress and exiting.")

                # 保存当前进度（不包括当前失败图像）
                checkpoint_data = {
                    "completed_images": list(completed_images),
                    "vlm_data": vlm_data
                }
                save_checkpoint(args.output, checkpoint_data)

                # 退出程序
                sys.exit(1)

            except Exception as e:
                logging.error(f"Error processing image {image_id}: {str(e)}\n{traceback.format_exc()}")
                # 即使出错也保存当前进度（不包括当前失败图像）
                checkpoint_data = {
                    "completed_images": list(completed_images),
                    "vlm_data": vlm_data
                }
                save_checkpoint(args.output, checkpoint_data)
                logging.info(f"Checkpoint saved after error on image {image_id}")

    except Exception as e:
        logging.error(f"Unexpected error in main loop: {str(e)}\n{traceback.format_exc()}")

    finally:
        # 确保最终保存检查点（不包括当前正在处理的图像）
        if current_image_id and current_image_id in completed_images:
            # 如果当前图像已成功完成，则包含它
            checkpoint_data = {
                "completed_images": list(completed_images),
                "vlm_data": vlm_data
            }
        else:
            # 如果当前图像未完成，则排除它
            checkpoint_data = {
                "completed_images": list(completed_images),
                "vlm_data": vlm_data
            }

        save_checkpoint(args.output, checkpoint_data)
        logging.info("Final checkpoint saved.")

    # 保存最终结果
    pred_path = os.path.join(args.output, "llm_pred.json")
    save_json(vlm_data, pred_path)

    logging.info(f"Processing completed. Results saved to {pred_path}")
    logging.info(f"Log file saved to {log_file}")
    print(f"预测答案已成功写入 {pred_path}")
    print(f"日志文件已保存到 {log_file}")