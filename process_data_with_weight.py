import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def compute_token_weight(token_loss_diff, loss_mask):
    token_weight = [0] * len(token_loss_diff)
    valid_indices = [j for j, m in enumerate(loss_mask) if m == 1]
    valid_scores = [token_loss_diff[j] for j in valid_indices]
    tn = len(valid_scores)
    if tn > 0:
        t_sorted_indices = sorted(range(tn), key=lambda k: valid_scores[k])
        t_top_20 = int(tn * 0.2)
        t_mid_60 = int(tn * 0.8)
        # 赋值区域：前20%
        for idx in t_sorted_indices[:t_top_20]:
            token_weight[valid_indices[idx]] = 1  # 前20%赋值
        # 赋值区域：20%-80%
        for idx in t_sorted_indices[t_top_20:t_mid_60]:
            token_weight[valid_indices[idx]] = 0  # 20%-80%赋值
        # 赋值区域：后20%
        for idx in t_sorted_indices[t_mid_60:]:
            token_weight[valid_indices[idx]] = 0  # 后20%赋值
    # mask=0的token_weight已为0
    return token_weight

def process_file(input_path, output_path):
    # 先读取所有数据，收集loss_diff
    items = []
    loss_diffs = []
    with open(input_path, "r", encoding="utf-8") as fin:
        for line in fin:
            item = json.loads(line)
            items.append(item)
            loss_diffs.append(item.get("loss_diff"))
    # 对loss_diff排序分组
    n = len(loss_diffs)
    sorted_indices = sorted(range(n), key=lambda i: loss_diffs[i])
    top_20 = int(n * 0.2)
    mid_60 = int(n * 0.8)
    # 构建item_weight数组
    item_weights = [0] * n
    # 赋值区域：前20%
    for idx in sorted_indices[:top_20]:
        item_weights[idx] = 1  # 前20%赋值
    # 赋值区域：20%-80%
    for idx in sorted_indices[top_20:mid_60]:
        item_weights[idx] = 0  # 20%-80%赋值
    # 赋值区域：后20%
    for idx in sorted_indices[mid_60:]:
        item_weights[idx] = 0  # 后20%赋值

    results = []
    # 多线程并行处理token_weight
    with ThreadPoolExecutor() as executor:
        futures = []
        for i, item in enumerate(items):
            token_loss_diff = item.get("token_loss_diff")
            loss_mask = item.get("loss_mask")
            futures.append(executor.submit(compute_token_weight, token_loss_diff, loss_mask))
        token_weights = []
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing token_weight"):
            token_weights.append(f.result())
    # 保证顺序
    token_weights = [tw for _, tw in sorted(zip(range(len(token_weights)), token_weights), key=lambda x: x[0])]

    for i, item in enumerate(items):
        prompt = item.get("prompt")
        output = item.get("output")
        text = item.get("text")
        token_weight = token_weights[i]
        # 处理item_weight
        item_weight = item_weights[i]

        results.append({
            "prompt": prompt,
            "output": output,
            "text": text,
            "token_weight": token_weight,
            "item_weight": item_weight
        })

    # 写入新文件
    with open(output_path, "w", encoding="utf-8") as fout:
        for item in results:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="输入文件路径")
    parser.add_argument("--output_path", type=str, required=True, help="输出文件路径")
    args = parser.parse_args()
    process_file(args.input_path, args.output_path)
