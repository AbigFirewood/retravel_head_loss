# import os
# import json
# import random
# import html

# from transformers import AutoTokenizer

# from loss_see import print_text_with_color_in_html  # 确保loss_see.py在同目录下

# tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")

# # 1. 读取数据
# input_path = "D:\\project\\Project\\Python\\Deep_learning\\buaa_projects\\Retrieval_Head\\test.jsonl"
# output_dir = os.path.dirname(input_path)

# bucket_size = 512
# buckets = {}
# buckets_given = {}
# all = []
# all_count = 0
# time = 0
# with open(input_path, "r", encoding="utf-8") as f:
#     for line in f:
#         item = json.loads(line)
#         data = item.get("data", "")
#         loss_diff = item.get("loss_diff", [])
#         token_loss_diff = item.get("token_loss_diff", [])
#         input_ids = item.get("input_ids", [])
#         output = item.get("output", "")
#         souse_len = item.get("souse_len", 0)
#         # 2. 统计token数
#         token_count = len(tokenizer.encode(data+output, add_special_tokens=False))
#         all_count += token_count
#         time += 1
#         # 3. 分桶
#         bucket_idx = token_count // bucket_size
#         bucket_start = bucket_idx * bucket_size
#         bucket_end = bucket_start + bucket_size
#         bucket_name = f"{bucket_start}-{bucket_end}"
#         if bucket_name not in buckets:
#             buckets[bucket_name] = []
#         if bucket_name not in buckets_given:
#             buckets_given[bucket_name] = []
#         buckets[bucket_name].append(item)
#         buckets_given[bucket_name].append({"prompt":item["data"], "output":item["output"], "Attinfluence":loss_diff})
#         all.append({"prompt":item["data"], "output":item["output"], "Attinfluence":loss_diff/token_count})
# print(f"平均总token数: {all_count/time}")
# all = all[:round(0.2*len(all))]  # 显示前20%的数据
# # 3. 保存
# # for bucket_name, items in buckets_given.items():
# output_path = os.path.join(output_dir, f"train.jsonl")
# with open(output_path, "w", encoding="utf-8") as f:
#         for item in all:
#             f.write(json.dumps(item, ensure_ascii=False) + "\n")
# # 4. 可视化一个随机样本
# for bucket_name, items in buckets.items():
#     # output_path = os.path.join(output_dir, f"{bucket_name}.jsonl")
#     # with open(output_path, "w", encoding="utf-8") as f:
#     #     for item in items:
#     #         f.write(json.dumps(item, ensure_ascii=False) + "\n")
#     # 随机抽取一个可视化
#     example = items[0]
#     input_ids = example.get("input_ids", [])
#     souse_len = example.get("souse_len", 0)
#     token_loss_diff = example.get("token_loss_diff", [])
#     # 将input_ids逐token解码为可显示的token序列，去除▁前缀
#     tokens = [html.escape(token.lstrip('▁').strip()) for token in tokenizer.convert_ids_to_tokens(input_ids)]
#     new_token_cnt = len(input_ids) - souse_len
#     vis_path = os.path.join(output_dir, f"{bucket_name}_vis.html")
#     print_text_with_color_in_html(tokens, new_token_cnt, token_loss_diff, show_token_border=True, path=vis_path)


import os
import json
import random
import html

from transformers import AutoTokenizer

from loss_see import print_text_with_color_in_html  # 确保loss_see.py在同目录下

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")

# 1. 读取数据
input_path = "D:\\project\\Project\\Python\\Deep_learning\\buaa_projects\\Retrieval_Head\\test.jsonl"
output_dir = os.path.dirname(input_path)

bucket_size = 512
buckets = {}
buckets_given = {}
all = []
all_count = 0
time = 0
with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        data = item.get("data", "")
        loss_diff = item.get("loss_diff", [])
        token_loss_diff = item.get("token_loss_diff", []) # 求和
        input_ids = item.get("input_ids", [])
        output = item.get("output", "")
        souse_len = item.get("souse_len", 0)
        # 2. 统计token数
        # token_count = len(tokenizer.encode(output, add_special_tokens=False))+1
        #@ all_count += token_count
        time += 1
        # 3. 分桶
        # bucket_idx = token_count // bucket_size
        # bucket_start = bucket_idx * bucket_size
        # bucket_end = bucket_start + bucket_size
        # bucket_name = f"{bucket_start}-{bucket_end}"
        # if bucket_name not in buckets:
        #     buckets[bucket_name] = []
        # if bucket_name not in buckets_given:
        #     buckets_given[bucket_name] = []
        # buckets[bucket_name].append(item)
        # buckets_given[bucket_name].append({"prompt":item["data"], "output":item["output"], "Attinfluence":loss_diff})
        # 统计token_loss_diff中非零数字的个数
        token_count = sum(1 for i in token_loss_diff if i != 0)
        # 求和token_loss_diff
        token_loss_diff_sum = sum(token_loss_diff)

        # 计算token_weight
        nonzero_indices = [i for i, v in enumerate(token_loss_diff) if v != 0]
        nonzero_values = [token_loss_diff[i] for i in nonzero_indices]
        sorted_indices = [x for _, x in sorted(zip(nonzero_values, nonzero_indices), reverse=True)]
        n = len(sorted_indices)
        top_20 = int(n * 0.2)
        bottom_20 = n - top_20
        # 生成权重列表
        token_weight = []
        for idx in range(len(token_loss_diff)):
            if token_loss_diff[idx] == 0:
                token_weight.append(0)
            else:
                rank = sorted_indices.index(idx)
                if rank < top_20:
                    token_weight.append(1)
                elif rank >= bottom_20:
                    token_weight.append(0.5)
                else:
                    token_weight.append(1)
        # 挑出token_loss_diff非0的数据并作为新list出现
        token_loss_diff_has_num = [i for i in token_loss_diff if i != 0]
        # 将token_loss_diff_has_num按照从大到小排序 测定两个分界点 将数据分为20% 20%-80% 80%-100%
        
        all.append({
            "prompt": item["data"],
            "output": item["output"],
            "Attinfluence": token_loss_diff_sum/token_count,
            "token_weight": token_weight[:1024]
        })

# 计算item_weightf
# 先按Attinfluence从大到小排序
all.sort(key=lambda x: x["Attinfluence"], reverse=True)
n = len(all)
top_20 = int(n * 0.2)
bottom_20 = n - top_20
for idx, item in enumerate(all):
    if idx >= bottom_20:
        item["item_weight"] = 0
    elif idx < top_20:
        item["item_weight"] = 1.0
    else:
        item["item_weight"] = 0

# print(f"平均总token数: {all_count/time}")
# 按照Attinfluence从大到小排序
# all.sort(key=lambda x: x["Attinfluence"], reverse=True)
# all = all[:round(0.2*len(all))]  # 显示前20%的数据
# 3. 保存
# for bucket_name, items in buckets_given.items():
output_path = os.path.join(output_dir, f"train.jsonl")
with open(output_path, "w", encoding="utf-8") as f:
        for item in all:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")