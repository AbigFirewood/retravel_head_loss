from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
import jsonlines
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
max_length = 0
all_length = 0
count = 0
with jsonlines.open(r"C:\\Users\\有志者\Downloads\AIME_Dataset_1983_2024_new.jsonl",'r') as r:
    for o in r:
        # 测试"prompt"
        print(tokenizer.encode(o["text"]))
        # 测试"prompt" token化后的长度并显示
        prompt_tokens = tokenizer.encode(o["text"]) # 1500
        prompt_length = len(prompt_tokens)
        print(f'prompt token化后长度: {prompt_length}')
        # 计算最大长度
        all_length = all_length+prompt_length
        count += 1
        if prompt_length > max_length:
            max_length = prompt_length
        
print(f'所有数据中最长的prompt token化后长度: {max_length}')   # 50000 长
print(f'所有数据中prompt token化后长度总和: {all_length}')   # 50000 长
print(f'所有数据中prompt token化后长度平均值: {all_length/len(r)}')
