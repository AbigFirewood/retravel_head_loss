
import os 
from data_utils.distill_datasets import DistillDataset
from torch.utils.data import DataLoader
import numpy as np
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)


def prepare_dataset(args, tokenizer):
    
        data = {}
        data["train"] = DistillDataset(
            args, "train", tokenizer,
            {}
        )
        print("Num of train data: {}".format(len(data["train"])))
        
        data["dev"] = DistillDataset(
            args, "dev", tokenizer,
            {}
        )
        print("Num of dev data: {}".format(len(data["dev"])))

        if os.path.exists(os.path.join(args.data_dir, "test.jsonl")):
            data["test"] = DistillDataset(
                args, "test", tokenizer,
                {}
            )
            print("Num of test data: {}".format(len(data["test"])))
        return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('-s', '--s_len', metavar='N', type=int, help='a number')
    # parser.add_argument('-e', '--e_len', metavar='N', type=int, help='a number')
    parser.add_argument('--model_path', type=str, default=None, help='path to model')
    parser.add_argument('--model_name', type=str, default=None, help='name of model')
    parser.add_argument('--model_name_suffix', type=str, default=None, help='name of model')
    parser.add_argument('--model_provider', type=str, default="LLaMA", help='which model to use')
    parser.add_argument('--api_key', type=str, default="", help='OpenAI API Key')
    parser.add_argument('--mask_topk', type=int, default=0, help='mask topk heads, input a negative value to mask random heads')
    parser.add_argument("--data-dir", type=str, default="D:\project\Project\Python\Deep_learning\\buaa_projects\\retravel_head\data_dir")
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument('--max-length', type=int, default=1024, # 文本长度的最大值 会pad到这个值
                       help='max length of input')
    parser.add_argument("--model-type", type=str, default="llama") # 需要改动
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Data Loader batch size')
    parser.add_argument("--num-workers", type=int, default=1)

    # from transformers import AutoTokenizer
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    dataset = prepare_dataset(args,tokenizer)# 数据集加载过程 结束
    # 加载数据
    train_dataloader = DataLoader(
                dataset['train'], 
                # sampler=sampler, 
                shuffle=True,
                drop_last=True,
                batch_size=1, 
                num_workers=1, 
                collate_fn=dataset["train"].collate
            )
            # 开始测试
    print("start test!")
            # 控制停止
    end_epoch = False
    train_iter = iter(train_dataloader) # 创建数据迭代器
    while True:
                global_batch = []
                try:
                        (input_batch, output_batch, _) = next(train_iter)
                        dataset["train"].move_to_device(
                            [input_batch, output_batch],"cpu")
                        global_batch.append({
                            "input_batch": input_batch,
                            "output_batch": output_batch,
                        })
                except StopIteration:
                        end_epoch = True
                        break
                if end_epoch:
                    break
                # 取batch和停止
                for batch in global_batch: # 前线传播
                    input_data=batch["input_batch"]["prompt"]
                    print(input_data)
                    # model = AutoModelForCausalLM.from_pretrained("D:\project\Project\Python\Deep_learning\\buaa_projects\Retrieval_Head\model")
                    # # logits = model(**input_data).logits
                    # print(logits.size())
                    # input_m  =batch["input_batch"]["attention_mask"].size()
                    output_data = batch["output_batch"]
                    # input_ids = tokenizer("hesdfasdf,asdfasdfasdf asdlfkj " , return_tensors="pt")['input_ids'].size()
                    input = "asdfasdfasdfasdf"
        

#   huggingface-cli download --resume-download TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T --local-dir D:\project\Project\Python\Deep_learning\buaa_projects\Retrieval_Head\model

# from jsonlines import jsonlines
# import random
# all = []
# with jsonlines.open('D:\project\Project\Python\Deep_learning\\buaa_projects\Retrieval_Head\OpenO1-SFT.jsonl', 'r') as writer:
#     for obj in writer:
#         all.append({"prompt": obj["instruction"], "output": obj["output"]})
# get = random.sample(all,1000)
# with jsonlines.open('D:\project\Project\Python\Deep_learning\\buaa_projects\Retrieval_Head\\test.jsonl', 'w') as writer:
#     for obj in get:
#         writer.write(obj)
        



