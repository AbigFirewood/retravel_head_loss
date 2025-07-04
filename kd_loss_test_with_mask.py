"""
This script is adapted from 
https://github.com/gkamradt/LLMTest_NeedleInAHaystack

# GPT-4
(
python -u needle_in_haystack.py --s_len 0 --e_len 128000\
    --model_provider OpenAI\
    --model_name gpt-4-1106-preview
    --api_key $OPENAI_API_KEY
) 2>&1  | tee logs/eval_gpt_4_128k.log

# LLaMA 2 32K. Remember to download the model first
(
python -u needle_in_haystack.py --s_len 0 --e_len 128000\
    --model_provider LLaMA\
    --model_path ../../../Llama-2-7B-32K-Instruct
) 2>&1  | tee logs/eval_llama2_32k_instruct.log

# LongChat. Remember to download the model first
(
python -u needle_in_haystack.py --s_len 0 --e_len 128000\
    --model_provider LLaMA\
    --model_path /ML-A800/models/longchat-7b-v1.5-32k
) 2>&1  | tee logs/eval_longchat.log

# Our llama-2-7b-80k, requires 4*80G A100
# require you to download the model first
(
python -u needle_in_haystack.py --s_len 0 --e_len 128000\
    --model_provider LLaMA\
    --model_path ../../../llama-2-7b-80k
) 2>&1  | tee logs/eval_llama-2-7b-80k.log
"""

#import tiktoken
import os 
import glob
import json
from transformers import AutoTokenizer, AutoConfig
import sys
import random
# sys.path.append("./faiss_attn/")
from source.modeling_llama import LlamaForCausalLM, LlamaConfig
from source.modeling_qwen2 import Qwen2ForCausalLM
from source.modeling_mixtral import MixtralForCausalLM
from source.modeling_mistral import MistralForCausalLM
from tqdm import tqdm
# from source.modeling_phi3 import Phi3ForCausalLM

import numpy as np
import argparse
from rouge_score import rouge_scorer
from data_utils.distill_datasets import DistillDataset
from torch.utils.data import DataLoader
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

#from openai import OpenAI
from datetime import datetime, timezone
from collections import defaultdict
import time
import torch



def reset_rope(model, model_max_train_len, scaling_factor):
    for l in model.model.layers:
        l.self_attn.rotary_emb.scaling_factor = scaling_factor
        l.self_attn.rotary_emb._set_cos_sin_cache(seq_len=model_max_train_len, device=l.self_attn.rotary_emb.inv_freq.device, dtype=torch.float32)
    return

class LLMNeedleHaystackTester:
    """
    This class is used to test the LLM Needle Haystack.
    """
    def __init__(self,
                 needle="\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n",
                 haystack_dir="PaulGrahamEssays",
                 retrieval_question="What is the best thing to do in San Francisco?",
                 results_version = 1,
                 context_lengths_min = 1000,
                 context_lengths_max = 128000,
                 context_lengths_num_intervals = 40,
                 context_lengths = None,
                 document_depth_percent_min = 0,
                 document_depth_percent_max = 100,
                 document_depth_percent_intervals = 10,
                 document_depth_percents = None,
                 document_depth_percent_interval_type = "linear",
                 model_provider = "OpenAI",
                 mask_topk=0,
                 anthropic_api_key = None,
                 model_name='',
                 model_name_suffix=None,
                 num_concurrent_requests = 1,
                 save_results = True,
                 save_contexts = True,
                 final_context_length_buffer = 200,
                 seconds_to_sleep_between_completions = None,
                 print_ongoing_status = True):
        """        
        :param needle: The needle to be found in the haystack. Default is None.
        :param haystack_dir: The directory of text files to use as background context (or a haystack) in which the needle is to be found. Default is Paul Graham Essays.
        :param retrieval_question: The question which with to prompt the model to do the retrieval.
        :param results_version: In case you would like to try the same combination of model, context length, and depth % multiple times, change the results version other than 1
        :param num_concurrent_requests: Due to volume, this object is set up to run concurrent requests, default = 1. Be careful of rate limits.
        :param save_results: Whether or not you would like to save your contexts to file. Warning: These will get long! Default = True
        :param save_contexts: Whether or not you would like to save your contexts to file. Warning: These will get long! Default is True.
        :param final_context_length_buffer: The amount of cushion you'd like to leave off the input context to allow for the output context. Default 200 tokens
        :param context_lengths_min: The minimum length of the context. Default is 1000.
        :param context_lengths_max: The maximum length of the context. Default is 200000.
        :param context_lengths_num_intervals: The number of intervals for the context length. Default is 35.
        :param context_lengths: The lengths of the context. Default is None.
        :param document_depth_percent_min: The minimum depth percent of the document. Default is 0.
        :param document_depth_percent_max: The maximum depth percent of the document. Default is 100.
        :param document_depth_percent_intervals: The number of intervals for the document depth percent. Default is 35.
        :param document_depth_percents: The depth percentages of the document. Default is None.
        :param document_depth_percent_interval_type: The type of interval for the document depth percent. Must be either 'linear' or 'sigmoid'. Default is 'linear'.
        :param model_provider: The provider of the model. Must be either 'OpenAI' or 'Anthropic'. Default is 'OpenAI'.
        :param openai_api_key: The API key for OpenAI. Default is None.
        :param anthropic_api_key: The API key for Anthropic. Default is None.
        :param model_name: The name of the model. Default is 'gpt-4-1106-preview'.
        :param seconds_to_sleep_between_completions: The number of seconds to sleep between completions. Default is None.
        :param print_ongoing_status: Whether or not to print the ongoing status. Default is True.
        """
        if not needle or not haystack_dir or not retrieval_question:
            raise ValueError("Needle, haystack, and retrieval_question must be provided.")
        
        self.needle = needle
        self.haystack_dir = haystack_dir
        self.retrieval_question = retrieval_question
        self.results_version = results_version
        self.num_concurrent_requests = num_concurrent_requests
        self.save_results = save_results
        self.final_context_length_buffer = final_context_length_buffer
        self.save_contexts = save_contexts
        self.seconds_to_sleep_between_completions = seconds_to_sleep_between_completions
        self.print_ongoing_status = print_ongoing_status
        self.model_provider = model_provider
        self.testing_results = []
        self.head_counter = defaultdict(list)
        self.mask_topk = mask_topk
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            self.multi_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"])>1
        else:
            self.multi_gpus = True
        if("/" in model_name):
            self.model_version = model_name.split("/")[-1]
        else: self.model_version = model_name
        if(model_name_suffix is not None): self.model_version += "_" + model_name_suffix

        if context_lengths is None:
            if context_lengths_min is None or context_lengths_max is None or context_lengths_num_intervals is None:
                raise ValueError("Either context_lengths_min, context_lengths_max, context_lengths_intervals need to be filled out OR the context_lengths_list needs to be supplied.")
            else:
                self.context_lengths = np.round(np.linspace(context_lengths_min, context_lengths_max, num=context_lengths_num_intervals, endpoint=True)).astype(int)
        else:
            self.context_lengths = context_lengths

        if document_depth_percents is None:
            if document_depth_percent_min is None or document_depth_percent_max is None or document_depth_percent_intervals is None:
                raise ValueError("Either document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals need to be filled out OR the document_depth_percents needs to be supplied.")
            else:
                if document_depth_percent_interval_type == 'linear':
                    self.document_depth_percents = np.round(np.linspace(document_depth_percent_min, document_depth_percent_max, num=document_depth_percent_intervals, endpoint=True)).astype(int)
                elif document_depth_percent_interval_type == 'sigmoid':
                    self.document_depth_percents = [self.logistic(x) for x in np.linspace(document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals)]
        else:
            self.document_depth_percents = document_depth_percents

        if document_depth_percent_interval_type not in [None, "linear", "sigmoid"]:
            raise ValueError("document_depth_percent_interval_type must be either None, 'linear' or 'sigmoid'. If you'd like your own distribution give a list of ints in via document_depth_percent_intervals")
        
        self.model_name = model_name

        if(self.model_provider not in ["OpenAI", "Anthropic"]):
            self.enc = AutoTokenizer.from_pretrained(model_name)
            print("loading from %s" % model_name)
            config = AutoConfig.from_pretrained(model_name)
            self.layer_num, self.head_num = config.num_hidden_layers, config.num_attention_heads
            print(f"layer number: {self.layer_num}, head number {self.head_num}")
            if "Qwen" in self.model_version:
                self.model_to_test = Qwen2ForCausalLM.from_pretrained(
                       model_name,torch_dtype="auto",device_map='auto',use_flash_attention_2="flash_attention_2"
                    )
            elif "Mixtral" in self.model_version:
                self.model_to_test = MixtralForCausalLM.from_pretrained(
                       model_name,torch_dtype="auto",device_map='auto',use_flash_attention_2="flash_attention_2",trust_remote_code=True,
                    )
            elif "Mistral" in self.model_version:
                self.model_to_test = MistralForCausalLM.from_pretrained(
                       model_name,torch_dtype="auto",device_map='auto',use_flash_attention_2="flash_attention_2",trust_remote_code=True,
                    )
            # elif "Phi3" in self.model_version:
            #     self.model_to_test = Phi3ForCausalLM.from_pretrained(
            #            model_name,torch_dtype="auto",device_map='auto',use_flash_attention_2="flash_attention_2",trust_remote_code=True,
            #         )
            else:
                self.model_to_test = LlamaForCausalLM.from_pretrained(model_name,
                    use_flash_attention_2="flash_attention_2", torch_dtype=torch.bfloat16,device_map='auto').eval()
            if 'llama-2-7b-80k' in self.model_version:
                scaling_factor = 10
                reset_rope(self.model_to_test, model_max_train_len=81920, scaling_factor=scaling_factor)
        # else: 
            # self.model_to_test = OpenAI(api_key=openai_api_key)
            # if(self.model_provider == "OpenAI"):
            #     self.enc = tiktoken.encoding_for_model(self.model_name)
            # elif(self.model_provider == "Anthropic"):
            #     self.enc = Anthropic().get_tokenizer()

        self.model_to_test_description = model_name
        # print(model_name)
        self.evaluation_model = None
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            self.multi_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"])>1
        else:
            self.multi_gpus = True
        model_name = model_name.split('/')[-1]
        # print(model_name)
        if self.mask_topk!=0:
            if model_name=='Mistral-7B-Instruct-v0.2':
                model_name = "Mistral-7B-v0.2-hf"
            with open(f"head_score/{model_name}.json", "r") as file:
                stable_block_list =  json.loads(file.readline())
            stable_block_list = [(l[0], np.mean(l[1])) for l in stable_block_list.items()]
            stable_block_list = sorted(stable_block_list, key=lambda x: x[1], reverse=True) 
            self.block_list = [[int(ll) for ll in l[0].split("-")] for l in stable_block_list][:100]
            print(self.mask_topk)
            if self.mask_topk > 0:
                print(f"masking out top {self.mask_topk} retrieval heads")
            else:
                print(f"masking out random {-self.mask_topk}  heads")
        else:
            self.block_list = []

    def logistic(self, x, L=100, x0=50, k=.1):
        if x == 0:
            return 0
        if x == 100:
            return 100
        return np.round(L / (1 + np.exp(-k * (x - x0))), 3)
    
    def bound_evaluate_and_log(self, *args):
        self.evaluate_and_log(*args)
    
    # def log_rank(content, rank=0):
    # if not dist.is_initialized() or dist.get_rank() == rank:
    #     logging.info(content)


    def prepare_dataset(self,args, tokenizer):
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
    
    
    def run_test(self, args):
        # 加载数据集
        dataset = self.prepare_dataset(args,self.enc) # 数据集加载过程 结束
        # Run through each iteration of context_lengths and depths
        self.bound_evaluate_and_log(dataset)
    
    def generate_prompt(self, context):
        # Generate the prompt for the Anthropic model
        # Replace the following line with the appropriate prompt structure
        test_format=f"This is a very long story book: <book> {context} </book>.\n"
        if self.model_version in ["Mistral-7B-Instruct-v0.2"]:
            prompt = [
            {"role": "user", "content": f"<book>{context}</book>\nBased on the content of the book, Question: {self.retrieval_question}\nAnswer:"},]
        return prompt
    
    def retrieval_calculate(self, attention_maxtrix,retrieval_score, inp, step_token,topk=1):
        for layer_idx in range(32):
            for head_idx in range(32):
                values, idx = attention_maxtrix[layer_idx][0][head_idx][-1].topk(topk)
                for v, i in zip(values, idx):
                    if  self.needle_start <= i < self.needle_end and inp.item()==self.prompt_ids[i].item():
                        retrieval_score[layer_idx][head_idx][0] += 1/(self.needle_end - self.needle_start)
                        retrieval_score[layer_idx][head_idx][1] += step_token
                        break

    def retrieval_head_accumulate(self, retrieval_score):
        for layer_idx in range(32):
            for head_idx in range(32):
                self.head_counter[f"{layer_idx}-{head_idx}"].append(retrieval_score[layer_idx][head_idx][0])

    def compute_cross_entropy_loss(self, logits, target, log=None):
        self.padding_id = -100
        pad_mask = target.ne(self.padding_id) # 没pad的地方为1
        target = target.unsqueeze(-1)
        target = torch.where(        # 处理promtoken 用-100替代
            target.eq(-100), 
            torch.zeros_like(target),
            target
        )
        logits = logits.masked_fill_(logits.isnan() | logits.isinf(), 0.0)
        lprobs = torch.log_softmax(logits, -1, dtype=torch.float32)
        nll_loss = -lprobs.gather(-1, target).squeeze(-1)
        loss_token = (nll_loss * pad_mask) # 处理了pad token
        nll_loss = (nll_loss * pad_mask).sum(dim=1)
        # if self.label_smoothing > 0:
        #     eps_i = self.label_smoothing / (lprobs.shape[-1] - 1)
        #     smooth_loss = -lprobs.sum(-1)
        #     loss = (1 - self.label_smoothing - eps_i) * nll_loss + eps_i * smooth_loss
        #     loss = (loss * pad_mask).sum()
        # else:
        loss = nll_loss
        
        if log is not None:
            log["nll_loss"] = nll_loss
        
        return loss,loss_token # 启动标签平滑


    def compute_token_accuracy(self, logits, target):
        pad_mask = target.ne(self.padding_id)
        acc = (logits.argmax(-1).eq(target) * pad_mask).sum()
        return acc
    
    def record_logits(self, logits, target, log, teacher_logits=None, teacher_target=None):
        pad_mask = target.eq(self.padding_id)
        pos_target = torch.where(
            pad_mask,
            torch.zeros_like(target),
            target
        )
        target_logits = logits.gather(-1, pos_target.unsqueeze(-1)).squeeze(-1)
        non_target_logits = (logits.sum(-1) - target_logits) / (logits.shape[-1] - 1)
        target_logits = target_logits.masked_fill_(pad_mask, 0.0).sum()
        non_target_logits = non_target_logits.masked_fill_(pad_mask, 0.0).sum()
        
        log["target_logits"] = target_logits
        log["non_target_logits"] = non_target_logits

        if teacher_logits is not None:
            assert teacher_target is not None
            teacher_pad_mask = teacher_target.eq(self.padding_id)
            pos_teacher_target = torch.where(
                teacher_pad_mask,
                torch.zeros_like(teacher_target),
                teacher_target
            )
            teacher_target_logits = teacher_logits.gather(-1, pos_teacher_target.unsqueeze(-1)).squeeze(-1)
            teacher_non_target_logits = (teacher_logits.sum(-1) - teacher_target_logits) / (teacher_logits.shape[-1] - 1)
            teacher_target_logits = teacher_target_logits.masked_fill_(teacher_pad_mask, 0.0).sum()
            teacher_non_target_logits = teacher_non_target_logits.masked_fill_(teacher_pad_mask, 0.0).sum()
            
            log["teacher_target_logits"] = teacher_target_logits
            log["teacher_non_target_logits"] = teacher_non_target_logits
    
    # def record_logging_output(self, logging_output, batch_denom, content):
    #     for k, v in content.items():
    #         record_v = v / batch_denom
    #         dist.all_reduce(record_v, dist.ReduceOp.SUM)
    #         record_v = record_v.item() / dist.get_world_size()
    #         if k in logging_output:
    #             logging_output[k].append(record_v)
    #         else:
    #             logging_output[k] = [record_v]
    #     return logging_output

    def loss(self, inp, output_data=None,block_list=None):
        outputs = self.model_to_test(input_ids=inp, use_cache=False, \
                 output_attentions=False, block_list=block_list)
        logits = outputs.logits
        loss,loss_token = self.compute_cross_entropy_loss(logits, output_data["label"]) # 平滑和非平滑
        return loss,loss_token

    def find_needle_idx(self, needle):
        needle_ids = self.enc(needle, add_special_tokens=False)["input_ids"]
        #print( self.enc.decode(needle_ids, skip_special_tokens=False))
        span_len = len(needle_ids)
        for i in range(len(self.prompt_ids)):
            
            token_span = self.prompt_ids[i : i + span_len]
            span_ids = set(token_span.tolist())
            overlap = float(len(span_ids.intersection(set(needle_ids)))) / len(set(needle_ids))
            if(overlap > 0.9):
                return i, i + span_len
        return -1, -1
    def construct_random_head(self, n):
        results = []
        seed_list = [i  for i in range(32)]
        random.shuffle(seed_list)
        while len(results) < n:
            l, h = random.choices(seed_list, k=2)
            if (l, h) in results or (l, h) in self.block_list:
                continue
            else:
                results.append((l, h))
        return results
    def evaluate_and_log(self,dataset):
        # 加载掩蔽模型
        if self.mask_topk > 0:
            print(self.mask_topk)
            block_list = self.block_list[:self.mask_topk]
            # save_name = f"{self.model_version}_block_top{self.mask_topk}"
        elif self.mask_topk == 0:
            block_list = None
            # save_name = self.model_version
        else:
            block_list = self.construct_random_head(-self.mask_topk)
            # save_name = f"{self.model_version}_block_random{-self.mask_topk}"
        
        # 加载数据
        train_dataloader = DataLoader(
            dataset['train'], 
            # sampler=sampler, 
            shuffle=True,
            drop_last=True,
            batch_size=args.batch_size, 
            num_workers=args.num_workers, 
            collate_fn=dataset["train"].collate
        )
        # 开始测试
        print("start test!")
        # 控制停止
        end_epoch = False
        train_iter = iter(train_dataloader) # 创建数据迭代器
        all_output = []
        with tqdm(total=len(train_dataloader), desc="Testing") as pbar:
            while True:
                global_batch = []
                try:
                    (input_batch, output_batch, _) = next(train_iter)
                    dataset["train"].move_to_device(
                    [input_batch, output_batch], self.model_to_test.device)
                    global_batch.append({
                            "input_batch": input_batch,
                            "output_batch": output_batch,
                    })
                except StopIteration:
                        end_epoch = True
                        break
                if end_epoch:
                    break
                # tqdm更新
                pbar.update(1)
                # 取batch和停止
                for batch in global_batch: # 前线传播
                    with torch.no_grad():
                        input_ids = batch["input_batch"]["input_ids"]
                        # print(input_ids[:,-1].shape)
                        output_data = batch["output_batch"]

                        # 获取数据集需保存内容
                        prompt = batch["input_batch"]["prompt"] # 答案
                        output = batch["input_batch"]["output"] # 目标
                        text = batch["input_batch"]["text"] # 文本内容
                        # souse_len = batch["input_batch"]["input_length"] # 输入长度

                        loss_vanilla,token_loss_vanilla = self.loss(input_ids,output_data=output_data, block_list=None)
                        loss_block,token_loss_block = self.loss(input_ids,output_data=output_data, block_list=block_list)
                        
                        # loss差值 转化为列表 与原有列表组合
                        loss_dif_list = ((loss_block- loss_vanilla)/loss_vanilla)
                        loss_dif_list = torch.nan_to_num(loss_dif_list, nan=0.0).tolist()

                        token_loss_dif = ((token_loss_block - token_loss_vanilla)/token_loss_vanilla)
                        token_loss_dif = torch.nan_to_num(token_loss_dif, nan=0.0).tolist()
                        # 组合为元祖
                        loss_mask = output_data["loss_mask"].tolist()
                        combined_list = list(zip(prompt,output,text, loss_dif_list,token_loss_dif,loss_mask))

                        all_output.extend(combined_list)

        # 完成循环 标注所有数据 排序
        sorted_list = sorted(all_output, key=lambda x: x[3], reverse=True)
        # 存储到jsonl中
        json_ready_data = [
                {"prompt": item[0],"output": item[1],"text": item[2], "loss_diff": item[3],"token_loss_diff": item[4], "loss_mask": item[5]} for item in sorted_list
            ]
        os.makedirs(os.path.dirname(file_dir), exist_ok=True)
        with open(file_dir, "w", encoding="utf-8") as f:
            for entry in json_ready_data:
                    f.write(json.dumps(entry) + "\n")
            
    def result_exists(self, context_length, depth_percent):
        """
        Checks to see if a result has already been evaluated or not
        """

        results_dir = 'results/' + self.model_version
        print("Searching existing results at %s" % results_dir)
        if not os.path.exists(results_dir):
            return False
        for filename in os.listdir(results_dir):
            if filename.endswith('.json'):
                with open(os.path.join(results_dir, filename), 'r') as f:
                    result = json.load(f)
                    context_length_met = result['context_length'] == context_length
                    depth_percent_met = result['depth_percent'] == depth_percent
                    version_met = result.get('version', 1) == self.results_version
                    model_met = result['model'] == self.model_name
                    # import ipdb; ipdb.set_trace()
                    if context_length_met and depth_percent_met and version_met and model_met:
                        return True
        return False

    def generate_context(self, context_length, depth_percent):
        # Load up tiktoken so we navigate tokens more easily

        # Get your Paul Graham files loaded into a string
        context = self.read_context_files()

        # Truncate the Paul Graham essays to the context length you desire
        context = self.encode_and_trim(context, context_length)

        # Insert your random statement according to your depth percent
        context = self.insert_needle(context, depth_percent, context_length)

        return context
    
    def encode_text_to_tokens(self, text):
        if self.model_provider in ["OpenAI", "LLaMA", "Mistral", "GLM"]:
            return self.enc.encode(text)
        elif self.model_provider == "Anthropic":
            # Assuming you have a different encoder for Anthropic
            return self.enc.encode(text).ids
        else:
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")
    
    def insert_needle(self, context, depth_percent, context_length):
        tokens_needle = self.encode_text_to_tokens(self.needle)
        tokens_context = self.encode_text_to_tokens(context)

        # Reducing the context length by 150 buffer. This is to account for system message, the user question, and response.
        context_length -= self.final_context_length_buffer

        # If your context + needle are longer than the context length (which it will be), then reduce tokens from the context by the needle length
        if len(tokens_context) + len(tokens_needle) > context_length:
            tokens_context = tokens_context[:context_length - len(tokens_needle)]

        if depth_percent == 100:
            # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
            tokens_new_context = tokens_context + tokens_needle
        else:
            # Go get the position (in terms of tokens) to insert your needle
            insertion_point = int(len(tokens_context) * (depth_percent / 100))
            # import ipdb; ipdb.set_trace()

            # tokens_new_context represents the tokens before the needle
            tokens_new_context = tokens_context[:insertion_point]

            # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
            if(self.model_provider in ["LLaMA", "LongLLaMA"]): period_tokens = [29889, 869]
            elif(self.model_provider == "Mistral"): period_tokens = [842, 28723]
            elif(self.model_provider == "GLM"): period_tokens = [918, 30930]
            else: period_tokens = self.encode_text_to_tokens('.')
            
            # Then we iteration backwards until we find the first period
            while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                insertion_point -= 1
                tokens_new_context = tokens_context[:insertion_point]

            print("insertion at %d" % insertion_point)
            # Once we get there, then add in your needle, and stick the rest of your context in on the other end.
            # Now we have a needle in a haystack
            tokens_new_context += tokens_needle + tokens_context[insertion_point:]

        # Convert back to a string and return it
        new_context = self.decode_tokens(tokens_new_context)
        return new_context

    def get_context_length_in_tokens(self, context):
        if self.model_provider in ["OpenAI", "LLaMA", "Mistral", "GLM"]:
            return len(self.enc.encode(context))
        elif self.model_provider == "Anthropic":
            # Assuming you have a different encoder for Anthropic
            encoded = self.enc.encode(context)
            return len(self.enc.encode(context).ids)
        else:
            
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")

    def read_context_files(self):
        context = ""
        max_context_length = max(self.context_lengths)

        while self.get_context_length_in_tokens(context) < max_context_length:
            for file in glob.glob(f"{self.haystack_dir}/*.txt"):
                with open(file, 'r') as f:
                    context += f.read()
        return context

    def get_tokens_from_context(self, context):
        if self.model_provider in ["OpenAI", "LLaMA", "Mistral", "GLM"]:
            return self.enc.encode(context)
        elif self.model_provider == "Anthropic":
            # Assuming you have a different encoder for Anthropic
            return self.enc.encode(context).ids
        else:
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")
        
    def decode_tokens(self, tokens, context_length=None):
        if self.model_provider in ["OpenAI", "LLaMA", "Mistral", "GLM"]:
            return self.enc.decode(tokens[:context_length])
        elif self.model_provider == "Anthropic":
            # Assuming you have a different decoder for Anthropic
            return self.enc.decode(tokens[:context_length])
        else:
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")

    def encode_and_trim(self, context, context_length):
        tokens = self.get_tokens_from_context(context)
        if len(tokens) > context_length:
            context = self.decode_tokens(tokens, context_length)
        return context
    
    def get_results(self):
        return self.testing_results
    
    def print_start_test_summary(self):
        print ("\n")
        print ("Starting Needle In A Haystack Testing...")
        print (f"- Model: {self.model_name}")
        print (f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}")
        print (f"- Document Depths: {len(self.document_depth_percents)}, Min: {min(self.document_depth_percents)}%, Max: {max(self.document_depth_percents)}%")
        print (f"- Needle: {self.needle.strip()}")
        print ("\n\n")

    def start_test(self, args):
        # if self.print_ongoing_status:
        #     self.print_start_test_summary()
        #asyncio.run(self.run_test())
        self.run_test(args)


if __name__ == "__main__":
    # Tons of defaults set, check out the LLMNeedleHaystackTester's init for more info
    parser = argparse.ArgumentParser()
    # parser.add_argument('-s', '--s_len', metavar='N', type=int, help='a number')
    # parser.add_argument('-e', '--e_len', metavar='N', type=int, help='a number')d
    parser.add_argument('--model_path', type=str, default=None, help='path to model')
    parser.add_argument('--model_name', type=str, default="tinyllama", help='name of model')
    parser.add_argument('--model_name_suffix', type=str, default="tinyllama", help='name of model')
    parser.add_argument('--model_provider', type=str, default="LLaMA", help='which model to use')
    parser.add_argument('--api_key', type=str, default="", help='OpenAI API Key')
    parser.add_argument('--mask_topk', type=int, default=30, help='mask topk heads, input a negative value to mask random heads')
    parser.add_argument("--data-dir", type=str, default="/home/pkuccadm/huwenp/emb/revis_emb/Retrieval_Head/data_dir")
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument('--max-length', type=int, default=1024,
                       help='max length of input')
    parser.add_argument("--model-type", type=str, default="llama") # 需要改动
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Data Loader batch size')
    parser.add_argument("--num-workers", type=int, default=1)
    

    # parser = add_args(parser)

    args = parser.parse_args()
    file_dir = args.data_dir+"/train_weight.jsonl"
    print(args)

    if(args.model_path is not None):
        # assert(args.model_name is None)
        model_name = args.model_path
    else: 
        assert(args.model_name is not None)

    ht = LLMNeedleHaystackTester(model_name=model_name, 
                                 model_name_suffix=args.model_name_suffix,
                                 model_provider=args.model_provider,
                                 save_contexts=True,
                                 save_results=True,
                                 mask_topk=args.mask_topk,
                                # context_lengths_min=args.s_len,
                                # context_lengths_max=args.e_len,
                                 )

    ht.start_test(args)
