import random
import asyncio
from opt.utils import detect_error, extract_edit_prompt
import sys
import time
import re
import numpy as np
import json
import openai
from openai.error import APIConnectionError

class Improve():
    def __init__(self,
                 inferring_reasons, 
                 inferring_reasons1,
                 refining_prompts, 
                 augumenting_prompts, 
                 train_data,
                 config,
                 request_model):
        self.inferring_reasons = inferring_reasons
        self.inferring_reasons1= inferring_reasons1
        self.refining_prompts = refining_prompts

        self.augumenting_prompts = augumenting_prompts
        self.train_data = train_data
        self.config = config
        self.request = request_model  # 要求此 request_model.request 为 async 函数
        self.used_data = []
        # 存储 4 个 API Key
        self.api_keys = [
           
        ]
        # 每个 API Key 的并发度限制为 1
        self.api_key_sems = { key: asyncio.Semaphore(1) for key in self.api_keys }
        # 用于轮询选择 API Key
        self.key_index = 0
        self.INTENT_KEYS = [
            r"the\s*most\s*relevant\s*intent",
            r"selected\s*intent",
            r"final\s*selected\s*intent",
            r"final\s*intent",
        ]

        self.mem_embeddings = np.load("/data/lzx/tuijian/PO4ISR-main/rag/fashion_memory_embeddings.npy")  # shape=(2566,3072)
        with open("/data/lzx/tuijian/PO4ISR-main/rag/fashion_memory_prompts.json", "r", encoding="utf-8") as f:
            self.mem_prompts = json.load(f)  # list of {"index": idx, "prompt": "..."}
        # 归一化一次，提高后续检索速度
        self.mem_emb_norm = self.mem_embeddings / np.linalg.norm(self.mem_embeddings, axis=1, keepdims=True)

        # 提示词模板
        self.prompt_template = (
            "Long-term history: {long_items}. "
            "Short-term history: {short_items}."
        )

    


    

# 假设仍放在类中
    async def extract_most_relevant_intent(self, text: str) -> str:
        """
        提取 <INTENT> … </INTENT> 内的句子；
        若不存在该标签，则回退到旧版关键字匹配。
        返回空串表示未找到。
        """

        text = text.strip()

        # ========== ① 新格式：<INTENT> … </INTENT> ==========
        tag_match = re.search(r"<INTENT>\s*(.*?)\s*</INTENT>", text, flags=re.S | re.I)
        if tag_match:
            intent = tag_match.group(1).strip(" .\n\r")
            return intent

        # ========== ② 旧格式回退：Key: value ==========
        for key in self.INTENT_KEYS:                           # 例 the\s*most\s*relevant\s*intent
            patt = rf"(?i)[\*_\-\#\s]*{key}[\*_\-\s]*[:：]\s*(.+)"
            m = re.search(patt, text)
            if m:
                intent_line = m.group(1)
                # 剥掉开头装饰符
                intent_line = intent_line.lstrip("*-_# ").lstrip()
                # 截断到行尾或列表符
                intent_line = re.split(r"[\n\r]| {2,}|[\-\*]\s|\d+\.", intent_line)[0]
                return intent_line.strip(" .")

        # ========== ③ 旧格式回退：标题行在上一行 ==========
        for key in self.INTENT_KEYS:
            patt = rf"(?im)^[>\-\*\#\s]*{key}\s*$"
            m = re.search(patt, text)
            if m:
                rest  = text[m.end():].lstrip()
                first = rest.splitlines()[0].strip()
                if first and not re.match(r"(?i).*intent", first):
                    return first.strip(" .")
        return "TheMostRelevantIntent was not found"
    


    async def do_request(self, user, system=None, message=None, api_key=None):
        # 调用传入的 request_model.request 方法
        response = await self.request.request(user, system=system, message=message, api_key=api_key)
        return response

    async def limited_request(self, user, system=None, message=None):
        # 轮询选择 API Key
        api_key = self.api_keys[self.key_index]
        self.key_index = (self.key_index + 1) % len(self.api_keys)
        async with self.api_key_sems[api_key]:
            return await self.do_request(user, system=system, message=message, api_key=api_key)





    async def get_embedding(self,text: str, api_key=None):
            openai.api_key = api_key
            openai.api_base = "http://154.9.228.223:5000/v1"
            model="text-embedding-3-large"
            retries=5
            text = text.replace("\n", " ")
            for i in range(retries):
                try:
                    resp = openai.Embedding.create(model=model, input=text)
                    return resp["data"][0]["embedding"]
                except APIConnectionError as e:
                    print(f"[Embedding] 连接错误，重试第 {i+1} 次：{e}")
                    time.sleep(2 ** i)  # 指数退避
            raise RuntimeError("多次尝试后仍无法获取 embedding")
    async def get_embedding2(self, text, api_key=None):
        # 调用传入的 request_model.request 方法
        response = await self.get_embedding(text, api_key=api_key)
        return response
    async def get_embedding1(self, text):
        api_key = self.api_keys[self.key_index]
        self.key_index = (self.key_index + 1) % len(self.api_keys)
        async with self.api_key_sems[api_key]:
            return await self.get_embedding2(text, api_key=api_key)
        

    async def evaluate_collect_error(self, prompt2, data, inferring_reasons):
        """
        仅返回“经过二次 <RERANK> 仍排序错误”的样本列表。
        每条元素格式:
            {
                'input'     : 原始输入，
                'output'    : 第二次回答（仍错误），
                'target'    : 目标商品，
                'preference': TheMostRelevantIntent（1st 提取，可能为空）
            }
        """

        prompt1 = """                Given the target user's own filtered complete interaction sequence (long-term) and recent interactions (short-term), retrieve the top 2 users with the highest behavioral similarity from all users.
                        user 1:$user 1$
                        user 2:$user 2$
                        Note:The above search is just a reference and is not necessarily required.

                """
        
        first_tasks = []
        for d in data:
            prompt_full=prompt1+prompt2
            #---------- a. 构造样本 embedding ----------
            long_items  = ", ".join(d["long_items"])
            short_items = ", ".join(d["short_items"])
            sample_text = self.prompt_template.format(long_items=long_items, short_items=short_items)

            emb_list = await self.get_embedding1(sample_text)       # <-- await 这里
            emb = np.array(emb_list, dtype="float32")
            emb_norm = emb / np.linalg.norm(emb)

            # ---------- b. 与记忆库计算相似度 ----------
            sims = self.mem_emb_norm.dot(emb_norm)                              # (2566,)

            # 剔除最高相似（可能是噪声或自身）
            sims[np.argmax(sims)] = -np.inf

            # 取次高的两个
            top2_idx = sims.argsort()[-2:][::-1]                                # [idx1, idx2]

            user1_prompt = self.mem_prompts[int(top2_idx[0])]["prompt"]
            user2_prompt = self.mem_prompts[int(top2_idx[1])]["prompt"]

            # ---------- c. 组装完整 prompt ----------
            prompt_full = prompt1.replace("$user 1$",user1_prompt) + prompt2.replace("$user 2$",user2_prompt)
            

            # ---------- d. 加入异步任务 ----------
            first_tasks.append(self.limited_request(d["input"], prompt_full))
            # first_tasks.append(self.limited_request(d["input"], prompt2))

        # ---------- 2) 批量并发请求 ----------
        first_resps = []
        for i in range(0, len(first_tasks), 4):
            first_resps += await asyncio.gather(*first_tasks[i : i + 4])

        # ---------- 2) 对首轮出错样本组装二次 Prompt ----------
        second_tasks, meta_cache = [], []
        errors_list1 = []
        for d, resp1 in zip(data, first_resps):
            if not detect_error(resp1, d["target"],d["target_index"]):
                pref = await self.extract_most_relevant_intent(resp1)
                
                
                
                content = (
                    inferring_reasons
                    .replace("$error_case$",  d["input"])
                    .replace("$preference$", pref)
                    .replace("$target$",     d["target"])
                )

                errors_list1.append(
                    {
                        "input":      d["input"],
                        "output":     resp1,               # 仅保留第二次回答
                        "target":     d["target"],
                        "preference": pref,
                    }
                )

                second_tasks.append(self.limited_request(content, ""))
                meta_cache.append((d, pref))
                
        

        

        if not second_tasks:            # 没有首轮错误→直接返回空
            return []


        ##############
        print("firlen",len(second_tasks))





        # ---------- 3) 并发取二次回答 ----------
        second_resps = []
        for i in range(0, len(second_tasks), 4):
            second_resps += await asyncio.gather(*second_tasks[i : i + 4])

        # ---------- 4) 只记录“二次仍错误”的样本 ----------
        errors_list2 = []
        for (d, pref), resp2 in zip(meta_cache, second_resps):
            if not detect_error(resp2, d["target"],d["target_index"]):       # 仍未把 target 排到阈值内
                errors_list2.append(
                    {
                        "input":      d["input"],
                        "output":     resp2,               # 仅保留第二次回答
                        "target":     d["target"],
                        "preference": pref,
                    }
                )
        

        #################
        

        print("errorlen",len(errors_list2))
        return errors_list1,errors_list2



    async def generate_similar_prompt(self, prompt_list):
        tasks = []
        for prompt in prompt_list:
            tmp = self.augumenting_prompts
            content = tmp.replace("$refined_prompt$", prompt)
            for i in range(self.config['addition_sample']):
                tasks.append(self.limited_request(user=content, system=''))
        similar_prompts = []
        # 每次批量执行 4 个任务
        for i in range(0, len(tasks), 4):
            batch = tasks[i:i+4]
            batch_results = await asyncio.gather(*batch)
            similar_prompts.extend(batch_results)
        return similar_prompts

    async def process_error(self, error, prompt, inferring_reasons1, refining_prompts, table):
        tmp_prompt = inferring_reasons1
        content = tmp_prompt.replace("$error_case$", error['input']).replace("$preference$", error["preference"])
        gradient = await self.limited_request(user=content, system='')

        tmp_prompt = refining_prompts.replace("$error_case$", error['input'])


        content = tmp_prompt.replace("$reasons$", gradient)
        
        edit_prompt = await self.limited_request(user=content, system='')

        edit_prompt_list = extract_edit_prompt(edit_prompt)
        
        similar_prompts = await self.generate_similar_prompt(edit_prompt_list)
        
        candidate_prompts = []
        
        candidate_prompts.extend(edit_prompt_list)
        candidate_prompts.extend(similar_prompts)
        
        if self.config['use_wandb'] and table is not None:
            for new_index, new_prompt in enumerate(edit_prompt_list):
                for mc_index in range(self.config['addition_sample']):
                    index = new_index * self.config['addition_sample'] + mc_index
                    if index < len(similar_prompts):
                        table.add_data(
                            error['input'],
                            prompt,
                            gradient,
                            new_prompt,
                            similar_prompts[index]
                        )
        
        return candidate_prompts

    async def run(self, prompt, table=None):
        candidate_prompts = []
        batch_data = random.sample(self.train_data, self.config['batch_size'])
        self.used_data += batch_data
        inferring_reasons = self.inferring_reasons.replace("$prompt$", prompt).replace("$num_feedbacks$", str(self.config['num_feedbacks']))
        inferring_reasons1 = self.inferring_reasons1.replace("$prompt$", prompt).replace("$num_feedbacks$", str(self.config['num_feedbacks']))
        



        errors_list1,errors_list2 = await self.evaluate_collect_error(prompt, batch_data,inferring_reasons)
        
        errors_list=errors_list1 + errors_list2
        print("errors_listlen:",len(errors_list))
        try:
            errors_group = random.sample(errors_list, self.config['error_batch_size'])
        except Exception:
            errors_group = errors_list

        
        start_time = time.time()


        if len(errors_list2) == 0:
            # 假设 edit_prompt_list 已定义或从配置读取
            edit_prompt_list = []  
            edit_prompt_list.append(prompt)
            similar_prompts = await self.generate_similar_prompt(edit_prompt_list)
            # 将相似 prompts 和原 prompt 全部加入候选
            candidate_prompts.extend(similar_prompts)
            similar_prompts1 = await self.generate_similar_prompt(edit_prompt_list)
            candidate_prompts.extend(similar_prompts1)
            candidate_prompts.append(prompt)
        else:
            refining_prompts = self.refining_prompts.replace("$prompt$", prompt)
            tasks = [
                self.process_error(error, prompt, inferring_reasons1, refining_prompts, table)
                for error in errors_group
            ]
            
            results = []
            # 每次并发处理 4 个任务
            for i in range(0, len(tasks), 4):
                batch = tasks[i:i+4]
                batch_results = await asyncio.gather(*batch)

                results.extend(batch_results)
            for candidate in results:
                candidate_prompts.extend(candidate)
        
        elapsed_time = time.time() - start_time
        print(f"Total request time: {elapsed_time:.2f} seconds")
        try:
            sample_candidate_prompts = random.sample(candidate_prompts, self.config['num_candidates'])
        except Exception:
            sample_candidate_prompts = candidate_prompts
            
        return sample_candidate_prompts
    
    def get_used_data(self):
        return self.used_data
