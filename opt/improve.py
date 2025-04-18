import random
import asyncio
from opt.utils import detect_error, extract_edit_prompt
import sys
import time

class Improve():
    def __init__(self,
                 inferring_reasons, 
                 refining_prompts, 
                 augumenting_prompts, 
                 train_data,
                 config,
                 request_model):
        self.inferring_reasons = inferring_reasons
        self.refining_prompts = refining_prompts
        self.augumenting_prompts = augumenting_prompts
        self.train_data = train_data
        self.config = config
        self.request = request_model  # 要求此 request_model.request 为 async 函数
        self.used_data = []
        # 存储 4 个 API Key
        self.api_keys = [
            "",
            "",
            "",
            ""
        ]
        # 每个 API Key 的并发度限制为 1
        self.api_key_sems = { key: asyncio.Semaphore(1) for key in self.api_keys }
        # 用于轮询选择 API Key
        self.key_index = 0

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

    async def evaluate_collect_error(self, prompt, data):
        tasks = [self.limited_request(val['input'], prompt) for val in data]
        responses = []
        # 每次批量执行 4 个任务
        for i in range(0, len(tasks), 4):
            batch = tasks[i:i+4]
            batch_responses = await asyncio.gather(*batch)
            responses.extend(batch_responses)
        errors_list = [
            {'input': val['input'], 'output': response}
            for val, response in zip(data, responses)
            if not detect_error(response, val['target'], val['target_index'])
        ]
        return errors_list

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

    async def process_error(self, error, prompt, inferring_reasons, refining_prompts, table):
        tmp_prompt = inferring_reasons
        content = tmp_prompt.replace("$error_case$", error['input'])
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
        errors_list = await self.evaluate_collect_error(prompt, batch_data)
        
        try:
            errors_group = random.sample(errors_list, self.config['error_batch_size'])
        except Exception:
            errors_group = errors_list

        inferring_reasons = self.inferring_reasons.replace("$prompt$", prompt).replace("$num_feedbacks$", str(self.config['num_feedbacks']))
        refining_prompts = self.refining_prompts.replace("$prompt$", prompt)
        
        start_time = time.time()
        tasks = [
            self.process_error(error, prompt, inferring_reasons, refining_prompts, table)
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
