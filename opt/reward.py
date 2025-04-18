import numpy as np
from opt.utils import detect_error, extract_item_list
import asyncio

def ndcg(target_index):
    if np.log2(target_index + 1) == 0:
        res = 0
    else:
        res = 1 / np.log2(target_index + 1)
    return res

class Reward(): 
    def __init__(self, config, request_model) -> None:
        self.reward_func = config['reward_func']
        self.request = request_model  # 要求 request_model.request 为 async 函数
        # 限制 reward 计算时的并发数量为 4
        self.sem = asyncio.Semaphore(4)
        # 设置 4 个 API Key（与 Improve 类类似）
        self.api_keys = [
            "","","",""
        ]
        
        # 为每个 API Key 设置独立信号量，确保同一时刻每个 API Key 只处理1个请求
        self.api_key_sems = { key: asyncio.Semaphore(1) for key in self.api_keys }
        self.key_index = 0

    async def calculate_reward(self, prompt, sample_data):
        reward = 0

        async def limited_request_func(data):
            # 轮询选择 API Key
            api_key = self.api_keys[self.key_index]
            self.key_index = (self.key_index + 1) % len(self.api_keys)
            # 限制当前 API Key 同时只处理一个请求
            async with self.api_key_sems[api_key]:
                # 使用关键字参数传递，避免参数顺序混乱
                return await self.request.request(user=data['input'], system=prompt, api_key=api_key)

        # 对 sample_data 中的每个数据项构建任务
        async with self.sem:
            tasks = [limited_request_func(data) for data in sample_data]
            responses = await asyncio.gather(*tasks)

        # 遍历每个数据项和对应响应，计算累计 reward
        for data, response in zip(sample_data, responses):
            if detect_error(response, data['target'], data['target_index'], mode='select'):
                result_list = extract_item_list(response, data['target'], data['target_index'])

                # 获取排名
                if int(result_list[-1]) != int(data['target_index']):
                    target_index = int(result_list[-1])
                # 当 result_list 的最后一个元素与 data['target_index'] 相等
                elif len(result_list) == 1:
                    target_index = int(result_list[0])
                # 如果列表长度大于 1，则使用倒数第二个
                elif len(result_list) > 1:
                    target_index = int(result_list[-2])

                reward += ndcg(target_index)
        return reward
