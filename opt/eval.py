import time
from tqdm import tqdm
from opt.metrics import Metric
from opt.request import Request
from opt.utils import extract_item_list, detect_error
import asyncio

class Eval():
    def __init__(self, config, data, text_table):
        self.conf = config
        self.requset = Request(config)  # Request 实例（request 文件中没有处理 API Key）
        self.data = data
        self.text_table = text_table
        self.error_list = []
        self.target_rank_list = []
        # 设置4个 API Key
        self.api_keys = [
            
        ]
        # 为每个 API Key 设置独立信号量，确保每个 API Key 同时只处理 1 个请求
        self.api_key_sems = { key: asyncio.Semaphore(1) for key in self.api_keys }
        # 用于轮询选择 API Key
        self.key_index = 0

    async def record_error(self, data, response):
        tmp = {}
        tmp['response'] = response
        tmp['target'] = data['target']
        tmp['input'] = data['input']
        tmp['target_index'] = data['target_index']
        return tmp

    # async def normal_eval(self, prompt):
    #     # 遍历每条数据，tqdm 用于显示进度（同步迭代器）
    #     # for data in tqdm(self.data):
    #     #     success = False
    #     #     response = None
    #     #     result_list = None
    #     #     candidate = None  # 用于保存候选数

    #     #     # 尝试最多 3 次获取有效响应
    #     #     for i in range(3):
    #     #         # 轮询选择 API Key
    #     #         api_key = self.api_keys[self.key_index]
    #     #         self.key_index = (self.key_index + 1) % len(self.api_keys)

    #     #         # 使用对应 API Key 的信号量，确保同一时刻该 API Key 只处理 1 个请求
    #     #         async with self.api_key_sems[api_key]:
    #     #             response = await self.requset.request(user=data['input'], system=prompt, api_key=api_key)

    #     #         result_list = extract_item_list(response, data['target'], data['target_index'])
                
    #     #         if not result_list:
    #     #             continue

    #     #         # 先判断 result_list 的最后一个元素是否与 data['target_index'] 不一致
    #     #         if int(result_list[-1]) != int(data['target_index']):
    #     #             candidate = int(result_list[-1])
    #     #         # 当 result_list 的最后一个元素与 data['target_index'] 相等
    #     #         elif len(result_list) == 1:
    #     #                 candidate = int(result_list[0])
    #     #             # 如果列表长度大于 1，则使用倒数第二个
    #     #         elif len(result_list) > 1:
    #     #                 candidate = int(result_list[-2])

    #     #         # if int(result_list[-1]) != int(data['target_index']):
    #     #         #     print("result_list[-1]",result_list[-1])
    #     #         #     print("tar:", data['target_index'])
    #     #         #     print("res:",response)
    #     #         #     print("name:",data['target'])



    #     #         # print("res:",response)
    #     #         # print("name:",data['target'])
    #     #         # print("result_list[-1]",result_list[-1])
    #     #         # print("tar:", data['target_index'])
    #     #         # print("num:", result_list)
    #     #         # print("candidate:",candidate)
                
    #     #         # print("\n")
    #     #         # print("\n")
    #     #         # print("\n")

    #     #         # 判断候选数是否在有效范围内（必须大于 0 且小于 self.conf['candidate_size']+1）
    #     #         if candidate is not None and 0 < candidate < self.conf['candidate_size'] + 1:
    #     #             self.target_rank_list.append(candidate)
    #     #             success = True
    #     #             break

    #     #     # 将最后一次的 response 写入表格
    #     #     self.text_table.add_data(data['input'], data['target'], response)

    #     #     # 如果未获得有效结果或结果不符合要求，则记录错误，并设定一个默认排名
    #     #     if not success:
    #     #         error = await self.record_error(data, response)
    #     #         self.error_list.append(error)
    #     #         self.target_rank_list.append(self.conf['candidate_size'] + 1)



    #     start_time = time.time()           # 整个过程起始时间
    #     group_start_time = time.time()     # 每 4 个数据的分组起始时间

    #     # 使用 enumerate 来获得索引，从 1 开始计数，方便每 4 个分组
    #     for idx, data in enumerate(tqdm(self.data), start=1):
    #         success = False
    #         response = None
    #         result_list = None
    #         candidate = None  # 用于保存候选数

    #         # 尝试最多 3 次获取有效响应
    #         for i in range(3):
    #             # 轮询选择 API Key
    #             api_key = self.api_keys[self.key_index]
    #             self.key_index = (self.key_index + 1) % len(self.api_keys)

    #             # 使用对应 API Key 的信号量，确保同一时刻该 API Key 只处理 1 个请求
    #             async with self.api_key_sems[api_key]:
    #                 response = await self.requset.request(user=data['input'], system=prompt, api_key=api_key)

    #             result_list = extract_item_list(response, data['target'], data['target_index'])
                
    #             if not result_list:

    #                 continue

    #             # 先判断 result_list 的最后一个元素是否与 data['target_index'] 不一致
    #             if int(result_list[-1]) != int(data['target_index']):
    #                 candidate = int(result_list[-1])
    #             # 当 result_list 的最后一个元素与 data['target_index'] 相等
    #             elif len(result_list) == 1:
    #                 candidate = int(result_list[0])
    #             # 如果列表长度大于 1，则使用倒数第二个
    #             elif len(result_list) > 1:
    #                 candidate = int(result_list[-2])

    #             # 判断候选数是否在有效范围内（必须大于 0 且小于 self.conf['candidate_size']+1）
    #             if candidate is not None and 0 < candidate < self.conf['candidate_size'] + 1:
    #                 self.target_rank_list.append(candidate)
    #                 success = True
    #                 break

    #         # 将最后一次的 response 写入表格
    #         self.text_table.add_data(data['input'], data['target'], response)

    #         # 如果未获得有效结果或结果不符合要求，则记录错误，并设定一个默认排名
    #         if not success:
    #             error = await self.record_error(data, response)
    #             self.error_list.append(error)
    #             self.target_rank_list.append(self.conf['candidate_size'] + 1)

    #         # 每处理完 4 个数据后统计耗时并输出
    #         if idx % 4 == 0:
    #             group_time = time.time() - group_start_time
    #             print(f"数据 {idx-3} 到 {idx} 总耗时: {group_time:.2f} 秒")
    #             group_start_time = time.time()  # 重新设置下一组的起始时间

    #     total_time = time.time() - start_time
    #     print(f"回答所有问题总耗时: {total_time:.2f} 秒")



    async def normal_eval(self, prompt):
        start_time = time.time()           # 整个过程起始时间
        group_start_time = time.time()     # 每 4 个数据一组的起始时间
        tasks = []

        # 内部定义处理单个样本的协程，不影响顶层结构
        async def process_sample(data):
            success = False
            response = None
            result_list = None
            candidate = None  # 用于保存候选数

            # 尝试最多 3 次获取有效响应
            for i in range(3):
                # 轮询选择 API Key
                api_key = self.api_keys[self.key_index]
                self.key_index = (self.key_index + 1) % len(self.api_keys)
                # 使用对应 API Key 的信号量，确保同一时刻该 API Key 只处理 1 个请求
                async with self.api_key_sems[api_key]:
                    response = await self.requset.request(user=data['input'], system=prompt, api_key=api_key)

                result_list = extract_item_list(response, data['target'], data['target_index'])
                if not result_list:
                    continue
                
                # 先判断 result_list 的最后一个元素是否与 data['target_index'] 不一致
                # candidate = int(result_list[-1])


                if int(result_list[-1]) != int(data['target_index']):
                    candidate = int(result_list[-1])
                # 当 result_list 的最后一个元素与 data['target_index'] 相等
                elif len(result_list) == 1:
                    candidate = int(result_list[0])
                # 如果列表长度大于 1，则使用倒数第二个
                elif len(result_list) > 1:
                    candidate = int(result_list[-2])
                
                

                # 判断候选数是否在有效范围内（必须大于 0 且小于 self.conf['candidate_size']+1）
                if candidate is not None and 0 < candidate < self.conf['candidate_size'] + 1:
                    self.target_rank_list.append(candidate)
                    success = True
                    break

            # 将最后一次的 response 写入表格
            self.text_table.add_data(data['input'], data['target'], response)

            # 如果未获得有效结果或结果不符合要求，则记录错误，并设定一个默认排名
            if not success:
                error = await self.record_error(data, response)
                self.error_list.append(error)
                self.target_rank_list.append(self.conf['candidate_size'] + 1)

        # 遍历 self.data，每 4 个样本并发执行
        for idx, data in enumerate(tqdm(self.data), start=1):
            tasks.append(process_sample(data))
            if idx % 4 == 0:
                # 并发等待 4 个任务完成
                await asyncio.gather(*tasks)
                tasks = []
                group_time = time.time() - group_start_time
                print(f"数据 {idx-3} 到 {idx} 总耗时: {group_time:.2f} 秒")
                group_start_time = time.time()
        # 若最后不足 4 个样本，将剩余任务一起处理
        if tasks:
            await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        print(f"回答所有问题总耗时: {total_time:.2f} 秒")




    async def run(self, prompt):
        await self.normal_eval(prompt)
        metric = Metric(self.target_rank_list, self.conf)
        result = metric.run()
        return result, self.target_rank_list, self.error_list
