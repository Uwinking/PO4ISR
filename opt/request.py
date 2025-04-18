import openai
import random
import time
import asyncio
import sys

class Request():
    def __init__(self, config):
        self.conifg = config
        
        openai.api_base = "http://154.9.228.223:5000/v1"  # 替换为你的 base_url，例如 "https://xiaoai.plus"
        # openai.api_base = "https://chatanywhere.tech/v1"
        # model = AutoModelForCausalLM.from_pretrained(
        #     "/data/lzx/tuijian/Qwen2-7B",
        #     torch_dtype="auto",
        #     device_map="auto"
        # )
        # tokenizer = AutoTokenizer.from_pretrained("/data/lzx/tuijian/Qwen2-7B")
    async def request(self, user, system=None, message=None,api_key=None):

        response = await self.openai_request(user,system, message,api_key=api_key)

        return response
    
    async def openai_request(self, user,system=None, message=None,api_key=None):
        '''
        fix openai communicating error
        https://community.openai.com/t/openai-error-serviceunavailableerror-the-server-is-overloaded-or-not-ready-yet/32670/19
        '''
        openai.api_key = api_key
        if system:
            message=[{"role":"system", "content":system}, {"role": "user", "content": user}]
        else:
            content = system + user
            message=[{"role": "user", "content": content}]




        model = "gemini-2.0-flash-lite-preview"
        for delay_secs in (2**x for x in range(0, 10)):
            try:
                response = await asyncio.to_thread(
                    openai.ChatCompletion.create,
                    model=model,
                    messages=message,
                    temperature=0.2,
                    frequency_penalty=0.0
                )
                break
            except openai.OpenAIError as e:
                randomness_collision_avoidance = random.randint(0, 1000) / 1000.0
                sleep_dur = delay_secs + randomness_collision_avoidance
                print(f"Error: {e}. Retrying in {round(sleep_dur, 2)} seconds.")
                await asyncio.sleep(sleep_dur)
                continue

        return response["choices"][0]["message"]["content"]
