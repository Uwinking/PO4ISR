import random
import time
import openai
import re
import json
import sys


def extract_item_list(response, target,target_index):
    try:
        response = response.replace(" ", " ")
        target = target.replace("  ", " ").replace("&amp;", "and").replace("&reg;","")
        
        index = response.rfind(target)
        if index != -1:
            preceding_text = response[:index].strip()
            numbers = re.findall(r'\d+', preceding_text)
            
            if numbers:
                #####
                result_list = numbers
            else:
                result_list = []
        else:
            result_list = []
            print("response:",response)
            print("target:",target)
            print("target_index",target_index)
            print("not find")
            print('\n')
    except:
        result_list = []
    return result_list


def detect_error(response, target,target_index, mode='improve'):
    
    result_list = extract_item_list(response, target,target_index)
    if not result_list:
        return False
    else:
        if mode == 'improve':
            threshold = 10
            
            if int(result_list[-1]) != int(target_index):
                rank = int(result_list[-1])
            # 当 result_list 的最后一个元素与 data['target_index'] 相等
            elif len(result_list) == 1:
                rank = int(result_list[0])
            # 如果列表长度大于 1，则使用倒数第二个
            elif len(result_list) > 1:
                rank = int(result_list[-2])


            if rank >= threshold:
                return False
            else:
                return True
        elif mode == 'select':
            return True

def extract_edit_prompt(response):
    pattern = r'<START>\s*(.*?)\s*<END>'
    result_list = re.findall(pattern, response, re.DOTALL)
    if len(result_list) == 0:
        pattern = r'<START>(.*?)<END>'
        result_list = re.findall(pattern, response, re.DOTALL)
    return result_list 

def load_eval_data(config):
    with open(f"{config['data_path']}{config['dataset']}/ID/test_seed_{config['seed']}.json", 'r') as json_file:
        test_data = json.load(json_file)
    return test_data


