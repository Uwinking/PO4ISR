import random
import wandb
import json
import random
import asyncio
import nest_asyncio

from tqdm import tqdm
from opt.config import init_config
from opt.request import Request
from opt.reward import Reward
from opt.improve import Improve
from opt.select import Select
import sys


async def generate_argmax_prompt(beam_candidate, val_data, reward_model, result_table):
    
    # sample_data=val_data  
    sample_data = random.sample(val_data, 50)


    # 创建一个任务列表，针对每个 prompt 调用 calculate_reward
    tasks = [reward_model.calculate_reward(prompt, sample_data) for prompt in beam_candidate]
    # 并行等待所有奖励计算完成
    reward_list = await asyncio.gather(*tasks)
    
    # 将每个 prompt 与对应的奖励记录到 result_table 中
    for prompt, reward in zip(beam_candidate, reward_list):
        result_table.add_data(prompt, reward)
    
    # 选出奖励最高的 prompt
    prompt_index = reward_list.index(max(reward_list))
    return beam_candidate[prompt_index]



if __name__ == '__main__':
    nest_asyncio.apply()
    initial_prompt = """For a long-sequence recommendation task, first filter the user’s entire interaction sequence to remove noise (keeping only stable preference signals), then define the recent sequence as the most recent interactions and treat it as the primary signal. Proceed as follows:

                        1. Short-sequence intent mining  
                            From the recent sequence, group semantically related items and distill their core intents. Output ShortIntentSummary in plain English.

                        2. Long-sequence intent mining (including combination discovery and intent inference)  
                            In the filtered full interaction sequence:  
                            a. Discover one or more meaningful item combinations, prioritizing those that include recent-sequence items.  
                            b. For each combination, infer the user’s interactive intent.  
                            c. From all inferred intents, select the single intent that best represents the user’s enduring preferences. Output LongIntentSummary.

                        3. Intent fusion and selection  
                            Compare ShortIntentSummary and LongIntentSummary, and choose TheMostRelevantIntent that best reflects the user’s current need.
                            Output exactly one line in the following format:
                            <INTENT> the most relevant intent sentence here </INTENT>
                        
                        4. Candidate reranking  
                            • Using TheMostRelevantIntent, evaluate and rerank all 20 candidate items from most-likely to least-likely to satisfy the user’s current need.  
                            • Output FinalRanking in *exactly* the following wrapper ,`item name` must be verbatim the same as the title in the candidate set and don’t add extra text:
                            <FinalRanking>
                            1 :"candidate item name"
                            2 :"candidate item name"
                            …
                            20 :"candidate item name"
                            </FinalRanking>
                    """
    
    inferring_reasons = """
                          I'm trying to write a zero-shot recommender prompt.
                          My current prompt is "$prompt$".

                          Below is an example where the ranking is poor:
                          $error_case$

                          The user preference summary extracted from long-& short-term history is:
                          $preference$
                          The target item we care about is:
                          $target$

                          Step 1  User-item interest matching
                          - List possible reasons **the user WOULD like** the target item.
                          - List possible reasons **the user WOULD NOT like** the target item.
                          Use the exact format:
                          <LIKE> reason-1; reason-2; … </LIKE>
                          <DISLIKE> reason-1; reason-2; … </DISLIKE>

                          Step 2  Error reflection
                          Give $num_feedbacks$ concrete reasons why the current prompt may have ranked this example incorrectly.
                          Wrap each reason with <START> and <END>.
                        
                          Step 3  Corrected candidate reranking  
                          - Even if the short- or long-sequence lists are empty, you must still produce a complete 20-item ranking.  
                          - Using the insights from Step 1 and Step 2 (or, if no insights are available, rely on the general relevance of the items), create a new ranking of **every** candidate.  
                          - List items from most → least likely to satisfy the user’s current intent.  
                          - For each entry, output “latest-rank item-name” exactly as it appears in the candidate set; each index must appear **once and only once**.  

                            Return the result in exactly this wrapper—no extra text before or after.

                        """
    
    inferring_reasons1 = "I'm trying to write a zero-shot recommender prompt.\n"\
                        "My current prompt is \"$prompt$\"\n"\
                        "But this prompt gets the following example wrong: $error_case$ "\
                        "give $num_feedbacks$ reasons why the prompt could have gotten this example wrong.\n"\
                        "Wrap each reason with <START> and <END>"
    
    refining_prompts = "I'm trying to write a zero-shot recommender prompt.\n"\
                       "My current prompt is \"$prompt$\"\n"\
                       "But this prompt gets the following example wrong: $error_case$\n"\
                       "Based on these example the problem with this prompt is that $reasons$.\n"\
                       "Based on the above information, please wrote one improved prompt.\n"\
                       "The prompt is wrapped with <START> and <END>.\n"\
                       "The new prompt is:"

    augumenting_prompts = "Generate a variation of the following instruction while keeping the semantic meaning.\n"\
                          "Input: $refined_prompt$\n"\
                          "Output:"

    conf = init_config()
    conf['initial_prompt'] = initial_prompt
    conf['inferring_reasons'] = inferring_reasons
    conf['refining_prompts'] = refining_prompts
    conf['augumenting_prompts'] = augumenting_prompts

    opt_request = Request(conf)
    if conf['use_wandb']:
        wandb.login(key=conf['wandb_api_key'])
        conf.pop('openai_api_key')
        run = wandb.init(
            project=f"PO4ISR_{conf['dataset']}_tune",
            config=conf,
        )
        text_table = wandb.Table(columns=["Input", "Prompt", "Reason", "Improved prompt", "Augumented prompt"])
        reward_table = wandb.Table(columns=["Prompt", "Reward"])
    else:
        text_table = None
    print("parameter initialization is comzplete")


    #修改了地址
    with open(f"../Dataset/{conf['dataset']}/Text/train_{conf['train_num']}.json", 'r') as json_file:
        train_data = json.load(json_file)
    with open(f"../Dataset/{conf['dataset']}/Text/valid.json", 'r') as json_file:
        val_data = json.load(json_file)

    beam_candidate = []
    prompt_candidate = []
    random.seed(conf['seed'])

    opt_reward = Reward(conf, opt_request)
    opt_improve = Improve(inferring_reasons,inferring_reasons1, refining_prompts, augumenting_prompts, train_data, conf, opt_request)
    opt_select = Select(train_data, conf, opt_reward)

    print("==============")
    print("The apo algorithm is running...")
    print("==============")
    beam_candidate.append(initial_prompt)
    pbar = tqdm(range(conf['search_depth']))
    for i in pbar:
        pbar.set_description("search_depth " + str(i+1))
        prompt_candidate = []
        all_pairs = []
        for prompt in beam_candidate:
            # Expand
            prompt_candidate = asyncio.run(opt_improve.run(prompt, text_table))
            # Select
            pairs = opt_select.run(prompt_candidate)
            all_pairs.extend(pairs)
            all_pairs.sort(key=lambda x: x[0], reverse=True)
        all_pairs.sort(key=lambda x: x[0], reverse=True)
        # 从中取出前 beam_width 个 prompt 作为下轮的 beam_candidate
        beam_width = min(conf['beam_width'], len(all_pairs))
        beam_candidate = [pair[1] for pair in all_pairs[:beam_width]]

        

    all_pairs.sort(key=lambda x: x[0], reverse=True)   # 按 score 降序
    beam_width=min(conf['beam_width'],len(all_pairs))
    beam_candidate = [p[1] for p in all_pairs[:beam_width]]
    
    pbar.close()
    # Argmax prompt
    new_prompt = generate_argmax_prompt(beam_candidate, val_data, opt_reward, reward_table)
    loop = asyncio.get_event_loop()
    new_prompt_str = loop.run_until_complete(new_prompt)
    
    if conf['use_wandb']:
        text_table.add_data(" ", prompt, " ", " ", new_prompt_str)
        run.log({"texts": text_table})
        run.log({"rewards": reward_table})
    print("Optimize finished")
