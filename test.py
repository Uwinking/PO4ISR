import os
import wandb
from opt.eval import Eval
from opt.config import init_config
from opt.utils import load_eval_data
import asyncio
import random

if __name__ == '__main__':
    # test_prompt = "Based on the user's current session interactions, you need to answer the following subtasks step by step:\n" \
    #         "1. Discover combinations of items within the session, where the number of combinations can be one or more.\n" \
    #         "2. Based on the items within each combination, infer the user's interactive intent within each combination.\n" \
    #         "3. Select the intent from the inferred ones that best represents the user's current preferences.\n" \
    #         "4. Based on the selected intent, please rerank the 20 items in the candidate set according to the possibility of potential user interactions and show me your ranking results with item index.\n" \
    #         "Note that the order of all items in the candidate set must be given, and the items for ranking must be within the candidate set.\n"
    test_prompt="""You are a discerning recommendation system, adept at understanding user tastes and ordering items accordingly. Your objective is to analyze a user's current activity and generate a prioritized list of potential items. Follow this process:

1.  **Session Item Examination:** Scrutinize the items the user has engaged with during their current session. Pay close attention to specific details, including brand affiliations, product families, series, hardware platforms, and content categories. If only one item is present, focus on its core attributes. If multiple items are available, identify trends, shared characteristics, and attribute combinations.

2.  **Preference Deduction (Precision is Paramount):** Based on the session analysis, determine the user's probable preferences. Move beyond general categories and aim for highly specific inferences. For instance:
    *   If the user viewed "Super Mario Sunshine - Gamecube," infer preferences like "Gamecube console, platformer genre."
    *   If the user interacted with "Madden NFL 10 - PlayStation 3" and "NBA 2K11 - Xbox 360", infer "sports games."
    *   If the user interacted with "Nintendo Switch Carrying Case - Black" and "The Legend of Zelda: Breath of the Wild", infer "Nintendo Switch accessories," "The Legend of Zelda franchise," and potentially "games for Nintendo Switch."

3.  **Candidate Set Assessment and Ranking (Relevance-Driven):** Evaluate the 20 items in the candidate set, ranking them according to their relevance to the inferred user preferences. Prioritize items that directly correspond to the identified characteristics. Consider the following:
    *   **Exact Matches:** Items sharing the same brand, platform, franchise, or product line as the session items should receive the highest ranking.
    *   **Related Items:** Items belonging to the same genre, platform, or possessing similar features should be ranked higher than unrelated items.
    *   **Contextual Awareness:** Consider the specific context of the session items. For example, if the user is browsing a carrying case, accessories for the same device should be prioritized.

4.  **Present Ranked List:** Output the ranked list, including the item index and item name, ensuring all items in the candidate set are represented and ranked. The items for ranking must be within the candidate set. The output should be a numbered list, with the most relevant item at the top (index 1) and the least relevant at the bottom (index 20).
"""
    conf = init_config()
    test_data1 = load_eval_data(conf)

    ######xiugai
    test_data=random.sample(test_data1, 100)
    key = conf['openai_api_key']
    if conf['use_wandb']:
        wandb.login(key=conf['wandb_api_key'])
        conf.pop('openai_api_key')
        run = wandb.init(
            project=f"PO4ISR_{conf['dataset']}_test",
            config=conf,
            name=f"seed_{conf['seed']}",
        )
        text_table = wandb.Table(columns=["Input", "Target", "Response"])
    else:
        text_table = None
    conf['openai_api_key'] = key

    eval_model = Eval(conf, test_data, text_table)
    results, target_rank_list, error_list = asyncio.run(eval_model.run(test_prompt))

    result_save_path = f"./res/metric_res/{conf['dataset']}/"
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)
    results.to_csv(f"{result_save_path}seed_{conf['seed']}.csv", index=False)
    
    if conf['use_wandb']:
        run.log({"texts": text_table})

