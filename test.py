import os
import wandb
from opt.eval import Eval
from opt.config import init_config
from opt.utils import load_eval_data
import asyncio
import random

if __name__ == '__main__':
    test_prompt = """For a long-sequence recommendation task, first filter the user’s entire interaction sequence to remove noise (keeping only stable preference signals), then define the recent sequence as the most recent interactions and treat it as the primary signal. Proceed as follows:

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
#     test_prompt="""Here's a variation of the instructions, maintaining the core meaning but using slightly different phrasing:

# Input: For long-sequence recommendations, we'll first clean the user's interaction history to isolate stable preferences. Then, we'll focus on the recent interactions as the primary signal. Here's the process:

# 1.  **Attribute Enrichment (Data Preparation):**
#     *   For every item in both the recent and filtered long sequences, gather *all* relevant attributes. These include:
#         *   **Product Type:** (e.g., "vest," "pants," "dress")
#         *   **Fashion Style:** (e.g., "casual," "bohemian," "formal," "trendy")
#         *   **Fabric Composition:** (e.g., "denim," "cotton," "lace," "chiffon")
#         *   **Key Features:** (e.g., "sleeveless," "plus size," "striped," "floral," "lace-up")
#         *   **Color/Pattern:** (e.g., "blue," "black," "striped," "floral")
#         *   **Specific Design Elements:** (e.g., "bootcut," "halter," "pleated")
#     *   Represent each item as a collection of these extracted attributes.

# 2.  **Short-Term Intent Analysis:**
#     *   Examine the recent sequence (most recent interactions), leveraging the attribute sets from Step 1.
#     *   Determine the central intent(s) revealed by the items.
#     *   Generate a concise ShortIntentSummary in a single sentence, highlighting the *specific* product type, style, fabric, and key features. Be as descriptive as possible.

# 3.  **Long-Term Intent Analysis (if applicable):**
#     *   Analyze the filtered full interaction sequence (if it extends beyond the recent sequence), using the attribute sets from Step 1.
#     *   Identify significant item groupings and their associated intents, *giving preference to combinations that include items from the recent sequence*.
#     *   Generate a concise LongIntentSummary in a single sentence, capturing the user's enduring preferences, if any. Include product type, style, fabric, and key features. If the long sequence is identical to the short sequence, or if no persistent preferences are evident, output "No discernible long-term intent."

# 4.  **Intent Consolidation and Selection:**
#     *   If the LongIntentSummary is "No discernible long-term intent", adopt the ShortIntentSummary as TheMostRelevantIntent.
#     *   Otherwise, compare the ShortIntentSummary and LongIntentSummary, and select TheMostRelevantIntent that best aligns with the user's current needs.
#     *   Output exactly one line in the following format:
#         `<INTENT> the most relevant intent sentence here </INTENT>`

# 5.  **Candidate Reordering:**
#     *   Using TheMostRelevantIntent, assess and reorder all 20 candidate items from most to least likely to satisfy the user's current preferences.
#     *   Consider the attributes of the candidate items (extracted in Step 1 implicitly).
#     *   Output FinalRanking in *exactly* the following wrapper, `item name` must be verbatim the same as the title in the candidate set and don’t add extra text:
#         `<FinalRanking>
#         1 :"candidate item name"
#         2 :"candidate item name"
#         …
#         20 :"candidate item name"
#         </FinalRanking>`
# """
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

