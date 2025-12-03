"""
Evaluate Qwen 0.5B on:
- TriviaQA
- HellaSwag
- GSM8K
- NaturalQuestions
- SimpleQA (custom task YAML, to be done)

Requires:
    pip install lm-eval transformers accelerate datasets
"""

import json
from pathlib import Path
import numpy as np
import lm_eval
from lm_eval.tasks import TaskManager




def _to_serializable(obj):
    # NumPy scalar (np.float32, np.int64, etc.)
    if isinstance(obj, np.generic):
        return obj.item()
    # NumPy dtype objects
    if isinstance(obj, np.dtype):
        return str(obj)
    # Anything else that json doesn't know how to handle
    return str(obj)

def main():
    
    #  Define model + tasks
    
    qwen_model_id = "Qwen/Qwen2.5-0.5B-Instruct"

    
    
    model_name = "hf"
    model_args = (
        f"pretrained={qwen_model_id},"
        "dtype=bfloat16,"
        "trust_remote_code=True,"
        "device_map=auto"
    )

    # Task names in lm-eval directory
    
    
    tasks = [
        "triviaqa",           # or "triviaqa_wiki" / variant
        "hellaswag",
        "gsm8k",
        "nq_open",  #
        #"simpleqa",           # custom YAML needs to be defined
    ]

    override_args = {
        "triviaqa": {
            "generation_kwargs": {
                "max_new_tokens": 64,
                "do_sample": False,
            }
        },
        "hellaswag": {
            "generation_kwargs": {
                "max_new_tokens": 64,
                "do_sample": False,
            }
        },
        # you can add others if they are generate_until tasks
        # "nq_open": {...}
    }


    #Tasks
  
    #tasks_dir = Path("tasks")  
    #if tasks_dir.exists():
    #    task_manager = TaskManager(include_path=str(tasks_dir))
    #else:
    task_manager = TaskManager()

    # Run evaluation
    
    results = lm_eval.simple_evaluate(
        model=model_name,
        model_args=model_args,
        tasks=tasks,
        gen_kwargs={"max_new_tokens": 64},
        num_fewshot=0,        
        batch_size=4,        
        task_manager=task_manager,
        limit=1000,           
        #override_args = override_args
        #output_path="eval_outputs/qwen0_5b_base",
    )

   
    # Save  JSON su
    
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "qwen0_5b_base.json"

    with out_path.open("w") as f:
        json.dump(results, f, indent=2, default=_to_serializable)

    print(f"Saved raw results to: {out_path}")

   #print metrics
    metrics = results["results"]
    print("\n=== Headline metrics (Qwen 0.5B base) ===")
    for task_name in tasks:
        if task_name not in metrics:
            print(f"{task_name:20s}  (no results found – check task name)")
            continue

        task_metrics = metrics[task_name]
        
        if "acc" in task_metrics:
            m_name = "acc"
        elif "accuracy" in task_metrics:
            m_name = "accuracy"
        elif "f1" in task_metrics:
            m_name = "f1"
        elif "em" in task_metrics:
            m_name = "em"
        else:
            
            m_name = next(iter(task_metrics.keys()))

        value = task_metrics[m_name]
        print(task_name, " ", m_name, " ", value)


if __name__ == "__main__":
    main()
