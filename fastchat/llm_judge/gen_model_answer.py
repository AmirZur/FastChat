"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import json
import os
import random
import time

import shortuuid
import torch
from tqdm import tqdm

from fastchat.llm_judge.common import load_questions, temperature_config
from fastchat.model import load_model, get_conversation_template
from fastchat.utils import str_to_torch_dtype
from pyreft import get_intervention_locations, ReftModel
import sys
sys.path.extend(['../../../../clones/guard-reft/baselines'])
# import get_template from HarmBench
from model_utils import get_template


def run_eval(
    model_path,
    reft_model_path,
    pos_size,
    num_skip_interventions,
    model_id,
    question_file,
    question_begin,
    question_end,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    num_gpus_total,
    max_gpu_memory,
    dtype,
    revision,
):
    questions = load_questions(question_file, question_begin, question_end)
    # random shuffle the questions to balance the loading
    random.shuffle(questions)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model_path,
                reft_model_path,
                model_id,
                questions[i : i + chunk_size],
                answer_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                dtype=dtype,
                revision=revision,
                pos_size=pos_size,
                num_skip_interventions=num_skip_interventions,
            )
        )

    if use_ray:
        ray.get(ans_handles)


def get_reft_input(
    tokenizer, 
    template, 
    prompt, 
    device='cuda:0',
    pos_size=1,
    num_skip_interventions=0,
    num_interventions=1
):
    before_tc, after_tc = template.split('{instruction}')
    cache_input_ids = tokenizer([before_tc], padding=False)['input_ids'] # some tokenizer have <s> for before_tc
    cache_input_ids += tokenizer([prompt, after_tc], padding=False, add_special_tokens=False)['input_ids']
    cache_input_ids = [torch.tensor(input_ids, dtype=torch.long).unsqueeze(0) for input_ids in cache_input_ids] # make tensor separately because can't return_tensors='pt' in tokenizer
    before_ids, behavior_ids, after_ids = cache_input_ids

    input_ids = torch.cat(cache_input_ids, dim=-1)
    input_ids = input_ids.to(device)

    behavior_length = behavior_ids.shape[-1]
    intervention_locations = get_intervention_locations(
        last_position=behavior_length, 
        first_n=pos_size, 
        last_n=pos_size,
        pad_mode="last",
        num_interventions=num_interventions,
        share_weights=True,
    )
    intervention_locations = (torch.tensor(intervention_locations) + before_ids.shape[-1]).tolist()
    # batch size of 1
    unit_locations = [[i] for i in intervention_locations]

    # skip first n interventions
    for i in range(num_skip_interventions):
        unit_locations[i] = None
    
    return input_ids, unit_locations


@torch.inference_mode()
def get_model_answers(
    model_path,
    reft_model_path,
    model_id,
    questions,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    max_gpu_memory,
    dtype,
    revision,
    pos_size=1,
    num_skip_interventions=0
):
    model, tokenizer = load_model(
        model_path,
        reft_model_path,
        revision=revision,
        device="cuda",
        num_gpus=num_gpus_per_model,
        max_gpu_memory=max_gpu_memory,
        dtype=dtype,
        load_8bit=False,
        cpu_offloading=False,
        debug=False,
    )
    assert isinstance(model, ReftModel), "Only supporting REFT models for now."

    stop_str = "<|user|>"
    for question in tqdm(questions):
        if question["category"] in temperature_config:
            temperature = temperature_config[question["category"]]
        else:
            temperature = 0.7

        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            # conv = get_conversation_template(model_id)
            # set return_fschat_conv=False for zephyr-7b
            template = get_template(model_path, return_fschat_conv=False)['prompt']
            turns = []
            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                prompt = template.format(instruction=qs)
                inputs = tokenizer([prompt], return_tensors="pt").to(model.get_device())
                input_ids = inputs["input_ids"]
                last_position = input_ids.shape[-1]
                intervention_locations = get_intervention_locations(
                    last_position=last_position, 
                    first_n=pos_size, 
                    last_n=pos_size,
                    pad_mode="last",
                    num_interventions=len(model.representations),
                    share_weights=True,
                )
                # (# interventions, # batches=1, # positions)
                unit_locations = torch.tensor(intervention_locations).unsqueeze(1).tolist()
                
                if temperature < 1e-4:
                    do_sample = False
                else:
                    do_sample = True

                # some models may error out when generating long outputs
                try:
                    _, output_ids = model.generate(
                        inputs,
                        unit_locations={"sources->base": (None, unit_locations)},
                        intervene_on_prompt=True,
                        do_sample=do_sample,
                        temperature=temperature,
                        max_new_tokens=max_new_token,
                    )
                    if model.config.is_encoder_decoder:
                        output_ids = output_ids[0]
                    else:
                        output_ids = output_ids[0][len(input_ids[0]) :]

                    # be consistent with the template's stop_token_ids
                    # if conv.stop_token_ids:
                    #     stop_token_ids_index = [
                    #         i
                    #         for i, id in enumerate(output_ids)
                    #         if id in conv.stop_token_ids
                    #     ]
                    #     if len(stop_token_ids_index) > 0:
                    #         output_ids = output_ids[: stop_token_ids_index[0]]

                    output = tokenizer.decode(
                        output_ids,
                        spaces_between_special_tokens=False,
                    )
                    # if conv.stop_str and isinstance(conv.stop_str, list):
                    #     stop_str_indices = sorted(
                    #         [
                    #             output.find(stop_str)
                    #             for stop_str in conv.stop_str
                    #             if output.find(stop_str) > 0
                    #         ]
                    #     )
                    #     if len(stop_str_indices) > 0:
                    #         output = output[: stop_str_indices[0]]
                    # elif conv.stop_str and output.find(conv.stop_str) > 0:
                    #     output = output[: output.find(conv.stop_str)]
                    if stop_str and output.find(stop_str) > 0:
                        output = output[: output.find(stop_str)]

                    for special_token in tokenizer.special_tokens_map.values():
                        if isinstance(special_token, list):
                            for special_tok in special_token:
                                output = output.replace(special_tok, "")
                        else:
                            output = output.replace(special_token, "")
                    output = output.strip()

                    # if conv.name == "xgen" and output.startswith("Assistant:"):
                    #     output = output.replace("Assistant:", "", 1).strip()
                except RuntimeError as e:
                    print("ERROR question ID: ", question["question_id"])
                    output = "ERROR"

                # conv.update_last_message(output)
                turns.append(output)

            choices.append({"index": i, "turns": turns})

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--reft-model-path",
        type=str,
        required=True,
        help="The path to the REFT weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--pos-size",
        type=int,
        default=1,
        help="The number of positions to intervene on.",
    )
    parser.add_argument(
        "--num-skip-interventions",
        type=int,
        default=0,
        help="The number of interventions to skip.",
    )
    parser.add_argument(
        "--model-id", type=str, required=True, help="A custom name for the model."
    )
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU and float32 on CPU.",
        default="bfloat16",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="The model revision to load.",
    )

    args = parser.parse_args()

    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init()

    question_file = f"data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    run_eval(
        model_path=args.model_path,
        reft_model_path=args.reft_model_path,
        pos_size=args.pos_size,
        num_skip_interventions=args.num_skip_interventions,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_token=args.max_new_token,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        max_gpu_memory=args.max_gpu_memory,
        dtype=str_to_torch_dtype(args.dtype),
        revision=args.revision,
    )

    reorg_answer_file(answer_file)
