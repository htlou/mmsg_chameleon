import argparse
import json
import logging
import os
import re
import sys
from typing import Literal, Optional
from PIL import Image
from tqdm import tqdm
import torch.multiprocessing as mp

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'mmsg', 'integrations'))

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger()


def run_interleaved_generation(
    model_id: str ,
    inference_mode: Literal["text-to-interleaved-text-image"] = "text-to-interleaved-text-image",
    prompt: Optional[str] = None,
    max_new_tokens: int = 2400,
    fast: bool = False,
    model_cache_dir: Optional[str] = None,
    outputs_dir: str = ".",
    seed: Optional[int] = None,
    input_data=None,
    output_data=None,
    device="auto"
) -> str:
    import torch
    from term_image.image import from_file
    from transformers import (
        ChameleonForConditionalGeneration,
        ChameleonProcessor,
        set_seed,
    )

    from chameleon_utils import postprocess_token_sequence

    if seed is not None:
        set_seed(seed)
    torch.set_printoptions(threshold=10_000)

    if fast:
        model = ChameleonForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
            device_map=device,
            token=os.environ.get("HF_TOKEN"),
            cache_dir=model_cache_dir,
        )
    else:
        model = ChameleonForConditionalGeneration.from_pretrained(
            model_id,
            device_map=device,
            token=os.environ.get("HF_TOKEN"),
            cache_dir=model_cache_dir,
        )
    processor = ChameleonProcessor.from_pretrained(
        model_id,
        token=os.environ.get("HF_TOKEN"),
        cache_dir=model_cache_dir,
    )

    inputs = input_data
    
    outputs = []
    concise_outputs = []
    
    for piece in tqdm(inputs, desc="Generating responses"):
        # piece['image_url'] = piece['image']
        # prompt = "Generate an image according to the following instruction: {}".format(piece['prompt'])
        prompt = f"Generate an image according to the following instruction: {piece['prompt']}"
        #prompt = "BEGINNING OF CONVERSATION: USER:what is the capital of France?<image> YOUR RESPONSE SHOULD NOT CONTAIN ANY IMAGE. ASSISTANT: "
        logger.info(f"Prompt: {prompt}")
        inputs = processor(prompt, return_tensors="pt").to(
            model.device, dtype=model.dtype
        )
        logger.info("Generating response...")
        with torch.inference_mode():
            output_token_ids_batch = model.generate(
                **inputs,
                multimodal_generation_mode="image-only",
                max_new_tokens=max_new_tokens,
                do_sample=True,
            )
        logger.info("Finished generation.")
        # logger.info(output_token_ids_batch)

        output_token_ids_batch = output_token_ids_batch.to(dtype=inputs["input_ids"].dtype).detach().cpu().numpy()

        response_token_ids = output_token_ids_batch[0][len(inputs["input_ids"][0]) :]

        full_outputs_dir = os.path.abspath(outputs_dir)
        if not os.path.exists(full_outputs_dir):
            logging.info(f"Creating directory: {full_outputs_dir}")
            os.mkdir(full_outputs_dir)

        response, concise_response = postprocess_token_sequence(
            response_token_ids, model, processor, full_outputs_dir, validate=True
        )
        response['prompt'] = prompt
        response['original'] = piece
        concise_response['prompt'] = prompt
        concise_response['original'] = piece
        

        logger.info(f"Response: {response['text']}")
        for image in response["images"]:
            if "save_path" not in image:
                continue
            logger.info(f"{image['save_path'] = }")
            # terminal_image = from_file(image["save_path"])
            # terminal_image.draw()
        
        outputs.append(response)
        concise_outputs.append(concise_response)
        torch.cuda.empty_cache()

    output_data.extend(concise_outputs)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate text-only content based on prompt which can include images."
    )
    parser.add_argument(
        "-m",
        "--model_id",
        type=str,
        required=False,
        default="leloykun/Anole-7b-v0.1-hf",
        help="The model ID to use for generation. This could be a huggingface repo or a path to a local directory.",
    )
    parser.add_argument(
        "-i",
        "--inference_mode",
        choices=["text-to-image", "text-image-to-image", "multi-image-to-image", "text-to-interleaved-text-image"],
        required=False,
        default="text-to-interleaved-text-image",
        help="",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        required=False,
        default=None,
        help="The prompt for generation. Will be appended by <image> or <image><image> if images are provided.",
    )
    parser.add_argument(
        "-i1",
        "--image_1_path",
        type=str,
        required=False,
        default=None,
        help="The path to the first image to be used for generation.",
    )
    parser.add_argument(
        "-n",
        "--max_new_tokens",
        type=int,
        required=False,
        default=4096,
        help="The maximum number of tokens to generate.",
    )
    parser.add_argument(
        "-f",
        "--fast",
        type=int,
        required=False,
        default=False,
        help="Whether to convert the model to bfloat16 & use Flash Attention 2",
    )
    parser.add_argument(
        "-c",
        "--model_cache_dir",
        type=str,
        required=False,
        default=None,
        help="The directory to cache the model in.",
    )
    parser.add_argument(
        "-o",
        "--outputs_dir",
        type=str,
        required=False,
        default=".",
        help="The directory to save the generated images in.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        required=False,
        default=42,
        help="The seed to use for generation.",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=False,
        default='input.json',
        help="The path to the input file.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=False,
        default='output.json',
        help="The path to the output file.",
    )
    args: argparse.Namespace = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    logger.info(f"Running interleaved generation... {args = }")
    mp.set_start_method('spawn')
    
    input_path = args.input_file
    output_path = args.output_file
    
    with open(input_path, 'r') as f:
        input_data = json.load(f)
    
    output_data = mp.Manager().list()
    
    num_processes = 8
    num_gpus = 8
    
    chunks = [input_data[i::num_processes] for i in range(num_processes)]
    
    processes = []
    
    for i in range(num_processes):
        device = f"cuda:{i%num_gpus}"
        p = mp.Process(target=run_interleaved_generation, args=(args.model_id, args.inference_mode, args.prompt, args.max_new_tokens, args.fast, args.model_cache_dir, args.outputs_dir, args.seed, chunks[i], output_data, device))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    with open(output_path, 'w') as f:
        json.dump(list(output_data), f, indent=4)

    
if __name__ == "__main__":
    main()
    
