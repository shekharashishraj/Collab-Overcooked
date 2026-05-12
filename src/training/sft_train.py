"""Stage 1: LoRA SFT of Qwen2.5-7B-Instruct on filtered GPT-4o trajectories.

Loads the JSONL produced by extract_sft_data.py (messages format), applies the
Qwen2.5 chat template, and trains LoRA adapters with TRL's SFTTrainer.

Usage:
    python src/training/sft_train.py --config src/training/configs/sft_qwen7b.yaml
"""

import argparse
import os
from pathlib import Path

import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

REPO = Path(__file__).resolve().parents[2]


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def to_text(example, tokenizer):
    """Render messages with the model's chat template (one string per row)."""
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max_steps", type=int, default=None, help="Override for smoke tests.")
    ap.add_argument("--output_dir", type=str, default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    output_dir = args.output_dir or cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    model_id = cfg["model_name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=cfg.get("trust_remote_code", True)
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_path = str(REPO / cfg["dataset_path"])
    eval_path = str(REPO / cfg["eval_dataset_path"])
    train_ds = load_dataset("json", data_files=train_path, split="train")
    eval_ds = load_dataset("json", data_files=eval_path, split="train")

    train_ds = train_ds.map(lambda ex: to_text(ex, tokenizer), remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(lambda ex: to_text(ex, tokenizer), remove_columns=eval_ds.column_names)

    peft_cfg = None
    if cfg.get("use_peft", True):
        peft_cfg = LoraConfig(
            r=cfg["lora_r"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg.get("lora_dropout", 0.05),
            target_modules=cfg["lora_target_modules"],
            bias="none",
            task_type="CAUSAL_LM",
        )

    sft_cfg_kwargs = dict(
        output_dir=output_dir,
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        num_train_epochs=cfg["num_train_epochs"],
        logging_steps=cfg["logging_steps"],
        eval_strategy=cfg["eval_strategy"],
        eval_steps=cfg["eval_steps"],
        save_strategy=cfg["save_strategy"],
        save_steps=cfg["save_steps"],
        save_total_limit=cfg["save_total_limit"],
        bf16=cfg["bf16"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        max_seq_length=cfg["max_seq_length"],
        packing=cfg.get("packing", False),
        dataset_text_field="text",
        report_to=cfg.get("report_to", "none"),
        seed=cfg.get("seed", 17),
    )
    if args.max_steps is not None:
        sft_cfg_kwargs["max_steps"] = args.max_steps
        sft_cfg_kwargs["num_train_epochs"] = 1
    sft_config = SFTConfig(**sft_cfg_kwargs)

    trainer = SFTTrainer(
        model=model_id,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_cfg,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[sft_train] Saved LoRA adapter + tokenizer to {output_dir}")


if __name__ == "__main__":
    main()
