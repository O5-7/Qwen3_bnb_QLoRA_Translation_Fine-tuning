import random
import warnings
import time

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Failed to load image Python extension.*")
warnings.filterwarnings("ignore", message="torch.utils.checkpoint: the use_reentrant.*")

from tqdm import tqdm

from os.path import join
import torch

# from torch.cuda.amp import autocast, GradScaler
from transformers import AutoModelForCausalLM, AutoTokenizer, get_wsd_schedule
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from torch.utils.data import Dataset, DataLoader, Sampler
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.optim import AdamW
from bitsandbytes.optim import AdamW8bit as ADAM
from transformers import BitsAndBytesConfig

"""<|im_start|>user
请介绍一下自己。<|im_end|>
<|im_start|>assistant
<think>

</think>

翻译：我是一个智能助手，可以为您提供帮助和支持。如果您有任何问题或需要帮助，请随时告诉我！<|im_end|>"""


class Q3_data(Dataset):
    def __init__(
        self,
        file_name: str,
        tokenizer: Qwen2TokenizerFast,
        test_limit=False,
        token_limit=512,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt_list = []
        with open(join("./translation_dataset", file_name), "r", encoding="utf-8") as f:
            for prompt in tqdm(
                f.readlines()[:1000] if test_limit else f.readlines(), ncols=120
            ):
                prompt = prompt.replace("\\n", "\n")
                prompt = prompt[:-1]
                prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.shape[0]
                if prompt_len <= token_limit and ("\u200a" not in prompt):
                    self.prompt_list.append(prompt)
        self.len = len(self.prompt_list)

    def get_sample(self, num):
        indices = random.sample(range(self.len), num)
        in_batch = [self.prompt_list[i] for i in indices]
        mask_len_list = []
        for prompt in in_batch:
            tr_start = prompt.find("翻译：")
            mask_len = self.tokenizer(
                prompt[: tr_start + 3], return_tensors="pt"
            ).input_ids.shape[1]
            mask_len_list.append(mask_len)
        return mask_len_list, in_batch

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.prompt_list[idx]


# generated_ids = model(
#     input_ids=model_inputs.input_ids,
#     attention_mask=model_inputs.attention_mask,
#     labels=model_inputs.input_ids,
#     return_dict=True,
# )

if __name__ == "__main__":
    Q_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model_name = "../dl_models/Qwen3-1.7B"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load the tokenizer and the model
    tokenizer: Qwen2TokenizerFast = AutoTokenizer.from_pretrained(model_name)
    model: Qwen3ForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=Q_config,
    )

    model = prepare_model_for_kbit_training(model)

    lora_rank = 16
    L_config = LoraConfig(
        r=lora_rank,
        use_rslora=True,
        # use_dora=True,
        target_modules=["q_proj", "v_proj", "k_proj"],
        lora_dropout=0.1,
        bias="none",
    )
    lora_model = get_peft_model(model, L_config).to(device)
    lora_model.train()
    lora_model.print_trainable_parameters()

    step_num = int(3e4)
    optimizer = ADAM(lora_model.parameters(), lr=1e-5)
    scheduler = get_wsd_schedule(
        optimizer=optimizer,
        num_warmup_steps=step_num * 0.05,
        num_stable_steps=step_num * 0.9,
        num_decay_steps=step_num * 0.05
    )

    lil_data = Q3_data("lil.txt", tokenizer, token_limit=512, test_limit=False)
    mc_data = Q3_data("mc.txt", tokenizer, token_limit=256)
    tr_data = Q3_data("tr.txt", tokenizer, token_limit=256)

    # scaler = GradScaler()

    loss_list = []

    with open("./loss_log.txt", "w", encoding="utf-8") as f:
        f.write("")

    time_last = time.time()
    for step in tqdm(range(step_num), ncols=120):
        delta = time.time() - time_last
        if delta > 2:
            print(batch)
        time_last = time.time()
        torch.cuda.empty_cache()
        mask = []
        batch = []
        data_list = [
            lil_data.get_sample(8),
        ]
        for _m, _b in data_list:
            mask += _m
            batch += _b

        inputs = tokenizer(batch, return_tensors="pt", padding=True)
        labels = inputs.input_ids.clone()
        for i, m in enumerate(mask):
            labels[i][:m] = -100
            labels[i][labels[i] == 151643] = -100
        inputs.to(device)
        labels.to(device)

        # print(inputs.input_ids[0])
        # print(inputs.attention_mask[0])
        # print(labels[0])
        # exit()

        optimizer.zero_grad()
        try:
            outputs = lora_model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                labels=labels,
                use_cache=False
            )
            loss = outputs.loss
            with open("./loss_log.txt", "a", encoding="utf-8") as f:
                f.write(f"{loss.detach().cpu().item()}\n")
            loss.backward()
            optimizer.step()
            scheduler.step()
            torch.cuda.empty_cache()
            # print(f"step={step:<6} loss={loss.item():.4f}")
            if step % 200 == 0 and step != 0:
                lora_model.eval()
                with torch.no_grad():
                    test_input = """<|im_start|>user\n文件：MayaEvents\n上下文：<||>.........<||>......<||>...<||>I tap on Maya’s name in my phone and think about how many other versions of me have been able to narrate that.<||>Sure, it may have taken the end of several worlds (Or several ends of one world) for me to {i}be able{/i} to share something like this with you, but...I’m here.<||>And hopefully soon, she will be as well.<||>As I stare down at a name that is perhaps the most important to me (Barring the recent intrusion of another girl I’ve known for far too long), I think about what I’m going to say when she picks up.<||>But then she picks up.<||>And I still have absolutely nothing.\n目标原文：<||>Sure, it may have taken the end of several worlds (Or several ends of one world) for me to {i}be able{/i} to share something like this with you, but...I’m here.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n翻译："""
                    test_aim = "当然，我可能经历了多个世界的末日(或者一个世界的多个末日){i}才能{/i}和你分享这样的事情，但是...我在这里。"
                    test_ids_atte = tokenizer(test_input, return_tensors="pt").to(
                        device
                    )
                    output_ids = model.generate(**test_ids_atte, max_new_tokens=100)
                    translation_ids = output_ids[0].tolist()
                    translation_ids = translation_ids[
                        test_ids_atte.input_ids.shape[1] :
                    ]
                    translation_res = tokenizer.decode(
                        translation_ids, skip_special_tokens=True
                    )
                    print("\n" + test_aim)
                    print(translation_res)
                    print(f"tokenLen = {len(translation_ids)}")
                lora_model.train()
                torch.cuda.empty_cache()

            if step % 5000 == 0 and step != 0:
                print(
                    f"step:{step} lora adapter saved====================================="
                )
                lora_model.save_pretrained(
                    f"model_saves/{'0' * (6 - len(str(step)))}{step}_lora_adapter"
                )
                tokenizer.save_pretrained(
                    f"model_saves/{'0' * (6 - len(str(step)))}{step}_lora_adapter"
                )
        except Exception as e:
            if 'out of memory' in str(e).lower():
                print("CUDA 显存不足，跳过该 step")
                print(batch)
            del inputs
            try:
                del labels
            except NameError:
                pass
            try:
                del outputs
            except NameError:
                pass
            torch.cuda.empty_cache()
            lora_model.to("cpu")
            lora_model.to("cuda")
            torch.cuda.empty_cache()
            step -= 1
            continue

    merged_model = lora_model.merge_and_unload()
    model_save_name = model_name[model_name.rfind("/") + 1 :]
    merged_model.save_pretrained(f"./{model_save_name}-LIL-LoRA-{lora_rank}-{step_num}")
    tokenizer.save_pretrained(f"./{model_save_name}-LIL-LoRA-{lora_rank}-{step_num}")
    print("model saved")
