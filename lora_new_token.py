import random
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Failed to load image Python extension.*")
from tqdm import tqdm

from os.path import join
import torch

# from torch.cuda.amp import autocast, GradScaler
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from torch.utils.data import Dataset, DataLoader, Sampler
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.optim import AdamW
from bitsandbytes.optim import PagedAdamW8bit
from transformers import BitsAndBytesConfig


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
                prompt += tokenizer.eos_token
                prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.shape[0]
                if prompt_len <= token_limit:
                    self.prompt_list.append(prompt)
        self.len = len(self.prompt_list)

    def get_sample(self, num):
        indices = random.sample(range(self.len), num)
        in_batch = [self.prompt_list[i] + self.tokenizer.eos_token for i in indices]
        return in_batch

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

    model_name = "../dl_models/Qwen3-0.6B-LIL-tokens"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load the tokenizer and the model
    tokenizer: Qwen2TokenizerFast = AutoTokenizer.from_pretrained(model_name)
    model: Qwen3ForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=Q_config,
    )

    model = prepare_model_for_kbit_training(model)

    L_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
    )
    lora_model = get_peft_model(model, L_config).to(device)
    lora_model.train()
    lora_model.print_trainable_parameters()

    step_num = 50000
    optimizer = PagedAdamW8bit(lora_model.parameters(), lr=1e-5)
    scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=2000,
        num_training_steps=step_num,
    )

    lil_data = Q3_data("lil.txt", tokenizer, token_limit=512, test_limit=True)
    mc_data = Q3_data("mc.txt", tokenizer, token_limit=256)
    tr_data = Q3_data("tr.txt", tokenizer, token_limit=256)

    # scaler = GradScaler()

    loss_list = []
    for step in tqdm(range(step_num)):
        mask = []
        batch = lil_data.get_sample(2)

        inputs = tokenizer(batch, return_tensors="pt", padding=True)
        labels = inputs.input_ids.clone()
        for label in labels:
            label:torch.Tensor
            label[label.tolist().index(151673)+1:] = -100
            # label[1:] = label.clone()[:-1]
        inputs.to(device)
        labels.to(device)

        # print(inputs.input_ids[0])
        # print(labels[0])
        # exit()

        optimizer.zero_grad()
        try:
            outputs = lora_model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            with open("./loss_log.txt", "a", encoding="utf-8") as f:
                    f.write(f"{loss.detach().cpu().item()}\n")
            loss.backward()
            optimizer.step()
            scheduler.step()
            # torch.cuda.empty_cache()
            print(f"step={step:<6} loss={loss.item():.4f}")
            if step % 50 == 0 and step != 0:
                lora_model.eval()
                with torch.no_grad():
                    test_input = """MayaEvents\n<||>.........<||>......<||>...<||>I tap on Maya’s name in my phone and think about how many other versions of me have been able to narrate that.\n<|start|><||>Sure, it may have taken the end of several worlds (Or several ends of one world) for me to {i}be able{/i} to share something like this with you, but...I’m here.<|end|>\n<||>And hopefully soon, she will be as well.<||>As I stare down at a name that is perhaps the most important to me (Barring the recent intrusion of another girl I’ve known for far too long), I think about what I’m going to say when she picks up.<||>But then she picks up.<||>And I still have absolutely nothing.\n<|translation|>"""
                    test_aim = "当然，我可能经历了多个世界的末日(或者一个世界的多个末日){i}才能{/i}和你分享这样的事情，但是...我在这里。"
                    test_ids_atte = tokenizer(test_input, return_tensors="pt").to(device)
                    output_ids = model.generate(**test_ids_atte)
                    translation_ids = output_ids[0].tolist()
                    translation_ids = translation_ids[translation_ids.index(151673) + 1:]
                    translation_res = tokenizer.decode(translation_ids)
                    print(test_aim)
                    print(translation_res)
                lora_model.train()
                torch.cuda.empty_cache()

            if step % 1000 == 0 and step != 0:
                print(f"step:{step} lora adapter saved=====================================")
                lora_model.save_pretrained(f"model_saves/{'0' * (6 - len(str(step)))}{step}_lora_adapter")
                tokenizer.save_pretrained(f"model_saves/{'0' * (6 - len(str(step)))}{step}_lora_adapter")
        except torch.OutOfMemoryError as e:
            print("OutOfMemoryError")
            step -= 1
            continue

    merged_model = lora_model.merge_and_unload()
    merged_model.save_pretrained("./Qwen3-0.6B-bnb4-LIL-LoRA-16-20000")
    tokenizer.save_pretrained("./Qwen3-0.6B-bnb4-LIL-LoRA-16-20000")
    print("model saved")
