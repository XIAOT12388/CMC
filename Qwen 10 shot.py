import os
import re
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------- 环境 ----------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ---------------- 路径 ----------------
base = 'your path'
test_path = 'your path'
mapping_path = f"{base}/ your path"
save_dir = 'your path'
os.makedirs(save_dir, exist_ok=True)

# ---------------- 设置中文字体 ----------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# --------------- 工具函数 --------------
def load_mapping(path):
    df = pd.read_csv(path, encoding="utf-8")
    mapping_dict = dict(zip(df.emoji, df.chinese))
    print(f"成功加载 {len(mapping_dict)} 个emoji映射")
    return mapping_dict

def replace_emoji(text, mapping_dict):
    if not isinstance(text, str):
        return text
    keys = sorted(mapping_dict.keys(), key=lambda x: (-len(x), x))
    pat = re.compile("|".join(map(re.escape, keys)))
    return pat.sub(lambda m: f"[{mapping_dict[m.group(0)]}]", text)

# 调试函数
def debug_probability_distribution(y_probs, model_name):
    y_probs_array = np.asarray(y_probs)
    print(f"\n=== {model_name} 概率分布调试 ===")
    print(f"概率值范围: [{y_probs_array.min():.6f}, {y_probs_array.max():.6f}]")
    print(f"唯一概率值数量: {len(np.unique(y_probs_array))}")

# 评估函数
def evaluate_performance_native(y_true, y_pred, y_probs, model_name):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_probs_array = np.asarray(y_probs, dtype=float)

    debug_probability_distribution(y_probs_array, model_name)

    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n=== {model_name} 评估结果 ===")
    print(f"Acc={acc:.4f}  Pre={pre:.4f}  Rec={rec:.4f}  F1={f1:.4f}")

    return {
        "accuracy": acc, "precision": pre, "recall": rec, "f1": f1,
        "model": model_name, "y_true": y_true, "y_probs": y_probs_array
    }

# 数据
mapping_dict = load_mapping(mapping_path)
test_df = pd.read_excel(test_path)
test_df["text"] = test_df["text"].fillna("")
test_df["text-emoji"] = test_df["text-emoji"].fillna("")
test_df["label"] = test_df["label"].astype(int)
test_df["text_emoji_processed"] = test_df["text-emoji"].apply(lambda x: replace_emoji(x, mapping_dict))
print(f"测试集大小: {len(test_df)}")
print(f"正例: {test_df['label'].sum()}, 负例: {len(test_df) - test_df['label'].sum()}")

#  模型加载
print("正在下载和加载 Qwen2.5-7B-Instruct...")
try:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True, low_cpu_mem_usage=True
    ).eval()
    print("✅ 模型加载成功!")
except Exception as e:
    print(f"❌ 模型加载失败: {e}"); exit(1)

# few-shot

def build_few_shot_prompt(test_text):
    system_content = """你是一位网络欺凌检测助手，请判断输入的文本是否为网络欺凌。
        输出要求：
        - 如果是，只输出数字：1
        - 如果不是，只输出数字：0
        - 不要输出任何其他文字或解释"""

    user_content = f"""请参考以下来自不同用户的示例：
        待判断文本:太可怕的女人啦				
        答案:1

        待判断文本:说你法盲你还不服气 网上一直说的是调解书…… 和判决书不是一个东西			
        答案:0

        待判断文本:看到这么多人怼你 我就放心了				
        答案:0

        待判断文本:别奢求和我偶遇啦 先把你几百块钱破烂机换了再说 一边玩去吧				
        答案：0

        待判断文本:毒王				
        答案：1

        待判断文本:原来你没读过书，怪不得素质差。
        答案:0

        待判断文本:她去了多家酒吧就该骂啊[怒][恶心]。
        答案:1 

        待判断文本:谭竹快下地狱吧，捞女。
        答案:1

        待判断文本:钱是给女人看的。
        答案:0	

        待判断文本: 捞女。 
        答案:1		

        现在开始判断：
        待判断文本: {test_text}
        答案:"""

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]

# 推理
def predict_labels_single(texts, desc="推理进度"):
    preds, probs = [], []

    token_1 = tokenizer.encode("1", add_special_tokens=False)[-1]
    token_0 = tokenizer.encode("0", add_special_tokens=False)[-1]
    print(f"Token IDs - '1': {token_1}, '0': {token_0}")

    for i in tqdm(range(len(texts)), desc=desc):
        try:
            prompt = build_few_shot_prompt(texts[i])
            prompt_str = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt_str, return_tensors="pt", truncation=True, max_length=512).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False,
                    temperature=0,
                    output_scores=True,
                    return_dict_in_generate=True,
                    pad_token_id=tokenizer.eos_token_id
                )
                logits = outputs.scores[0][0]

                logit_0 = logits[token_0].item()
                logit_1 = logits[token_1].item()
                probs_binary = torch.softmax(torch.tensor([logit_0, logit_1]), dim=0)
                probability = probs_binary[1].item()

                preds.append(int(probability > 0.5))
                probs.append(probability)

        except Exception:
            preds.append(0)
            probs.append(0.5)

        torch.cuda.empty_cache()

    return preds, probs

# --------------- 主流程 ---------------
test_y = test_df["label"].values

print("开始Qwen2.5-7B 预测...")

test_texts_original = test_df["text"].tolist()
test_preds_original, test_probs_original = predict_labels_single(test_texts_original, desc="原始文本推理")

test_texts_processed = test_df["text_emoji_processed"].tolist()
test_preds_processed, test_probs_processed = predict_labels_single(test_texts_processed, desc="处理文本推理")

# 去掉可能的 NaN
valid_mask_original = ~np.isnan(test_probs_original)
test_y_original = np.array(test_y)[valid_mask_original]
test_probs_original = np.array(test_probs_original)[valid_mask_original]
test_preds_original = np.array(test_preds_original)[valid_mask_original]

valid_mask_processed = ~np.isnan(test_probs_processed)
test_y_processed = np.array(test_y)[valid_mask_processed]
test_probs_processed = np.array(test_probs_processed)[valid_mask_processed]
test_preds_processed = np.array(test_preds_processed)[valid_mask_processed]

print(f"原始文本有效样本数: {len(test_y_original)}/{len(test_y)}")
print(f"Emoji处理有效样本数: {len(test_y_processed)}/{len(test_y)}")

# --------------- 评估（原生） ---------------
metrics_original = evaluate_performance_native(test_y_original, test_preds_original, test_probs_original, "Qwen-原始文本")
metrics_processed = evaluate_performance_native(test_y_processed, test_preds_processed, test_probs_processed, "Qwen-Emoji处理")

# 总结果
print("\n=== 性能对比分析 ===")
print(f"准确率差异: {metrics_processed['accuracy'] - metrics_original['accuracy']:+.4f}")
print(f"精确率差异: {metrics_processed['precision'] - metrics_original['precision']:+.4f}")
print(f"召回率差异: {metrics_processed['recall'] - metrics_original['recall']:+.4f}")
print(f"F1分数差异: {metrics_processed['f1'] - metrics_original['f1']:+.4f}")

result_df = pd.DataFrame({
    "text_original": [test_texts_original[i] for i in np.where(valid_mask_original)[0]],
    "text_processed": [test_texts_processed[i] for i in np.where(valid_mask_processed)[0]],
    "true_label": test_y,
    "pred_original": test_preds_original,
    "prob_original": test_probs_original,
    "pred_processed": test_preds_processed,
    "prob_processed": test_probs_processed
})
result_df.to_excel(f"{save_dir}/test_results_qwen7b.xlsx", index=False)
metrics_comparison = pd.DataFrame([metrics_original, metrics_processed])
metrics_comparison.to_excel(f"{save_dir}/test_metrics_qwen7b.xlsx", index=False)

print(f"\n✅ 原生版评估完成！结果已保存至 {save_dir}")