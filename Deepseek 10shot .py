import os
import re
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)
from openai import OpenAI
import time
import json
import hashlib
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ---------------- 环境 ----------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ---------------- API配置 ----------------
client = OpenAI(
    api_key=os.environ.get('your API_KEY'),
    base_url="https://api.deepseek.com"
)

# ---------------- 路径 ----------------
test_path = 'your path'
save_dir = 'your path'
cache_file = f"{save_dir}/api_cache.json"
mapping_file = 'your path'
os.makedirs(save_dir, exist_ok=True)

# emoji映射函数
def load_mapping(mapping_file):
    """加载emoji到中文语义的映射"""
    mapping_dict = {}
    try:
        df = pd.read_csv(mapping_file, encoding='utf-8')
        for _, row in df.iterrows():
            emoji = row['emoji']
            description = row['chinese']
            mapping_dict[emoji] = description
        print(f"成功加载 {len(mapping_dict)} 个emoji映射")
    except Exception as e:
        print(f"加载emoji映射文件失败: {e}")
    return mapping_dict


# 修改emoji处理函数
def replace_emoji_with_special_token(text, mapping_dict):
    """将文本中的emoji替换为中文语义描述"""
    if pd.isna(text) or not isinstance(text, str):
        return ""

    sorted_keys = sorted(mapping_dict.keys(), key=lambda x: (-len(x), x))
    pattern = re.compile('|'.join(re.escape(k) for k in sorted_keys))

    def replace_func(match):
        emoji = match.group()
        return f"[{mapping_dict[emoji]}]"

    return pattern.sub(replace_func, text)


# 缓存管理
def load_cache():
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


# 修改get_text_hash函数
def get_text_hash(text):

    if pd.isna(text):  # 处理NaN
        text = ""
    elif not isinstance(text, str):
        text = str(text)
    return hashlib.md5(text.encode('utf-8')).hexdigest()

# 调试函数：检查概率分布
def debug_probability_distribution(y_probs, model_name):

    y_probs_array = np.asarray(y_probs)

    print(f"\n=== {model_name} 概率分布调试 ===")
    print(f"概率值范围: [{y_probs_array.min():.6f}, {y_probs_array.max():.6f}]")
    print(f"唯一概率值数量: {len(np.unique(y_probs_array))}")

    # 统计概率值分布
    unique_probs, counts = np.unique(y_probs_array, return_counts=True)
    print(f"前10个概率值及其频次:")
    for prob, count in zip(unique_probs[:10], counts[:10]):
        print(f"  {prob:.6f}: {count}次")

    if len(unique_probs) < 10:
        print(f"所有概率值: {unique_probs}")

# 数据
mapping_dict = load_mapping(mapping_file)

test_df = pd.read_excel(test_path)
test_df["text"] = test_df["text"].fillna("").astype(str)
test_df["text-emoji"] = test_df["text-emoji"].fillna("").astype(str)
test_df["label"] = test_df["label"].astype(int)

test_df["text_emoji_processed"] = test_df["text-emoji"].apply(
    lambda x: replace_emoji_with_special_token(x, mapping_dict)
)

print(f"测试集大小: {len(test_df)}")
print(f"正例: {test_df['label'].sum()}, 负例: {len(test_df) - test_df['label'].sum()}")

# few-shot demo
def build_messages(test_text):
    """统一的prompt构建函数"""
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


# 概率提取函数
def extract_true_probability_reliable(response):
    try:
        choice = response.choices[0]

        # 严格检查logprobs是否存在
        if not choice.logprobs or not choice.logprobs.content:
            print('[WARN] logprobs missing')
            return None

        # 检查第一个token的top_logprobs
        token_logprobs = choice.logprobs.content[0].top_logprobs
        if not token_logprobs:
            print('[WARN] top_logprobs empty')
            return None

        # 收集所有token的logprobs
        logprob_dict = {}
        for top_logprob in token_logprobs:
            token = top_logprob.token.strip()
            logprob_dict[token] = top_logprob.logprob

        # 关键：如果0和1都在logprobs中，计算softmax
        if "0" in logprob_dict and "1" in logprob_dict:
            logprob_0 = logprob_dict["0"]
            logprob_1 = logprob_dict["1"]

            # 使用softmax计算真实概率
            max_logprob = max(logprob_0, logprob_1)
            logprob_0_adj = logprob_0 - max_logprob
            logprob_1_adj = logprob_1 - max_logprob

            exp_0 = np.exp(logprob_0_adj)
            exp_1 = np.exp(logprob_1_adj)
            sum_exp = exp_0 + exp_1

            probability = exp_1 / sum_exp
            return float(probability)

        else:
            print('[WARN] 0/1 tokens missing in top_logprobs')
            return None

    except Exception as e:
        print(f'[ERROR] 概率提取失败: {e}')
        return None

# API调用函数
def call_deepseek_api(messages, text_hash, cache, max_retries=1):

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                max_tokens=1,
                temperature=0,
                logprobs=True,
                #top_logprobs=20,
                stream=False
            )

            content = response.choices[0].message.content.strip()
            prob = extract_true_probability_reliable(response)

            # 如果概率提取失败，重试而不是使用固定值
            if prob is None:
                print(f'第{attempt + 1}次尝试概率提取失败，进行重试')
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                else:
                    # 最终失败时返回None，让上层处理
                    print(f'[ERROR] 无法获取真实概率，文本哈希: {text_hash[:8]}')
                    return content, parse_model_output(content), None

            pred = parse_model_output(content)

            cache[text_hash] = {
                'content': content,
                'pred': pred,
                'prob': prob  # 保存真实概率
            }

            return content, pred, prob

        except Exception as e:
            print(f"API调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 移除随机等待
                time.sleep(wait_time)
            else:
                # 最终失败时返回None
                return "0", 0, None

    return "0", 0, None


# 解析函数
def parse_model_output(answer):
    """严格解析模型输出，返回预测标签"""
    if not answer:
        return 0

    answer = answer.strip()

    if answer == "1":
        return 1
    elif answer == "0":
        return 0
    else:
        if answer not in ["0", "1"]:
            print(f"警告：模型输出异常: '{answer}'，默认返回0")
        return 0


# 批量预测函数
def predict_labels_api(texts, batch_delay=2, desc="DeepSeek推理"):
    preds = []
    probs = []
    failed_count = 0

    # 用于调试：记录概率分布
    unique_probs_set = set()

    cache = load_cache()
    cache_hits = 0
    cache_misses = 0

    for i, text in enumerate(tqdm(texts, desc=desc)):
        text_hash = get_text_hash(text)

        if text_hash in cache and isinstance(cache[text_hash], dict):
            cache_hits += 1
            cached_result = cache[text_hash]

            # 确保缓存中有必要的字段
            if 'pred' not in cached_result:
                cached_result['pred'] = parse_model_output(cached_result.get('content', ''))

            if 'prob' not in cached_result or cached_result['prob'] is None:
                # 如果缓存中没有概率或概率为None，重新计算
                cache_misses += 1
                cache_hits -= 1
                del cache[text_hash]
                continue
            else:
                pred = cached_result['pred']
                prob = cached_result['prob']
        else:
            cache_misses += 1
            messages = build_messages(text)
            content, pred, prob = call_deepseek_api(messages, text_hash, cache)

        # 处理概率为None的情况
        if prob is None:
            failed_count += 1
            if pred == 1:
                prob = 0.9  # 固定值，不随机
            else:
                prob = 0.1  # 固定值，不随机
            print(f'[WARN] 使用默认概率: {prob} for pred: {pred}')

        preds.append(pred)
        probs.append(prob)
        unique_probs_set.add(round(prob, 4))

        if i % 20 == 0:
            save_cache(cache)

        if text_hash not in cache and i < len(texts) - 1:
            time.sleep(batch_delay)

    save_cache(cache)

    return preds, probs


# 评估函数
def evaluate_performance_fixed(y_true, y_pred, y_probs, model_name):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_probs_array = np.asarray(y_probs, dtype=float)

    # 调试概率分布
    debug_probability_distribution(y_probs_array, model_name)

    # 基础指标
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n=== {model_name} 评估结果 ===")
    print(f"Acc    = {acc:.4f}")
    print(f"Pre    = {pre:.4f}")
    print(f"Rec    = {rec:.4f}")
    print(f"F1     = {f1:.4f}")

    return {
        "accuracy": acc,
        "precision": pre,
        "recall": rec,
        "f1": f1,
        "model": model_name,
        "y_true": y_true,
        "y_probs": y_probs_array
    }
#  主流程
test_y = test_df["label"].values

# 预测原始文本
print("\n1. 预测原始文本（无emoji处理）...")
test_texts_original = test_df["text"].tolist()
test_preds_original, test_probs_original = predict_labels_api(
    test_texts_original, batch_delay=2, desc="原始文本推理"
)

# 预测处理后的文本
print("\n2. 预测处理后的文本（包含emoji语义）...")
test_texts_processed = test_df["text_emoji_processed"].tolist()
test_preds_processed, test_probs_processed = predict_labels_api(
    test_texts_processed, batch_delay=2, desc="处理文本推理"
)

# 评估两种版本的性能
print("\n3. 评估模型性能...")
metrics_original = evaluate_performance_fixed(test_y, test_preds_original, test_probs_original, "DeepSeek-原始文本")
metrics_processed = evaluate_performance_fixed(test_y, test_preds_processed, test_probs_processed, "DeepSeek-Emoji处理")


# 对比分析
print("\n=== 性能对比分析 ===")
print(f"准确率差异: {metrics_processed['accuracy'] - metrics_original['accuracy']:+.4f}")
print(f"精确率差异: {metrics_processed['precision'] - metrics_original['precision']:+.4f}")
print(f"召回率差异: {metrics_processed['recall'] - metrics_original['recall']:+.4f}")
print(f"F1分数差异: {metrics_processed['f1'] - metrics_original['f1']:+.4f}")

# --------------- 保存结果 ---------------
print("\n7. 保存最终结果...")
result_df3 = pd.DataFrame({
    "text_original": test_texts_original,
    "text_processed": test_texts_processed,
    "true_label": test_y,
    "pred_original": test_preds_original,
    "prob_original": test_probs_original,
    "pred_processed": test_preds_processed,
    "prob_processed": test_probs_processed
})

result_df3.to_excel(f"{save_dir}/test_result.xlsx", index=False)

metrics_comparison = pd.DataFrame([metrics_original, metrics_processed])
metrics_comparison.to_excel(f"{save_dir}/test_metrics.xlsx", index=False)

