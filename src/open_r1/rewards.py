import re
import os
import json
import torch
import numpy as np
import warnings
from datetime import datetime
from open_r1.utils import extract_bbox_answer, compute_iou, parse_json, HungarianMatcher, calculate_reward, get_cate_to_id, extract_boxes, pr1_counting_format_reward, num_scale

def log(content, sol, other_info, reward, tag=None):
    log_dir = os.getenv("LOG_DIR", None)
    os.makedirs(log_dir, exist_ok=True)
    if log_dir is None:
        warnings.warn("LOG_DIR is not set, log will not be saved")
        return
    log_path = os.path.join(log_dir, f"{tag}.log")
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    with open(log_path, "a") as f:
        try:
            f.write(f"------------- {current_time} {tag} reward: {reward} -------------\n")
            f.write(f"Content: {content}\n")
            f.write(f"Solution: {sol}\n")
            if other_info is not None:
                for k, v in other_info.items():
                    f.write(f"{k}: {v}\n")
        except:
            f.write("writeing error")

def format_reward(completions, pattern, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def think_format_reward(completions, **kwargs):
    """<think>...</think><answer>...</answer>"""
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    return format_reward(completions, pattern)

# Grounding Reward
def pr1_grounding_format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<|object_ref_start|>.*?<|object_ref_end|><|box_start|>.*?<|box_end|><|im_end|>"
    completion_contents = [completion[0]["content"].replace("<|endoftext|>", "") for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def pr1_grounding_reward(completions, solution, **kwargs):
    rewards = []
    contents = [completion[0]["content"].replace("<|endoftext|>", "") for completion in completions]
    for completion, sol in zip(contents, solution):
        bbox, is_qwen2vl = extract_bbox_answer(completion)
        iou = compute_iou(bbox, eval(sol))
        rewards.append(iou**2)
        log(completion + f"\nBounding box: {bbox}", sol, None, iou**2, "pr1_grounding_reward")
    return rewards

# Detection Reward
def pr1_detection_reward(completions, solution, return_reward_dict=False, **kwargs):
    cate_to_id = get_cate_to_id()
    rewards = []
    reward_dicts = []
    matcher = HungarianMatcher(threshold=0.5)

    for completion, sol in zip(completions, solution):
        try:
            reward = 0.0
            # Step 1: Parse the completion, get the format reward
            content = json.loads(completion[0]["content"].replace("```json", "").replace("```", ""))
            pred_id, pred_box, format_reward = parse_json(content, cate_to_id)
            reward += format_reward * 0.25
            
            # Step 2: Matching between prediction and ground truth
            num_pred = len(pred_id)
            num_target = len(sol["target_id"])
            if num_pred == 0 or num_target == 0:
                rewards.append(reward)
                continue
            
            pred_logits = torch.zeros(len(pred_id), 90)
            pred_logits[range(len(pred_id)), pred_id.long() - 1] = 1.0
            
            outputs = {
                "pred_logits": pred_logits.unsqueeze(0),
                "pred_boxes": pred_box.unsqueeze(0)
            }
            targets = [{"labels": torch.tensor(sol["target_id"]) - 1, 
                      "boxes": torch.tensor(sol["target_box"])}]
            
            indices = matcher(outputs, targets)
            num_matched = len(indices[0][0])

            matched_pred_boxes = pred_box[indices[0][0]]
            matched_target_boxes = torch.tensor(sol["target_box"])[indices[0][1]]
            matched_pred_ids = pred_id[indices[0][0]]
            matched_target_ids = torch.tensor(sol["target_id"])[indices[0][1]]
            
            if num_matched == 0:
                rewards.append(reward)
                continue

            # Step 3: Calculate the F1 Score Reward
            precision = num_matched / num_pred if num_pred > 0 else 0.0
            recall = num_matched / num_target if num_target > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            f1_scale = float(os.getenv("F1_SCALE", 0.75))
            reward += f1 * f1_scale
            
            # Step 4: Calculate the CLS Reward and IOU Reward
            cls_reward, iou_reward = calculate_reward(
                matched_pred_boxes,
                matched_target_boxes,
                matched_pred_ids,
                matched_target_ids
            )
            cls_coe = float(os.getenv("CLS_COE", 1.0))
            iou_coe = float(os.getenv("IOU_COE", 1.0))

            reward += cls_reward * cls_coe + iou_reward * iou_coe

            # Step 5: Calculate the Penalty(optional)
            do_penailty = os.getenv("DO_PENAIlty", "True")
            if do_penailty == "True":
                fp = (num_pred - num_matched) / num_pred
                fn = (num_target - num_matched) / num_target
                reward -= (fp + fn) * 0.5
            reward = reward.clamp(0, 3).item()

            # Step 6: Calculate the Coverage Bonus(optional)
            coverage_bonus = 0.0
            cov_scale = float(os.getenv("COVERAGE_SCALE", 0))
            gt_ids = torch.tensor(sol["target_id"])
            unique_cats = torch.unique(gt_ids)
            sum_ratio = 0.0
            for cat in unique_cats:
                target_count = (gt_ids == cat).sum().item()
                pred_count = (pred_id == cat.item()).sum().item()
                if target_count > 0:
                    ratio = min(pred_count / target_count, 1.0)
                else:
                    ratio = 0.0
                sum_ratio += ratio
            coverage_bonus = sum_ratio / len(unique_cats) * cov_scale
            reward += coverage_bonus

            rewards.append(reward)

            reward_dict = {
                "format_reward": format_reward,
                "num_reward": f1,
                "cls_reward": cls_reward.item(),
                "iou_reward": iou_reward.item(),
                "coverage_bonus": coverage_bonus,
                "reward": reward
            }
            reward_dicts.append(reward_dict)

            # Step 7: Log the reward
            log({'pred_id': pred_id, 'pred_box': pred_box}, sol, reward_dict, reward, "detection_reward")
            try:
                metrics = kwargs['metrics']
                metrics['format_reward'].append(format_reward)
                metrics['num_reward'].append(f1)
                metrics['cls_reward'].append(cls_reward.item())
                metrics['iou_reward'].append(iou_reward.item())
                metrics['coverage_bonus'].append(coverage_bonus)
            except Exception as e:
                pass
        
        except Exception as e:
            rewards.append(0.0)
    if return_reward_dict:
        return rewards, reward_dicts
    return rewards

# Counting Reward

def pr1_counting_reward(completions, solution, **kwargs):
    '''
    Input `completions` is a list of "rollouts" for a prompt.
    Input `solution` is the corresponding ground truth for each rollout.
    Output is a list of reward scores.
    
    example:
        completions: [
            '<|object_ref_start|>people<|object_ref_end|><|box_start|>(495,246),(999,997)<|box_end|><|im_end|>',
            '<|object_ref_start|>people<|object_ref_end|><|box_start|>(497,248),(919,397)<|box_end|><|im_end|>',
            '<|object_ref_start|>people<|object_ref_end|><|box_start|>(499,216),(919,987)<|box_end|><|im_end|>',
        ]
        solution: [
            [[221, 343], [476, 173]],
            [[221, 343], [476, 173]],
            [[221, 343], [476, 173]],
        ]
    '''
    # Step 1: Calculate format-based rewards for all completions.
    fmt_reward = pr1_counting_format_reward(completions)
    
    # Step 2: Extract the raw text content from completions and clean it.
    contents = [completion[0]["content"].replace("<|endoftext|>", "") for completion in completions]
    
    # Step 3: Extract bounding box center points from each cleaned completion string.
    bboxs = [extract_boxes(content) for content in contents]
    rewards = []
    
    # Step 4: Iterate over each completion and calculate its final reward.
    for completion, sol, bbox, fmt in zip(contents, solution, bboxs, fmt_reward):
        if fmt == 1.0:
            try:
                reward = num_scale(bbox, sol)
            except:
                print(f"Error in single_object_detection_bbox_reward: {completion} {bbox} {sol}")
                reward = 0.0
        else:
            reward = 0.0
        rewards.append(reward)
    return rewards

# OCR Reward
