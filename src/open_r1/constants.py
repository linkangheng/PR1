from open_r1.rewards import *

# REWARD MAPING
reward_funcs_registry = {
    "pr1_grounding": pr1_grounding_reward,
    "pr1_grounding_format": pr1_grounding_format_reward,
    "pr1_detection": pr1_detection_reward,
    "pr1_counting": pr1_counting_reward,
    "pr1_ocr": pr1_ocr_reward,
}

# SYSTEM PROMPTS
LLAVA_SYS = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

QWEN2_SYS = (
    "You are a helpful assistant. "
)

R1V_SYS = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

system_prompt_registry = {
    "default": QWEN2_SYS,
    "llava": LLAVA_SYS,
    "qwen": QWEN2_SYS,
    "r1v": R1V_SYS,
}

question_template_registry = {
    "default": "{question}",
    "r1v": "{question} Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags.",
    "pr1_grounding": "Output the bounding box of the {question} in the image.",
    "pr1_counting": "Output all the bounding boxes of the {question}",
    "pr1_detection": "Please output bbox coordinates and names of person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush.",
    "pr1_ocr": "{question}",
}
 
answer_template_registry = {
    "default": "{answer}",
    "r1v": "<answer> {answer} </answer>",
    "pr1_grounding": "{answer}",
    "pr1_counting": "{answer}",
    "pr1_detection": "{answer}",
    "pr1_ocr": "{answer}",
}