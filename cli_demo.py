from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = ''

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype='auto'
).eval()

messages = [
    {"role": "user", "content": "我叔是债务纠纷担保人，可是债务人不知所踪，我叔被迫还债，后我叔将债务人起诉法院，可是将进两年法院无任何回应，请问我叔怎样讨回损失。"}
]

input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
output_ids = model.generate(input_ids.to('cuda'))
response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

print(response)