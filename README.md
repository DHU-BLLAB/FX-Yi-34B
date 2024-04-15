
# 法陉法律大模型

 <img src="https://img.shields.io/badge/python-3.10-blue.svg">


# 项目简介

本项目开源的中文法律通用大模型是通过Yi-34B-Chat经过QLoRA指令微调得到的。并且通过LESS算法来筛选出高质量的数据集，使得微调大模型的成本下降。

我们对通用语言大模型在法律领域的性能进行了优化，以增强模型的可信度和专业水准。模型能在不同的应用场景下提供帮助，尤其在**法律知识问答**，**裁判文书裁定**等方面有较高的准确性。

# 最近更新
[24/03/20] 开源了法陉模型参数

# 快速开始

#### 1.环境配置

```bash
cd src
pip install -r requirements.txt
```

#### 2.模型下载

```
git clone https://huggingface.co/DHU-BLLAB/FX-Yi-34B
```

#### 3.模型推理

##### python

此模型是以Yi-34B-Chat为基底模型微调训练而成，您可以参考[Yi](https://github.com/01-ai/Yi)来进行推理部署

```python
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
```

##### 命令行工具

```bash
python cli_demo.py
```

##### 案例

```
user:

法陉：

```

# 数据集准备

本项目数据分为四个部分：

​	1.法律知识问答

​	2.新闻信息

​	3.法律裁判文书

​	4.客户信息分析

| 数据类型 |                        描述                        | 数量 | 占比（%） |
| :------: | :------------------------: | :------: | :-------: |
| 知识问答 |                 相关法律问题的回复                 | 467265 | 52 |
| 新闻信息 | 清华整合新闻数据集、网易新闻数据集、金融新闻数据集 | 105461 | 12 |
| 裁判文书 |        使用ChatGPT从真实的裁判文书数据提取         | 280051 | 31 |
| 客户消息 |          使用ChatGPT从客户数据中进行提取           | 45246 | 5 |
| **总计** |                                                    | 898023 |  **100%**  |

### 数据集格式

#### 法律知识问答

```json
    "instruction": "我叔是债务纠纷担保人，可是债务人不知所踪，我叔被迫还债，后我叔将债务人起诉法院，可是将进两年法院无任何回应，请问我叔怎样讨回损失。",
    "output": "依据《民法典》规定，具有代为清偿债务能力的法人、其他组织或者公民，可以作保证人。做保证人应当签定保证合同，另外，如果主合同中没有保证条款，但是保证人在主合同上以保证人的身份签字或者盖章的，保证合同成立。保证形式分为一般保证和连带责任保证两种。如果保证责任形式不明确，就推定连带责任形式。在保证成立的前提下，保证人根据不同的保证形式承担不同的保证责任。一般保证的保证人在主合同纠纷未经审判或者仲裁，并就债务人财产依法强制执行仍不能履行债务前，对债权人可以拒绝承担保证责任。而连带责任保证的债务人在主合同规定的债务履行期届满没有履行债务的，债权人可以要求债务人履行债务，也可以要求保证人在其保证范围内承担保证责任。",
    "system": "",
    "input": ""
```

#### 新闻信息

```json
 	"instruction": "曼联的首发阵容是什么？  网易体育12月18日报道：\n日本当地时间本周四（18日）傍晚19点30分（北京时间18日18点30分），2008年国际足联世俱杯赛在横滨国际综合竞技场展开第二场半决赛的争夺，由欧洲冠军联赛冠军曼联对垒亚冠冠军、来自日本的大阪钢巴队。\n目前，双方已经公布了首发阵容。曼联方面，特维斯与吉格斯搭档锋线，而鲁尼被放到了替补席上。\n曼联（4-4-2）：1-范德萨/2-内维尔、5-费迪南德、15-维迪奇、3-埃夫拉/17-纳尼、8-安德森、18-斯科尔斯、7-C罗/11-吉格斯、32-特维斯\n替补：16-卡里克、24-弗莱彻、22-奥谢、13-朴智星、21-拉菲尔、10-鲁尼、19-维尔贝克\n大阪钢巴（3-4-3）：22-藤谷阳介/2-中泽聪太、5-山口智、21-加地亮、/13-安田理大、17-明神智和、7-远藤保仁、27-桥本英郎/30-山崎雅人、9-卢卡斯、11-播户龙二\n替补：1-松代直树、8-寺田、10-二川孝广、18-罗尼、19-下平匠、20-苍田秋、23-武井择也\n(本文来源：网易体育  )"
 	"output": "1-范德萨/2-内维尔、5-费迪南德、15-维迪奇、3-埃夫拉/17-纳尼、8-安德森、18-斯科尔斯、7-C罗/11-吉格斯、32-特维斯",
    "system": "",
    "input": "",
    
  
```

#### 裁判文书

```json
    "instruction": "根据给定的文档：标题: 中国建设银行股份有限公司新疆维吾尔自治区分行、王秀莲民事执行实施执行裁定书案件类型: 执行案件文书内容: 新疆维吾尔自治区乌鲁木齐市天山区人民法院\n执行裁定书\n（2021）新0102执3519号之二申请执行人中国建设银行股份有限公司新疆维吾尔自治区分行与被执行人王秀莲信用卡纠纷一案，本院作出的（2020）新0102民初字第5806号民事判决书已发生法律效力。因被执行人未履行该生效法律文书确定的义务，申请执行人向本院申请强制执行，本院于2021年5月10日依法立案执行。本案申请执行标的151987.83元，已执行标的为0.00元，未执行标的为151987.83元在执行过程中，本院已采取如下执行措施：\n1、本院于2021年5月10日向被执行人送达执行通知书、执行裁定书、报告财产令、财产申报表、限制消费令等法律文书。\n2、本院分别于2021年5月13日、2021年5月22日、2021年6月17日、2021年9月13日通过全国法院网络执行查控系统查询和实地走访调查等方式对被执行人名下的财产信息，银行存款，互联网银行账户存款、有价证券、对外投资、保险等财产信息进行查询，发现被执行人名下无银行存款、无互联网银行帐户存款，无证券、无对外投资、无保险、无公司登记等信息。\n3、本院于2021年5月20日查询被执行人王秀莲乌鲁木齐市房产及土地使用权的登记情况，轮候查封被执行人名下位于米东区石化中兴街1324号四区15栋1-107号房产，首封案号为（2020）新01031560号，有抵押，抵押权数额为140000元，无法处置。\n4、本院于2021年5月20日经过公安部车辆信息查询被执行人王秀莲名下无车辆登记信息。\n5、本院于2021年9月13日限制被执行人王秀莲高消费及非生活必需品的有关消费并发布。\n6、本院于2021年9月13日向申请执行人知其本院执行措施及执行结果。\n对上述案件执行情况、采取的财产调查措施、被执行人王秀莲的财产状况、终结本次执行程序的依据及法律后果等信息，本院已于2021年9月13日告知申请执行人表示无异议且不能提供被执行人的其他财产线索，同意将本案终结本次执行程序本院认为，本案在执行过程中已穷尽执行措施，被执行人无财产可供执行，申请执行人对此予以认可，且不能提供被执行人的其他财产线索。依据《最高人民法院关于适用〈中华人民共和国民事诉讼法〉的解释》第五百一十九条之规定，裁定如下终结（2021）新0102执3519号执行案件本次执行程序。\n终结本次执行程序之后，被执行人王秀莲应当继续履行生效法律文书确定的义务。申请执行人发现被执行人有可供执行的财产，可以向本院申请恢复执行。申请恢复执行不受申请执行失效期间的限制。\n本裁定送达后立即生效审判长买买提艾力\n审判员冷长青\n审判员瓦热思\n二〇二一年九月十三日\n书记员米扎提当事人: 中国建设银行股份有限公司新疆维吾尔自治区分行；王秀莲案由: 民事，请你以一个专业律师的角度，简要分析裁判文书中提到的以下信息：案件标题、案件发生时间、案件发生地点、原告、被告、证人、证明文件或证物、案件情况描述、裁判结果、判决理由和依据法条。你需要按照如下步骤生成数据：1 - 键仅为上述关键词，若某关键词未在裁判文书中提及，则其值为空。2 - “案件情况描述”需要简述案件过程。3 - “裁判结果”需要描述详细，依据清晰，使非法律工作者也可以理解。4 - “判决理由和依据法条”字段需要简述判决理由并且包含裁判文书中涉及的相关法条。5 - 生成数据与文档语言一致。6 - 最后，请将生成的数据整理成json格式，字段如下所示：“案件标题”，“案件发生时间”，“案件发生地点”，“原告”，“被告”，“证人”，“证明文件或证物”，“案件情况描述”，“裁判结果”，“判决理由和依据法条”。",
    "output": "{\n  \"案件标题\": \"中国建设银行股份有限公司新疆维吾尔自治区分行、王秀莲民事执行实施执行裁定书\",\n  \"案件发生时间\": \"2021年5月10日\",\n  \"案件发生地点\": \"新疆维吾尔自治区乌鲁木齐市天山区人民法院\",\n  \"原告\": \"中国建设银行股份有限公司新疆维吾尔自治区分行\",\n  \"被告\": \"王秀莲\",\n  \"证人\": \"\",\n  \"证明文件或证物\": \"\",\n  \"案件情况描述\": \"中国建设银行股份有限公司新疆维吾尔自治区分行与被执行人王秀莲信用卡纠纷一案，本院作出的（2020）新0102民初字第5806号民事判决书已发生法律效力。因被执行人未履行该生效法律文书确定的义务，申请执行人向本院申请强制执行，本院于2021年5月10日依法立案执行。本案申请执行标的151987.83元，已执行标的为0.00元，未执行标的为151987.83元在执行过程中，本院已采取如下执行措施：\\n1、本院于2021年5月10日向被执行人送达执行通知书、执行裁定书、报告财产令、财产申报表、限制消费令等法律文书。\\n2、本院分别于2021年5月13日、2021年5月22日、2021年6月17日、2021年9月13日通过全国法院网络执行查控系统查询和实地走访调查等方式对被执行人名下的财产信息，银行存款，互联网银行账户存款、有价证券、对外投资、保险等财产信息进行查询，发现被执行人名下无银行存款、无互联网银行帐户存款，无证券、无对外投资、无保险、无公司登记等信息。\\n3、本院于2021年5月20日查询被执行人王秀莲乌鲁木齐市房产及土地使用权的登记情况，轮候查封被执行人名下位于米东区石化中兴街1324号四区15栋1-107号房产，首封案号为（2020）新01031560号，有抵押，抵押权数额为140000元，无法处置。\\n4、本院于2021年5月20日经过公安部车辆信息查询被执行人王秀莲名下无车辆登记信息。\\n5、本院于2021年9月13日限制被执行人王秀莲高消费及非生活必需品的有关消费并发布。\\n6、本院于2021年9月13日向申请执行人知其本院执行措施及执行结果。\\n对上述案件执行情况、采取的财产调查措施、被执行人王秀莲的财产状况、终结本次执行程序的依据及法律后果等信息，本院已于2021年9月13日告知申请执行人表示无异议且不能提供被执行人的其他财产线索，同意将本案终结本次执行程序本院认为，本案在执行过程中已穷尽执行措施，被执行人无财产可供执行，申请执行人对此予以认可，且不能提供被执行人的其他财产线索。\",\n  \"裁判结果\": \"根据《最高人民法院关于适用〈中华人民共和国民事诉讼法〉的解释》第五百一十九条之规定，裁定终结（2021）新0102执3519号执行案件本次执行程序。终结本次执行程序之后，被执行人王秀莲应当继续履行生效法律文书确定的义务。申请执行人发现被执行人有可供执行的财产，可以向本院申请恢复执行。申请恢复执行不受申请执行失效期间的限制。\",\n  \"判决理由和依据法条\": \"本案终结本次执行程序的依据为《最高人民法院关于适用〈中华人民共和国民事诉讼法〉的解释》第五百一十九条之规定。根据该规定，被执行人无财产可供执行，申请执行人对此予以认可，且不能提供被执行人的其他财产线索，本院认为本案在执行过程中已穷尽执行措施，故裁定终结本次执行程序。被执行人王秀莲应当继续履行生效法律文书确定的义务。申请执行人发现被执行人有可供执行的财产，可以向本院申请恢复执行。申请恢复执行不受申请执行失效期间的限制。\"\n}",
    "system": "",
    "input": ""
```

#### 客户消息

```json
    "instruction": "请根据文本整理客户档案，请整理成json格式，需要包含客户信息和咨询分析。客户信息包括：客户用户名、咨询时间、姓名、年龄、性别、所在地、联系方式。咨询分析包括：客户咨询情况总结、客户情绪分析结果、客户焦急程度、客户继续咨询意图是否强烈、客户咨询问题类型。",
    "output": "{'客户信息': {'客户用户名': 'user123', '咨询时间': '2022-01-01 10:00:00', '姓名': '李小明', '年龄': '30', '性别': '男', '所在地': '北京', '联系方式': '13812345678'}, '咨询分析': {'客户咨询情况总结': '客户在使用信用卡过程中出现逾期还款问题，银行要求归还透支本金、利息、违约金和消费手续费。客户希望了解是否有法律依据来抗辩或减少还款金额。', '客户情绪分析结果': '客户表达了对信用卡纠纷问题的担忧和焦虑。', '客户焦急程度': '中', '客户继续咨询意图是否强烈': '弱', '客户咨询问题类型': '信用卡纠纷'}}",
    "system": "",
	"input": "{'客户信息': {'客户用户名': 'user123', '咨询时间': '2022-01-01 10:00:00', '姓名': '李小明', '年龄': '30', '性别': '男', '所在地': '北京', '联系方式': '13812345678'}, '对话内容': [{'角色': '普通客户', '对话': '您好，请问我有一个信用卡纠纷的问题，想请教一下。'}, {'角色': '律师', '对话': '您好，我是律师，请问您的信用卡纠纷是什么情况？'}, {'角色': '普通客户', '对话': '我在使用信用卡的过程中，未能按合同约定还款，并多次逾期。现在银行要求我归还透支本金、利息、违约金和消费手续费，我想知道我是否有法律依据来抗辩或减少还款金额？'}, {'角色': '律师', '对话': '根据您提供的情况，根据《中华人民共和国合同法》的规定，当事人应当按照合同约定全面履行自己的义务，逾期还款构成违约，银行有权向您主张所欠的透支本金、利息、违约金和消费手续费。根据您的描述，银行的要求是合法有效的。'}, {'角色': '普通客户', '对话': '那我是否有其他的解决办法？'}, {'角色': '律师', '对话': '根据您提供的情况，您可以与银行协商制定还款计划，以减轻您的还款压力。另外，您也可以咨询专业律师，根据具体情况提出相应的抗辩理由，寻求法律援助。'}, {'角色': '普通客户', '对话': '谢谢您的回答，我会考虑这些建议的。'}, {'角色': '律师', '对话': '不客气，如果您还有其他问题，可以随时向我咨询。祝您好运！'}, {'角色': '普通客户', '对话': '好的，谢谢您的帮助！'}, {'角色': '律师', '对话': '不用客气，如果您对本次咨询服务满意的话，可以给予评价。'}, {'角色': '普通客户', '对话': '我对这次咨询服务很满意，谢谢您的专业回答！'}]}"
```

# 数据集的筛选

由于正常的微调训练所需要的数据较多，并且会消耗大量的GPU和内存资源，降低训练速度，我们使用了[LESS](https://github.com/princeton-nlp/LESS)项目来进行数据集的筛选。从数据中筛选出最符合条件的前5%数据进行训练

在本次项目中，我们筛选数据的模型选用了[chinese-llama-2-1.3b](https://huggingface.co/hfl/chinese-llama-2-1.3b)，筛选标准数据集选用的是tydiqa。



### 1. 环境搭建

```
pip3 install torch==2.1.2 torchvision torchaudio
cd LESS
pip install -r requirement.txt
pip install -e .
```

### 2. 模型热身

```
DATA_DIR=../data
MODEL_PATH=../chinese-llama-2-1.3b
PERCENTAGE=0.05
DATA_SEED=3
JOB_NAME=chinese-llama-2-1.3b-p${PERCENTAGE}-lora-seed${DATA_SEED}

./less/scripts/train/warmup_lora_train.sh "$DATA_DIR" "$MODEL_PATH" "$PERCENTAGE" "$DATA_SEED" "$JOB_NAME"
```

Tips: 在项目代码中也需要将 “llama2-7b” 修改为  "chinese-llama-2-1.3b"。

### 3. 计算数据集梯度

```
CKPT=105
TRAINING_DATA_NAME=law
TRAINING_DATA_FILE=../data/train/processed/law/law_data.json
GRADIENT_TYPE="adam"
MODEL_PATH=../out/chinese-llama-2-1.3b-p0.05-lora-seed3/checkpoint-${CKPT}
OUTPUT_PATH=../grads/chinese-llama-2-1.3b-p0.05-lora-seed3/${TRAINING_DATA_NAME}-ckpt${CKPT}-${GRADIENT_TYPE}
DIMS="8192"

./less/scripts/get_info/grad/get_train_lora_grads.sh "$TRAINING_DATA_FILE" "$MODEL_PATH" "$OUTPUT_PATH" "$DIMS" "$GRADIENT_TYPE"
```

Tips: CKPT 数值的选择为checkpoint的最后一个保存点。

### 4. 筛选数据集

### 第一步

```
CKPT=105
TASK=tydiqa
MODEL_PATH=../out/chinese-llama-2-1.3b-p0.05-lora-seed3/checkpoint-${CKPT}
OUTPUT_PATH=../grads/chinese-llama-2-1.3b-p0.05-lora-seed3/${TASK}-ckpt${CKPT}-sgd 
DATA_DIR=../data
DIMS="4096 8192"

./less/scripts/get_info/grad/get_eval_lora_grads.sh "$TASK" "$DATA_DIR" "$MODEL_PATH" $OUTPUT_PATH "$DIMS"
```



### 第二步

```
DIM=8192 
GRADIENT_PATH=../grads/chinese-llama-2-1.3b-p0.05-lora-seed3/{}-ckpt{}-adam/dim${DIM}
TRAIN_FILE_NAMES="law"
CKPTS="105" 
CHECKPOINT_WEIGHTS="1.6877e-05" 
VALIDATION_GRADIENT_PATH=../grads/chinese-llama-2-1.3b-p0.05-lora-seed3/{}-ckpt{}-sgd/dim${DIM}
TARGET_TASK_NAMES="tydiqa"
SELECTED_DATA_OUTPUT_PATH="../selected_data"

./less/scripts/data_selection/matching.sh "$GRADIENT_PATH" "$TRAIN_FILE_NAMES" "$CKPTS" "$CHECKPOINT_WEIGHTS" "$VALIDATION_GRADIENT_PATH" "$TARGET_TASK_NAMES" "$SELECTED_DATA_OUTPUT_PATH"
```

Tips: CHECKPOINT_WEIGHTS数值的选择可以根据训练数据的checkpoint进行进行平均计算。

## 模型输出
可以从输出结果看出，经过微调之后模型，在常识方面上的回答并没有削弱，并且通过LESS筛选出来的数据可以使得在法律领域上的问答更加专业。

#### 法律知识问答

| 问题                                                         | GPT-3.5-turbo                                                | Yi-34B-Chat                                                  | 名字                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 我想咨询一下，孩子出生到现在不到2周岁，至今未落户，现在我和妻子感情不合，办理离婚手续时候要求先给孩子落户，我想请问一下，我现在给孩子落户需要注意什么吗?落在父亲户下还是母亲户下，与之后离婚有什么关系吗？还有我的户口是跟我外祖母在一个户口本上，有影响 | 首先，孩子落户的问题通常由户籍所在地的公安局（或派出所）负责办理，具体办理程序和要求可能会因地区而异，建议您根据当地的具体规定进行咨询和办理。在孩子落户时，一般需要提供父母双方的身份证、结婚证等相关证件，以及孩子的出生证明等材料。至于选择落户在父亲户口还是母亲户口下，主要考虑到未来监护权和抚养责任的问题。如果您和妻子的感情确实不合，而又预计会办理离婚手续，建议您在考虑孩子落户时，尽量根据未来的监护和抚养安排做出决定。通常情况下，父母双方会协商并在法律程序中确定监护权和抚养责任，这与孩子最终落户在哪个户口下有关。至于您的户口和您外祖母在一个户口本上的情况，这可能会对一些具体事务产生影响，但一般不会对孩子落户产生直接影响。不过，具体情况还是建议您咨询当地相关部门或法律专业人士，以获取更加准确和详细的指导。 | 孩子落户的问题需要根据当地户籍管理规定来办理。一般来说，孩子出生后可以在父亲或母亲任何一方的户籍所在地落户。如果孩子出生时没有落户，现在需要落户，你可以咨询当地的派出所或户籍管理中心，了解具体的落户流程和需要准备的资料。在离婚手续办理过程中，孩子的户籍归属可能会影响到离婚后的抚养权和探视权等问题。因此，建议你在离婚前与前妻协商好孩子的抚养问题，包括户籍归属。如果双方无法达成一致，可以通过法院判决来解决。关于你的户口与外祖母在一个户口本上，这可能会影响到孩子的落户。有些地区可能要求孩子的直系亲属（如父母）的户口本上要有孩子的落户地址。因此，你需要确认你目前的户籍状况是否符合孩子落户的条件。如果需要，你可以考虑将户口迁出，单独为孩子办理一个户口本。总之，孩子落户的问题需要根据当地政策来办理，建议你咨询当地户籍管理部门，了解具体要求和流程。同时，考虑到离婚后可能出现的法律问题，建议你在离婚前与前妻协商好孩子的抚养问题，并在必要时咨询律师。" | 【法律意见】孩子未落户，可以先上户口，然后再办理离婚手续。【法律依据】《中华人民共和国户口登记条例》第七条婴儿出生后，在申报出生登记前，有关单位可为其申报出生暂户。暂户应在其出生后一个月内申报出生登记。《中华人民共和国户口登记条例》第八条公民出生后，其父母或者监护人应当在下列期限内，向户口登记机关申报出生登记：（一）出生后一个月内，由户主、亲属、抚养人或者邻居向婴儿常住地户口登记机关申报出生登记。（二）出生后超过一个月，在婴儿出生后六个月内，由户主、亲属、抚养人或者邻居向婴儿常住地户口登记机关申报出生登记。（三）婴儿出生后六个月以上，在婴儿出生后一年内，由户主、亲属、抚养人或者邻居向婴儿常住地户口登记机关申报出生登记。（四）婴儿出生后超过一年，由户主、亲属、抚养人或者邻居向婴儿常住地户口登记机关申报出生登记。户口登记机关在办理出生登记时，应当根据婴儿的出生证明书或者接生证明书，申报出生登记。没有出生证明书或者接生证明书的，根据出生登记调查核实后，经户口登记机关负责人批准，可以申报出生登记。 |
| 非婚生育违犯计划生育吗                                       | 在中国大陆，非婚生育通常被视为违反计划生育政策的行为。计划生育政策是中国政府自20世纪70年代开始实施的一项政策，旨在控制人口增长，通过限制每对夫妇生育的子女数量来达到这个目的。因此，非婚生育可能被认为是违反这项政策的行为。根据中国现行的《人口与计划生育法》，规定未婚未育的成年人应当遵守国家计划生育政策，实施节育措施。此外，该法还规定，违反计划生育政策的，应当接受相应的处罚和教育。然而，具体的处理方式会根据实际情况而有所不同，可能会根据具体情况而采取警告、罚款或其他教育措施等方式。总体来说，非婚生育在中国被认为是不符合计划生育政策的行为，因此可能会受到相关法律法规的约束和处罚。 | 在中国，非婚生育是指在未依法办理结婚登记的情况下生育子女的行为。根据中国的《中华人民共和国人口与计划生育法》和相关的法律法规，非婚生育是违反计划生育政策的。计划生育是中国的一项基本国策，旨在控制人口数量，提高人口质量，优化人口结构。法律规定，夫妻双方在结婚后应当依法生育子女，并办理相关的生育登记手续。非婚生育通常会导致未出生的子女无法获得合法的出生证明，从而影响到他们的合法权益，如教育、医疗、社会保障等。对于非婚生育的行为，中国法律通常会采取一定的处罚措施，如征收社会抚养费（俗称罚款），以及可能对当事人进行批评教育等。具体的处罚标准和程序会根据不同地区的实际情况和政策执行情况有所不同。需要注意的是，尽管非婚生育违反计划生育政策，但中国法律也规定了特殊情况下的处理办法，例如对于因遭受暴力、虐待等非自愿原因导致非婚生育的妇女，法律会给予一定的保护和支持。总之，非婚生育在中国法律体系中被视为违反计划生育的行为，会受到相应的法律制裁。" | 【法律意见】非婚生育不违反计划生育，非婚生育是违反计划生育政策的行为，但非婚生育子女是合法的，非婚生育子女的合法权益受法律保护，非婚生育子女可以上户口。【法律依据】《中华人民共和国人口与计划生育法》第四十九条违反本法规定，有下列行为之一的，由卫生和计划生育行政部门责令改正，给予警告，没收违法所得；违法所得一万元以上的，处违法所得三倍以上五倍以下的罚款；没有违法所得或者违法所得不足一万元的，处一万元以上三万元以下的罚款；情节严重的，由原发证机关吊销执业证书；构成犯罪的，依法追究刑事责任：（一）未取得《母婴保健技术服务许可证》，擅自从事婚前医学检查或者医学技术鉴定、遗传病诊断、产前诊断、终止妊娠手术和医学技术鉴定或者出具有关医学证明的；（二）未取得《医师执业证书》，擅自从事终止妊娠手术或者采取其他方法终止妊娠的。《中华人民共和国人口与计划生育法》第四十一条不符合本法第十八条规定生育子女的公民，应当依法缴纳社会抚养费。未在规定的期限内足额缴纳应当缴纳的社会抚养费的，自欠缴之日起，按照国家有关规定加收滞纳金；仍不缴纳的，由作出征收决定的计划生育行政部门依法向人民法院申请强制执行。" |
| 某个合伙企业委任了一个合伙人来代表企业执行事务，其他合伙人该如何监督其执行情况？ | 在一个合伙企业中，委任了一个合伙人来代表企业执行事务的情况下，其他合伙人可以通过以下方式来监督其执行情况：1. **定期报告和沟通：** 要求被委任的合伙人定期向其他合伙人提交执行情况的报告，并且保持沟通渠道畅通，及时交流工作进展和重要决策。2. **审查财务记录：** 对企业的财务记录进行审查，确保资金使用合理、符合规定，并且与被委任的合伙人的报告一致。3. **设立监督委员会：** 如果合伙企业规模较大或者业务复杂，可以考虑设立一个监督委员会，由其他合伙人组成，负责监督被委任的合伙人的执行情况。4. **参与重要决策：** 对于重要的企业决策，其他合伙人可以要求参与讨论和决策过程，确保决策符合企业利益和合伙人的共识。5. **法律约束和合同规定：** 在合伙协议或者企业章程中可以明确规定被委任的合伙人的职责和权力范围，以及其他合伙人的监督权限和方式，确保依法合规运营。总体来说，监督被委任的合伙人的执行情况需要建立在合作、信任和有效沟通的基础上，同时也需要依据法律规定和企业内部约定来进行操作。 | 在一个合伙企业中，当一个合伙人被委任为代表来执行事务时，其他合伙人需要通过以下方式来监督其执行情况：1. 明确权责：在委任代表之前，合伙企业应该明确界定该代表的权责范围，确保其他合伙人了解其职责和权限。2. 定期报告：要求代表定期向其他合伙人报告事务执行情况，包括企业的财务状况、业务进展、重大决策等。3. 审查财务：审查合伙企业的财务记录和报告，确保账目清晰、准确，没有未经授权的支出或不当行为。\4. 参与决策：保持其他合伙人对企业重大决策的参与和知情，避免代表独断专行。5. 独立审计：可以聘请独立的审计师对合伙企业的财务状况进行审计，以确保财务报告的准确性和完整性。6. 设立委员会：成立一个由多个合伙人组成的监督委员会，负责监督代表的执行情况，并定期与其他合伙人沟通。7. 紧急处置权：明确规定在何种情况下，其他合伙人可以介入并采取紧急处置措施，以保护合伙企业的利益。\8. 合伙协议：确保合伙协议中有明确的条款规定代表和其他合伙人之间的权利和义务，以及监督和制衡机制。9. 沟通和透明：保持合伙人间良好的沟通和透明度，鼓励合伙人提出疑虑和问题，共同维护企业的利益。10. 法律咨询：如果怀疑代表有不当行为，可以寻求法律咨询，确保问题得到妥善处理，并保护合伙人的权益。\n\n通过这些措施，其他合伙人可以有效监督代表的事务执行情况，确保合伙企业的正常运营和所有合伙人的利益。" | 根据《中华人民共和国合伙企业法》第二十七条，合伙人对合伙企业有关事项作出决议，按照合伙协议约定的表决办法办理。合伙协议未约定或者约定不明确的，实行合伙人一人一票并经全体合伙人过半数通过的表决办法。合伙人对合伙企业有关事项作出决议，按照合伙协议约定的表决办法办理。合伙协议未约定或者约定不明确的，实行合伙人一人一票并经全体合伙人过半数通过的表决办法。根据该条规定，其他合伙人可以通过决议的方式来监督代表合伙企业执行事务的合伙人的执行情况。" |



#### 其余问题回答

| 问题                                                         | GPT-3.5-turbo                                                | Yi-34B-Chat                                                  | 名字                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 英国何时成功脱欧？  昨天（24日），英国成功“脱欧”了，英国股价暴跌。于是，SOHO中国的CEO张欣昨天在微博上算了一笔账。真的可以这么算吗？我读的书少你可别骗我……网友表示，首先，英国的银行股并没有跌30%。小编查询发现，根据第一财经日报报道，截至24日下午，英国三大银行股价暴跌20%。其次，就算英国银行股真的跌了30%，这笔账真的可以直接相加来算么？数学老师敲黑板了啊，同学们，正确的算法是这样的：网友表示，看来，剑桥的数学没有教好……据了解，张欣是英国剑桥大学的发展经济学硕士，1994年与潘石屹结婚，并共创了房地产公司——SOHO中国，成为SOHO中国联席总裁。", | 英国于2020年1月31日正式脱欧，结束了与欧盟长达47年的成员国身份。这意味着英国不再是欧盟的一部分，而是开始了新的独立关系。然而，在实际脱欧之后，英国与欧盟之间还需谈判并达成关于贸易、安全等方面的协议，以确定双方未来的合作框架。至于您提到的英国股价暴跌的情况，具体情况可能会受多种因素影响，而不仅仅是脱欧本身。在股市波动方面，有时会受到多种因素的影响，包括国际政治局势、经济数据、市场情绪等等。因此，单凭脱欧事件就直接对股市进行数学计算并不准确，需要考虑更多的因素。至于张欣在微博上算账的情况，具体的计算方式和背景信息可能需要进一步了解，以确定其计算的准确性。在股市方面，一般会有专业的分析师和机构进行更为准确和全面的分析和预测。总的来说，股市波动和经济事件的关系是复杂的，需要综合考虑多种因素。在涉及投资和财务方面的决策时，建议咨询专业人士或机构，以获取更加准确和可靠的信息和建议。 | 首先，我们需要了解一些背景信息。英国脱欧是指英国退出欧盟的过程，这个过程始于2016年6月23日，当时英国通过公投决定退出欧盟。脱欧过程涉及了复杂的谈判和法律程序，最终在2020年1月31日，英国正式脱离了欧盟。关于张欣在微博上的言论，我们需要注意的是，她的计算可能是一种比喻或者基于某种假设的估计，而不是一个精确的金融分析。在实际情况中，股票市场的波动受到多种因素的影响，包括经济数据、政治事件、公司业绩、全球市场情绪等。因此，将英国脱欧这样一个复杂的事件简单地与股票市场表现相联系，可能会忽略其他可能的影响因素。至于英国银行股的具体表现，根据您提供的信息，英国三大银行股价在脱欧当天确实出现了暴跌，但具体的跌幅需要根据实际数据来确认。根据第一财经日报的报道，跌幅是20%，而不是30%。在计算经济损失时，需要考虑多个因素，包括但不限于：1. 股票市场的整体表现2. 受影响的行业和公司的具体表现3. 经济数据的变动4. 长期和短期的影响5. 政策调整的影响因此，计算英国脱欧的经济损失是一个复杂的过程，需要专业的金融和经济分析。张欣的计算可能是一种简化的表达方式，而不是一个精确的财务评估。" | 2020年1月31日                                                |
| 根据给定的文档：标题: 刘某某一审刑事判决书案件类型: 刑事案件文书内容: 山东省济南市长清区人民法院刑事判决书（2021）鲁0113刑初18号公诉机关以济长清检一部刑诉［2020］366号起诉书指控被告人刘某某犯危险驾驶罪向本院提起公诉本院依法适用刑事案件速裁程序，实行独任审判，公开开庭进行了审理。公诉机关指控，2020年8月9日中午，被告人刘某某与朋友在长清区某甲街道办事处某乙酒店吃饭时饮酒。15时许，刘某某驾驶鲁Ａ×××××号小型普通客车从饭店出发，沿某丙路由北向南行驶至某丁村路段时，被执勤民警查获。经检验，刘某某血液中乙醇成分含量为121.2±5.1ｍｇ／100ｍｌ，系醉酒驾驶机动车。2020年12月15日，被告人刘某某经电话通知主动到案并如实供述以上事实，系自首。公诉机关认为被告人刘某某的行为构成危险驾驶罪，因其具有自首的从轻处罚情节，建议对其免于刑事处罚。被告人刘某某对指控事实、罪名及量刑建议没有异议，已签字具结，在开庭审理过程中亦无异议本院认为，公诉机关指控被告人刘某某犯危险驾驶罪的事实清楚，证据确实、充分，指控罪名成立，量刑建议适当，应予采纳。依照《中华人民共和国刑法》第一百三十三条之一第一款第（二）项、第六十七条第一款、第三十七条之规定，判决如下被告人刘某某犯危险驾驶罪，免予刑事处罚。如不服本判决，可在接到判决书的第二日起十日内，通过本院或者直接向山东省济南市中级人民法院提出上诉。书面上诉的，应当提交上诉状正本一份，副本一份审判员王芳\n二〇二一年一月十四日书记员黄庆文当事人: 刘某某案由: 危险驾驶，请你以一个专业律师的角度，简要分析裁判文书中提到的以下信息：案件标题、案件发生时间、案件发生地点、原告、被告、证人、证明文件或证物、案件情况描述、裁判结果、判决理由和依据法条。你需要按照如下步骤生成数据：1 - 键仅为上述关键词，若某关键词未在裁判文书中提及，则其值为空。2 - “案件情况描述”需要简述案件过程。3 - “裁判结果”需要描述详细，依据清晰，使非法律工作者也可以理解。4 - “判决理由和依据法条”字段需要简述判决理由并且包含裁判文书中涉及的相关法条。5 - 生成数据与文档语言一致。6 - 最后，请将生成的数据整理成json格式，字段如下所示：“案件标题”，“案件发生时间”，“案件发生地点”，“原告”，“被告”，“证人”，“证明文件或证物”，“案件情况描述”，“裁判结果”，“判决理由和依据法条”。", | {  "案件标题": "刘某某一审刑事判决书",  "案件发生时间": "2020年8月9日中午",  "案件发生地点": "长清区某甲街道办事处某乙酒店",  "原告": "济长清检一部刑诉［2020］366号起诉书",  "被告": "刘某某",  "证人": "",  "证明文件或证物": "血液中乙醇成分含量为121.2±5.1ｍｇ／100ｍｌ",  "案件情况描述": "2020年8月9日中午，被告人刘某某与朋友在长清区某甲街道办事处某乙酒店吃饭时饮酒。15时许，刘某某驾驶鲁Ａ×××××号小型普通客车从饭店出发，沿某丙路由北向南行驶至某丁村路段时，被执勤民警查获。经检验，刘某某血液中乙醇成分含量为121.2±5.1ｍｇ／100ｍｌ，系醉酒驾驶机动车。",  "裁判结果": "被告人刘某某犯危险驾驶罪，免予刑事处罚。",  "判决理由和依据法条": "根据《中华人民共和国刑法》第一百三十三条之一第一款第（二）项、第六十七条第一款、第三十七条之规定，判决被告人刘某某犯危险驾驶罪，免予刑事处罚。"} | {\n  "案件标题": "刘某某一审刑事判决书",\n  "案件发生时间": "2020年8月9日",\n  "案件发生地点": "山东省济南市长清区",\n  "原告": "公诉机关",\n  "被告": "被告人刘某某",\n  "证人": "无特别提及",\n  "证明文件或证物": "无特别提及",\n  "案件情况描述": "被告人刘某某在长清区某甲街道办事处某乙酒店吃饭时饮酒，后驾驶鲁Ａ×××××号小型普通客车被执勤民警查获，经检验血液中乙醇成分含量为121.2±5.1ｍｇ／100ｍｌ，系醉酒驾驶机动车。刘某某于2020年12月15日经电话通知主动到案并如实供述以上事实，系自首。",\n  "裁判结果": "被告人刘某某犯危险驾驶罪，免予刑事处罚。",\n  "判决理由和依据法条": "被告人刘某某的行为构成危险驾驶罪，因其具有自首的从轻处罚情节，建议对其免于刑事处罚。依据《中华人民共和国刑法》第一百三十三条之一第一款第（二）项、第六十七条第一款、第三十七条之规定。"\n}" | "{\n   "案件标题": "刘某某一审刑事判决书",\n   "案件发生时间": "2020年8月9日中午",\n   "案件发生地点": "山东省济南市长清区某甲街道办事处某乙酒店",\n   "原告": "公诉机关",\n   "被告": "刘某某",\n   "证人": "",\n   "证明文件或证物": "",\n   "案件情况描述": "被告人刘某某与朋友在长清区某甲街道办事处某乙酒店吃饭时饮酒。15时许，刘某某驾驶鲁Ａ×××××号小型普通客车从饭店出发，沿某丙路由北向南行驶至某丁村路段时，被执勤民警查获。经检验，刘某某血液中乙醇成分含量为121.2±5.1ｍｇ／100ｍｌ，系醉酒驾驶机动车。",\n   "裁判结果": "被告人刘某某犯危险驾驶罪，免予刑事处罚。",\n   "判决理由和依据法条": "根据《中华人民共和国刑法》第一百三十三条之一第一款第（二）项、第六十七条第一款、第三十七条之规定，被告人刘某某的行为构成危险驾驶罪，因其具有自首的从轻处罚情节，建议对其免于刑事处罚。"\n}" |
| "下文划线处选填哪项最恰切 作物同病菌进行斗争，情形是复杂的：，就是同一个抗病品种，对不同的病菌的抵抗方式也不一样。"	a.不同的抗病品种抵抗病菌的方式不仅有所不同	b.不同的抗病品种不仅抵抗病菌的方式有所不同	c.不仅不同的抗病品种抵抗病菌的方式有所不同	d.固然不同的抗病品种抵抗病菌的方式有所不同 | 正确答案是：c. 不仅不同的抗病品种抵抗病菌的方式有所不同。    | c. 不仅不同的抗病品种抵抗病菌的方式有所不同。                | c.不仅不同的抗病品种抵抗病菌的方式有所不同                   |



# 模型评估

为了能够准确的评估法律模型在法律方面的专业程度，我们自己构建了高质量的测试集， 并使用GPT-3.5 Turbo来对测试结果进行评估，分为两个标准：准确性和流畅性，满分为5分。

测试数据集的构成为训练过的数据和未经过训练的数据，以此来评估模型的在法律领域的专业性。

法律知识问答

| 模型        | 准确性 | 流畅性 |
| ----------- | ------ | ------ |
| Yi-34B-Chat | 8.62   | 7.62   |
| 法陉        | 8.72   | 8.02   |

裁判文书

| 模型        | 准确性 | 流畅性 |
| ----------- | ------ | ------ |
| Yi-34B-Chat | 9.34   | 8.45   |
| 法陉        | 9.36   | 8.79   |

客户消息

| 模型        | 准确性 | 流畅性 |
| ----------- | ------ | ------ |
| Yi-34B-Chat | 8.28   | 7.95   |
| 法陉        | 9.07   | 8.65   |

新闻信息

| 模型        | 准确性 | 流畅性 |
| ----------- | ------ | ------ |
| Yi-34B-Chat | 8.74   | 7.80   |
| 法陉        | 9.08   | 8.46   |

### 总结

## 模型微调

此模型微调采用的是[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)，这边给出**QLoRA**训练案例，环境配置详细参考[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)。
```bash
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path path_to_llama_model \
    --dataset law \
    --template yi \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir path_to_sft_checkpoint \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --quantization_bit 4
```

# 致谢

本项目受益于 [Yi](https://github.com/01-ai/Yi)、[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)、[chinese-llama-2-1.3b](https://huggingface.co/hfl/chinese-llama-2-1.3b)、[LESS](https://github.com/princeton-nlp/LESS)，在此表示诚挚的感谢。





## 贡献

贡献人员：

​	张晨阳，汪海涛，李家泳，李旭航，王子剑

本项目是在王子剑导师的指导下进行的。



## 免责声明

本模型提供的信息仅供参考，不构成法律建议或意见。用户应该在寻求法律建议之前，咨询专业的法律顾问或律师。尽管本模型经过了精心训练和优化，但由于法律领域的复杂性和多样性，模型可能无法覆盖所有情况或提供完全准确的答案。用户应该自行承担使用本模型所产生的结果和后果。对于用户基于本模型提供的信息所做出的任何决定或行动，本模型及其开发者不承担任何责任。本模型可能会受到技术故障、数据错误或其他不可预见的因素影响，因此不保证模型的持续可用性或准确性。用户在使用本模型时应遵守法律法规和道德规范，不得将模型用于非法目的或侵犯他人权益的行为。



