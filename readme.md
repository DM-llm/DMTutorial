
# Lagent 自定义你的 Agent 智能体
## Agent介绍
智能体，即Agent，是一种能自主规划，完成人类所指定的任务的AI工具

  


OpenAI应用研究主管翁丽莲(Lilian Weng)撰写过一篇blog: [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/)，将 Agents 定义为**LLM + memory + planning skills + tool use**，即大语言模型、记忆、任务规划、工具使用的集合。

  ![image](https://github.com/user-attachments/assets/1fb582fa-b3cb-4f1b-9570-9af2c96cb137)

![image](https://github.com/user-attachments/assets/c7c0896b-f248-4db6-a4a0-a482ea418fbf)


其中，LLM是Agent的大脑，属于“中枢”模型，要求有以下3种能力：

1.  planning skills：对问题进行拆解得到解决路径，既进行任务规划
1.  tool use：评估自己所需的工具，进行工具选择，并生成调用工具请求
1.  memory：短期记忆包括工具的返回值，已经完成的推理路径；长期记忆包括可访问的外部长期存储，例如知识库
<https://datawhaler.feishu.cn/sync/PYnsdkDa6sgvLYbNeTTcWZGanTc>


## Lagent 介绍

Lagent 是一个轻量级开源智能体框架，旨在让用户可以高效地构建基于大语言模型的智能体。同时它也提供了一些典型工具以增强大语言模型的能力。

Lagent 目前已经支持了包括 AutoGPT、ReAct 等在内的多个经典智能体范式，也支持了如下工具：

- Arxiv 搜索
- Bing 地图
- Google 学术搜索
- Google 搜索
- 交互式 IPython 解释器
- IPython 解释器
- PPT
- Python 解释器

其基本结构如下所示：

![image](https://github.com/InternLM/lagent/assets/24351120/cefc4145-2ad8-4f80-b88b-97c05d1b9d3e)

## 环境配置

开发机选择 30% A100，镜像选择为 Cuda12.2-conda。

首先来为 Lagent 配置一个可用的环境。

```bash
# 创建环境
conda create -n agent_camp4 python=3.10 -y
# 激活环境
conda activate agent_camp4
# 安装 torch
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y
# 安装其他依赖包
pip install termcolor==2.4.0
pip install lmdeploy==0.5.2
pip install class-registry
```

接下来，我们通过源码安装的方式安装 lagent。

```bash
# 创建目录以存放代码
mkdir -p /root/agent_camp4
cd /root/agent_camp4
git clone https://github.com/InternLM/lagent.git
cd lagent && git checkout 81e7ace && pip install -e . && cd ..
pip install griffe==0.48.0

```

## Lagent Web Demo 使用

接下来，我们将使用 Lagent 的 Web Demo 来体验 InternLM2.5-7B-Chat 的智能体能力。

首先，我们先使用 LMDeploy 部署 InternLM2.5-7B-Chat，并启动一个 API Server。

```bash
conda activate agent_camp4
lmdeploy serve api_server /share/new_models/Shanghai_AI_Laboratory/internlm2_5-7b-chat --model-name internlm2_5-7b-chat
```
![image](https://github.com/user-attachments/assets/99f3c0f3-427d-4d37-8360-efd3ab1cfdf5)


然后，我们在另一个**窗口**中启动 Lagent 的 Web Demo。

```bash
cd /root/agent_camp4/lagent
conda activate agent_camp4
streamlit run examples/internlm2_agent_web_demo.py
```
![image](https://github.com/user-attachments/assets/da7cf168-6b7f-447c-8db4-a514732a2f91)


在等待两个 server 都完全启动（如下图所示）后，我们在 **本地** 的 PowerShell 中输入如下指令来进行端口映射：

```bash
ssh -p 44628 root@ssh.intern-ai.org.cn -CNg -L {本地机器_PORT}:127.0.0.1:{开发机_PORT} -o StrictHostKeyChecking=no
```
![image](https://github.com/user-attachments/assets/26e4b8a8-519f-4f67-9ffa-ca35aa4cf5a6)




接下来，在本地浏览器中打开 `localhost:8501`，并修改**模型名称**一栏为 `internlm2_5-7b-chat`，修改**模型 ip**一栏为`127.0.0.1:23333`。

> [!IMPORTANT]
> 输入后需要按下回车以确认！

然后，我们在插件选择一栏选择 `ArxivSearch`，并输入指令“帮我搜索一下 MindSearch 论文”。

![Web Demo](https://github.com/user-attachments/assets/34ac1001-8bfa-4d2a-8346-d871a0e0f03c)

最后，可以看到，模型已经回复了相关信息。

![result](https://github.com/user-attachments/assets/d21b64c2-acf4-48e1-a1e5-73775e6b36d4)

## 基于 Lagent 自定义智能体

在本节中，我们将带大家基于 Lagent 自定义自己的智能体。

Lagent 中关于工具部分的介绍文档位于 https://lagent.readthedocs.io/zh-cn/latest/tutorials/action.html 。

使用 Lagent 自定义工具主要分为以下几步：

1. 继承 `BaseAction` 类
2. 实现简单工具的 `run` 方法；或者实现工具包内每个子工具的功能
3. 简单工具的 `run` 方法可选被 `tool_api` 装饰；工具包内每个子工具的功能都需要被 `tool_api` 装饰

下面我们将实现一个调用 MagicMaker API 以完成文生图的功能。

首先，我们先来创建工具文件：

```bash
cd /root/agent_camp4/lagent
touch lagent/actions/magicmaker.py
```

然后，我们将下面的代码复制进入 `/root/agent_camp4/lagent/lagent/actions/magicmaker.py`

```python
import json
import requests

from lagent.actions.base_action import BaseAction, tool_api
from lagent.actions.parser import BaseParser, JsonParser
from lagent.schema import ActionReturn, ActionStatusCode


class MagicMaker(BaseAction):
    styles_option = [
        'dongman',  # 动漫
        'guofeng',  # 国风
        'xieshi',   # 写实
        'youhua',   # 油画
        'manghe',   # 盲盒
    ]
    aspect_ratio_options = [
        '16:9', '4:3', '3:2', '1:1',
        '2:3', '3:4', '9:16'
    ]

    def __init__(self,
                 style='guofeng',
                 aspect_ratio='4:3'):
        super().__init__()
        if style in self.styles_option:
            self.style = style
        else:
            raise ValueError(f'The style must be one of {self.styles_option}')
        
        if aspect_ratio in self.aspect_ratio_options:
            self.aspect_ratio = aspect_ratio
        else:
            raise ValueError(f'The aspect ratio must be one of {aspect_ratio}')
    
    @tool_api
    def generate_image(self, keywords: str) -> dict:
        """Run magicmaker and get the generated image according to the keywords.

        Args:
            keywords (:class:`str`): the keywords to generate image

        Returns:
            :class:`dict`: the generated image
                * image (str): path to the generated image
        """
        try:
            response = requests.post(
                url='https://magicmaker.openxlab.org.cn/gw/edit-anything/api/v1/bff/sd/generate',
                data=json.dumps({
                    "official": True,
                    "prompt": keywords,
                    "style": self.style,
                    "poseT": False,
                    "aspectRatio": self.aspect_ratio
                }),
                headers={'content-type': 'application/json'}
            )
        except Exception as exc:
            return ActionReturn(
                errmsg=f'MagicMaker exception: {exc}',
                state=ActionStatusCode.HTTP_ERROR)
        image_url = response.json()['data']['imgUrl']
        return {'image': image_url}

```

最后，我们修改 `/root/agent_camp4/lagent/examples/internlm2_agent_web_demo.py` 来适配我们的自定义工具。

1. 在 `from lagent.actions import ActionExecutor, ArxivSearch, IPythonInterpreter` 的下一行（第9行）——添加 `from lagent.actions.magicmaker import MagicMaker`
![image](https://github.com/user-attachments/assets/b39da521-dc46-4ced-9843-227fa5e56de1)


3. 在第27行添加 `MagicMaker()`。
   
![image](https://github.com/user-attachments/assets/fb5fa737-bc63-4d64-b1a4-5390148be7a6)



接下来，启动 Web Demo 来体验一下吧！我们同时启用两个工具，然后输入“请帮我生成一幅山水画”

![instruction](https://github.com/user-attachments/assets/699308cd-6b17-4515-a42e-d120bd8e9a2b)

![result](https://github.com/user-attachments/assets/c62cea67-1b9f-4a45-ba7f-6c5836d6db7e)

然后，我们再试一下“帮我搜索一下 MindSearch 论文”。

![result](https://github.com/user-attachments/assets/03a39808-db97-4321-883e-7a0446e95343)
