import re

from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.llms import SparkLLM
from langchain_openai import ChatOpenAI

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
# from langchain_community.embeddings import QianfanEmbeddingsEndpoint
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory, RunnableParallel
import os
from langchain_community.llms import Tongyi

# # 日志记录相关
from ..cognitive.experiment_logger import record_llm_call
from ..cognitive.experiment_logger import log_print


os.environ["IFLYTEK_SPARK_APP_ID"] = "Your App ID"
os.environ["IFLYTEK_SPARK_API_SECRET"] = "Your API Secret"
os.environ["IFLYTEK_SPARK_API_KEY"] = "Your API Key"

os.environ["OPENAI_API_KEY"] = "Your OpenAI Key"
os.environ["OPENAI_BASE_URL"] = "Your OpenAI Base URL"

os.environ["DASHSCOPE_API_KEY"] = "Your DashScope API Key"


# ollama模型白名单
OLLAMA_MODEL_LIST = {
    'think': [],
    'nothink': []
}

class LLMAgent:
    # 构造参数：
    #   agent_name*:str,agent名称，也可以使用ID替代，是区分agent对话记忆的唯一标识
    #   has_chat_history：布尔值，决定是否开启对话历史记忆，开启后agent会记住之前对其的所有询问与回答，默认开启。
    #   llm_model: str,调用大模型，目前支持“ChatGPT”，“Spark”
    #   online_track：bool,是否开启langsmith线上追踪
    #   json_format：bool,是否以json格式做出回答
    #   system_prompt = ''
    def __init__(self,
                 agent_name,
                 has_chat_history=True,
                 llm_model="ChatGPT",
                 online_track = False,
                 json_format = True,
                 system_prompt = ''
                 ):
        self.system_prompt = system_prompt
        self.agent_name = agent_name
        self.has_chat_history = has_chat_history
        self.llm_model = llm_model
        self.online_track = online_track
        self.json_format = json_format
    #   调用参数
    #   system_prompt:str,系统提示词
    #   user_prompt:str,用户提示词
    #   input_param_dict:参数列表字典，该字典可以替换prompt中的待定参数
    #   is_first_call:布尔值，若为第一次调用，则清空该agent_name对应的数据库。否则继承对话记忆
    def get_response(self, user_template, new_system_template = None,input_param_dict=None, is_first_call=False, flag_debug_print=False, country_name=None):
        if input_param_dict is None:
            input_param_dict = {}
        if new_system_template is None:
            system_template = self.system_prompt
        else:
            system_template = new_system_template
        if self.online_track:
            pass

        # 1. Create prompt template
        if self.json_format:
            user_template += "\nPlease give your response in JSON format.Return a JSON object."
        if self.has_chat_history:
            system_template = PromptTemplate.from_template(system_template).invoke(input_param_dict).to_string()
            user_template = PromptTemplate.from_template(user_template).invoke(input_param_dict).to_string()
            prompt_template = ChatPromptTemplate.from_messages([
                ('system', system_template),
                MessagesPlaceholder(variable_name="history"),
                ('user',  user_template),
            ])
        else:
            prompt_template = ChatPromptTemplate.from_messages([
                ('system', system_template),
                ('user', user_template),
            ])
        # prompt_template.invoke(input_param_dict)

        # 2. Create model
        if self.llm_model == 'ChatGPT':
            raise Exception('ChatGPT API is not available')
        elif self.llm_model == 'Spark':
            model = SparkLLM(
                api_url='ws://spark-api.xf-yun.com/v1.1/chat',
                model='lite'
            )
        elif self.llm_model == 'qwen3-max':
            model = Tongyi(
                model="qwen3-max",
            )
        elif self.llm_model == 'qwen-turbo':
            model = Tongyi(
                model="qwen-turbo",
            )
        else:
            try:
                model = ChatOpenAI(
                    model=self.llm_model,
                )
            except Exception as e:
                print(e)
                raise Exception(f'LLMAgent.llm_model ({self.llm_model}) is not allowed')

        # 3. Create parser
        if self.json_format:
            parser = JsonOutputParser()
        else:
            parser = StrOutputParser()

        # 4. Create chain
        if self.has_chat_history: # TODO: 当前不可用！！！待修改
            pass
        else:
            if self.llm_model in OLLAMA_MODEL_LIST['think']:
                try:
                    chain = prompt_template| model

                    # 记录LLM调用
                    call_details = f"Agent: {self.agent_name}, Model: {self.llm_model} (think mode)"
                    record_llm_call(call_details)
                    result = chain.invoke(input_param_dict)

                    pattern = r"<think>(.*?)</think>"
                    think = re.findall(pattern, str(result), re.DOTALL)[0]
                    if flag_debug_print:
                        print("下面仅为思维内容")
                        print(think)
                    result = re.sub(pattern, '', str(result), flags=re.DOTALL)
                    if flag_debug_print:
                        print("下面仅为删除思维后的结果")
                        print(result)
                    result = parser.invoke(result)
                except Exception as e:
                    log_print("下面为错误信息", level="ERROR")
                    log_print(e, level="ERROR")
                    log_print(f"user_template: {user_template}", level="ERROR")
                    log_print(f"input_param_dict: {input_param_dict}", level="ERROR")
                    log_print(f"system_template: {system_template}", level="ERROR")
                    return 'llm报错','llm报错'
            else:
                try:
                    chain = (prompt_template|
                            model|
                            parser)

                    # 记录LLM调用
                    call_details = f"Agent: {self.agent_name}, Model: {self.llm_model}"
                    record_llm_call(call_details)

                    result = chain.invoke(input_param_dict)

                    if flag_debug_print:
                        print(result)
                except Exception as e:
                    log_print("下面为错误信息", level="ERROR")
                    log_print(e, level="ERROR")
                    log_print(f"user_template: {user_template}", level="ERROR")
                    log_print(f"input_param_dict: {input_param_dict}", level="ERROR")
                    log_print(f"system_template: {system_template}", level="ERROR")
                    # log_print(f"result: {result}", level="ERROR")
                    return e

        if self.llm_model in OLLAMA_MODEL_LIST['think']:
            return result, think
        else:
            return result


