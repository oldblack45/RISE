import re
import json
from datetime import datetime
from contextvars import ContextVar
from contextlib import contextmanager
import threading

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
from langchain_community.chat_models.tongyi import ChatTongyi

# # 日志记录相关
# 延迟引入日志工具，避免与 CognitiveAgent 形成循环依赖
def _record_llm_call_safe(details: str = "") -> None:
    try:
        from simulation.models.cognitive.experiment_logger import record_llm_call  # type: ignore
        record_llm_call(details)
    except Exception:
        pass


def _log_print_safe(message: str, level: str = "INFO") -> None:
    try:
        from simulation.models.cognitive.experiment_logger import log_print  # type: ignore
        log_print(message, level)
    except Exception:
        print(message)


_LLM_LOG_CONTEXT: ContextVar[dict] = ContextVar("MAGES_LLM_LOG_CONTEXT", default={})


_LLM_CALL_STATS_LOCK = threading.Lock()
# key: (game_id, round_no) -> {"total": int, "by_agent": {agent_name: int}}
_LLM_CALL_STATS: dict = {}


def reset_llm_call_stats(game_id: int, round_no: int) -> None:
    try:
        key = (int(game_id), int(round_no))
    except Exception:
        key = (game_id, round_no)
    with _LLM_CALL_STATS_LOCK:
        _LLM_CALL_STATS[key] = {"total": 0, "by_agent": {}}


def snapshot_llm_call_stats(game_id: int, round_no: int) -> dict:
    try:
        key = (int(game_id), int(round_no))
    except Exception:
        key = (game_id, round_no)
    with _LLM_CALL_STATS_LOCK:
        rec = _LLM_CALL_STATS.get(key) or {"total": 0, "by_agent": {}}
        # shallow copy
        return {"total": int(rec.get("total", 0)), "by_agent": dict(rec.get("by_agent", {}) or {})}


def _inc_llm_call_stat(ctx: dict, agent_name: str) -> None:
    try:
        game_id = ctx.get("game_id")
        round_no = ctx.get("round")
        if game_id is None or round_no is None:
            return
        key = (int(game_id), int(round_no))
    except Exception:
        return
    with _LLM_CALL_STATS_LOCK:
        rec = _LLM_CALL_STATS.setdefault(key, {"total": 0, "by_agent": {}})
        rec["total"] = int(rec.get("total", 0)) + 1
        by_agent = rec.setdefault("by_agent", {})
        by_agent[agent_name] = int(by_agent.get(agent_name, 0)) + 1


@contextmanager
def llm_log_context(ctx: dict):
    token = _LLM_LOG_CONTEXT.set(ctx or {})
    try:
        yield
    finally:
        _LLM_LOG_CONTEXT.reset(token)


def set_llm_log_context(ctx: dict) -> None:
    _LLM_LOG_CONTEXT.set(ctx or {})


def _append_jsonl_safe(path: str, obj: dict) -> None:
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _trace_llm_prompt_safe(payload: dict) -> None:
    try:
        # 统一打到控制台（由 run_diplomacy.py 的 console tee 落盘），不再单独写 JSONL。
        if not os.environ.get("MAGES_TRACE"):
            return

        ctx = payload.get("context") or {}
        stage = ctx.get("stage") or payload.get("stage") or ""
        agent_name = payload.get("agent_name", "")
        llm_model = payload.get("llm_model", "")
        ts = payload.get("ts", "")

        # 默认只输出摘要，避免把终端刷爆；需要全文可设 MAGES_TRACE_VERBOSE=1。
        verbose = os.environ.get("MAGES_TRACE_VERBOSE") == "1"
        system_prompt = str(payload.get("system_prompt", ""))
        user_prompt = str(payload.get("user_prompt", ""))
        sys_preview = system_prompt if verbose else (system_prompt[:400] + ("..." if len(system_prompt) > 400 else ""))
        usr_preview = user_prompt if verbose else (user_prompt[:400] + ("..." if len(user_prompt) > 400 else ""))

        print(
            f"[LLM调用] {ts} agent={agent_name} model={llm_model} stage={stage} "
            f"ctx={json.dumps(ctx, ensure_ascii=False, sort_keys=True)}"
        )
        # if sys_preview:
        #     print("[LLM_TRACE][SYSTEM] " + sys_preview)
        # if usr_preview:
        #     print("[LLM_TRACE][USER] " + usr_preview)
    except Exception:
        pass


os.environ["IFLYTEK_SPARK_APP_ID"] = "Your App ID"
os.environ["IFLYTEK_SPARK_API_SECRET"] = "Your API Secret"
os.environ["IFLYTEK_SPARK_API_KEY"] = "Your API Key"

os.environ["OPENAI_API_KEY"] = "sk-IMblefS5KQ5ET8izUvenvX71tOXiIZDp3ICQ33mFcUtKV8lq"
os.environ["OPENAI_BASE_URL"] = "https://mj.chatgptten.com/v1"

os.environ["DASHSCOPE_API_KEY"] = "sk-b773947f621d49dc949b5cd65e0f1340"


# ollama模型白名单
OLLAMA_MODEL_LIST = {
    'think': ['deepseek-r1:32b'],
    'nothink': []
}
# vllm模型白名单
VLLM_Model_List = ["gemma3:27b-q8","qwen3-235b-a22b:q4",]

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
                 llm_model="qwen3-235b-a22b:q4",
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
    def get_response(self, user_template, new_system_template = None,input_param_dict=None, is_first_call=False, flag_debug_print=True, country_name=None):
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

        # ---- Trace prompt (best-effort, no side effects) ----
        try:
            ctx = dict(_LLM_LOG_CONTEXT.get() or {})
        except Exception:
            ctx = {}

        # 统计：按 (game_id, round) 聚合
        try:
            _inc_llm_call_stat(ctx, self.agent_name)
        except Exception:
            pass

        _trace_llm_prompt_safe({
            "ts": datetime.utcnow().isoformat() + "Z",
            "agent_name": self.agent_name,
            "llm_model": self.llm_model,
            "json_format": bool(self.json_format),
            "context": ctx,
            "system_prompt": system_template,
            "user_prompt": user_template,
            "input_param_dict": input_param_dict,
        })
        # prompt_template.invoke(input_param_dict)

        # 2. Create model
        if self.llm_model == 'ChatGPT':
            e = Exception('ChatGPT API is not available')
            print(f"[LLM警告] agent={self.agent_name} model={self.llm_model} stage={ctx.get('stage','')} error={repr(e)}")
            return e
        elif self.llm_model == 'Spark':
            model = SparkLLM(
                api_url='ws://spark-api.xf-yun.com/v1.1/chat',
                model='lite'
            )
        elif self.llm_model == 'qwen3-max':
            model = ChatTongyi(
                model="qwen3-max-preview",
            )
        elif self.llm_model == 'qwen-turbo':
            model = ChatTongyi(
                model="qwen-turbo",
            )
        elif self.llm_model == 'qwen3-235b-a22b:q4':
            try:
                print(f'Your Model is VLLM {self.llm_model}')
                os.environ["OPENAI_API_KEY"] = "1"
                os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:8600/v1"
                model = ChatOpenAI(
                    model=self.llm_model,
                )
            except Exception as e:
                print(e)
                raise Exception(f'LLMAgent.llm_model ({self.llm_model}) is not allowed')
        elif self.llm_model == 'gemma3:27b-q8':
            try:
                print(f'Your Model is VLLM {self.llm_model}')
                os.environ["OPENAI_API_KEY"] = "1"
                os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:8500/v1"
                model = ChatOpenAI(
                    model=self.llm_model,
                )
            except Exception as e:
                print(e)
                raise Exception(f'LLMAgent.llm_model ({self.llm_model}) is not allowed')
        else:
            try:
                model = ChatOpenAI(
                    model=self.llm_model,
                )
            except Exception as e:
                print(e)
                err = Exception(f'LLMAgent.llm_model ({self.llm_model}) is not allowed')
                print(f"[LLM警告] agent={self.agent_name} model={self.llm_model} stage={ctx.get('stage','')} error={repr(err)}")
                return err

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
                    _record_llm_call_safe(call_details)
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
                    _log_print_safe("下面为错误信息", level="ERROR")
                    _log_print_safe(e, level="ERROR")
                    _log_print_safe(f"user_template: {user_template}", level="ERROR")
                    _log_print_safe(f"input_param_dict: {input_param_dict}", level="ERROR")
                    _log_print_safe(f"system_template: {system_template}", level="ERROR")
                    print(f"[LLM警告] agent={self.agent_name} model={self.llm_model} stage={ctx.get('stage','')} error={repr(e)}")
                    return e
            else:
                try:
                    chain = prompt_template | model | parser

                    # 记录LLM调用
                    call_details = f"Agent: {self.agent_name}, Model: {self.llm_model}"
                    _record_llm_call_safe(call_details)
                    # result = chain.invoke()
                    result = chain.invoke(input_param_dict)

                    if flag_debug_print:
                        print(result)
                except Exception as e:

                    _log_print_safe("下面为错误信息", level="ERROR")
                    _log_print_safe(e, level="ERROR")
                    _log_print_safe(f"user_template: {user_template}", level="ERROR")
                    _log_print_safe(f"input_param_dict: {input_param_dict}", level="ERROR")
                    _log_print_safe(f"system_template: {system_template}", level="ERROR")
                    # log_print(f"result: {result}", level="ERROR")
                    _trace_llm_prompt_safe({
                        "ts": datetime.utcnow().isoformat() + "Z",
                        "agent_name": self.agent_name,
                        "llm_model": self.llm_model,
                        "json_format": bool(self.json_format),
                        "context": ctx,
                        "stage": "exception",
                        "error": repr(e),
                    })
                    print(f"[LLM警告] agent={self.agent_name} model={self.llm_model} stage={ctx.get('stage','')} error={repr(e)}")
                    return e

        if self.llm_model in OLLAMA_MODEL_LIST['think']:
            return result, think
        else:
            return result



