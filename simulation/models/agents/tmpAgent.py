
from simulation.models.agents.LLMAgent import LLMAgent
x=LLMAgent(agent_name="X", has_chat_history=False, online_track=False, json_format=False, system_prompt='',
                         llm_model='qwen3-235b-a22b:q4')
x.get_response(user_template="hello",input_param_dict={})