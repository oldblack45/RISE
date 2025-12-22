
from simulation.models.agents.LLMAgent import LLMAgent
x=LLMAgent(agent_name="X", has_chat_history=False, online_track=False, json_format=False, system_prompt='',
                         llm_model='gemma3:27b-q8')
x.get_response(user_template="hello",input_param_dict={})