from LLMAgent import LLMAgent

x = LLMAgent(agent_name="X", has_chat_history=False, online_track=True, json_format=False, system_prompt='',
                         llm_model='gpt-4o')
result =x.get_response(user_template="hello", input_param_dict={})
print(result)