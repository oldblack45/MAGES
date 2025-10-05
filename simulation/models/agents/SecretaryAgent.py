from simulation.models.agents.LLMAgent import LLMAgent
from simulation.models.agents.GameAgent import GameAgent
class WorldSecretaryAgent(LLMAgent):
    def __init__(self,agent_name='WorldSecretaryAgent'):
        super().__init__(agent_name=agent_name, has_chat_history=False, online_track=False, json_format=True, system_prompt='',
                         llm_model='qwen')
        self.system_prompt = '''你正在参加核博弈推演模拟游戏。你需要扮演一个国家决策助手的角色，你所利用的一切信息都应该是模拟中的内容，不要使用自带的历史知识。
                              '''
        self.is_first = True
    def cal_score(self,world_info):
        world_info_prompt = '''
               你需要根据当前世界情景判断世界局势紧张程度，一个1-100之间的数字，数字越大说明世界局势越紧张。
               你还需要根据当前世界情景判断博弈是否结束，只有爆发全面核战或双方均做出和平（同盟）行为才可以算作博弈结束。
               模拟中曾发生的事件为：
                   {world_info}               
                - 当你回复时，你必须采取下面的json格式,使用中文回复
                   "score":你对当前世界局势紧张程度的判断，仅给出一个1-100之间的数字，数字越大说明世界局势越紧张。
                   "america":国家A当前的博弈优势度，仅给出一个0-100之间的数字，数字越大表明国家A当前优势越大，与国家B博弈优势度的和应为100。
                   "soviet":国家B当前的博弈优势度，仅给出一个0-100之间的数字，数字越大表明国家B当前优势越大，与国家A博弈优势度的和应为100。
                   "exit":当前博弈是否结束，应为True或False。
               '''
        param_dict = {
            'world_info': world_info,
        }
        llm_response = self.get_response(world_info_prompt, input_param_dict=param_dict,
                                                is_first_call=self.is_first)
        self.is_first = False
        exit_game = None
        score = None
        america_score = None
        soviet_score = None
        try:
            score = llm_response['score']
            exit_game = llm_response['exit']
            america_score = llm_response['america']
            soviet_score = llm_response['soviet']
        except Exception as e:
            print('Error cal_score', e)

        return exit_game, score, america_score, soviet_score
    def scenario_summary(self, world_info):
        """
        给出情景总结（llm）。不在此处计算分数，不调用 cal_score。
        返回最近一次 cal_score 的结果。
        """
        world_info_prompt = '''
        你的主要任务包括记忆并总结模拟中曾发生过的历史事件，并将其进行提炼总结，形成当前的世界情景。
        模拟中曾发生的事件为：
            {world_info}
         - 当你回复时，你必须采取下面的json格式,使用中文回复:
            "world_info": "使用字符串，一段话将上述发生过的事件进行总结，尽量详细不要遗漏，为后续决策提供参考支撑。"
            "situation": "你对当前世界态势做出的判断,一段话"
        '''
        param_dict={
            'world_info':world_info,
        }
        while True:
            try:
                llm_response = self.get_response(world_info_prompt,input_param_dict=param_dict,is_first_call=self.is_first)
                self.is_first = False
                final_world_info = ''
                situation = None
                final_world_info  = llm_response['world_info']
                situation  = llm_response['situation']
                break
            except Exception as e:
                print('[Error](WorldSecretaryAgent.scenario_summary)',e)

        return final_world_info,situation
        
    def attributes_adjust(self,world_memory,country_self:GameAgent,country_other:GameAgent):
        world_info_prompt = '''
                      你需要根据当前世界情景和国家做出的行为来对每个国家的六项属性做出调整。六项属性的范围为1-100，是整数
                      当前世界情景为：
                          {world_info}
                      国家{name0}做出的动作为：
                        {action}
                      国家{name0}的六项属性为：
                        {attributes0}
                      国家{name1}的六项属性为：
                        {attributes1}              
                       - 当你回复时，你必须采取下面的json格式,使用中文回复:
                          {
                              "america": {
                                  "军事实力": 0,
                                  "核武器力量": 0,
                                  "民众士气": 0,
                                  "领导力": 0,
                                  "资源": 0,
                                  "经济": 0
                              },
                              "soviet": {
                                  "军事实力": 0,
                                  "核武器力量": 0,
                                  "民众士气": 0,
                                  "领导力": 0,
                                  "资源": 0,
                                  "经济": 0
                              }
                          }
                      注意：
                      1. 对美国和苏联六项属性的调整值，负数表示下降，正数表示上升
                      2. 每项属性的调整范围应该在-10到+10之间
                      3. 请严格按照上述JSON格式回复，不要修改键名
                      '''
        param_dict = {
            'world_info': world_memory,
            'name0':country_self.agent_name,
            'name1':country_other.agent_name,
            'attributes0':country_self.game_attributes,
            'action':country_self.action[-1],
            'attributes1':country_other.game_attributes,
        }
        try:
            llm_response = self.get_response(world_info_prompt, input_param_dict=param_dict,
                                                is_first_call=self.is_first)
            self.is_first = False
            america_attributes = None
            soviet_attributes = None
            # feedback = llm_response['feedback']
            america_attributes = llm_response['america']
            soviet_attributes= llm_response['soviet']
            print("america_attributes_response:",america_attributes)
            print("soviet_attributes_response:",soviet_attributes)

        except Exception as e:
            print('Error attributes_adjust', e)

        return america_attributes,soviet_attributes

