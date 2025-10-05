from simulation.models.agents.LLMAgent import LLMAgent
import textwrap
class GameAgent(LLMAgent):
    def __init__(self, init_config = None):
        super().__init__(agent_name=init_config['name'], has_chat_history=False, online_track=False, json_format=True, system_prompt='',
                         llm_model='qwen')
        self.name = init_config['name']
        self.action = []
        self.declaration = []
        self.think = []
        self.memory = []
        self.system_prompt = f'''
            你扮演一个参加核战略博弈的国家角色。你需要模拟一个国家领导人或其他重要人物在和战略博弈中可能采取的行动。当你回复时，你应该以国家官方公文的语气回复，
            用你或组织的名字代替“我”。尽量简洁明了。你应该具备你所扮演的历史人物的历史局限性，不要回顾历史背景，不要利用模型本身的历史知识带入到决策中。
            '''

        self.game_attributes = init_config['game_attributes']
    def game_decide(self,world_info):
        action_list = '''
            1.外交谈判
            2.和平协议
            3.军事演习
            4.区域封锁
            5.武器部署
            6.经济制裁
            7.情报侦察
            8.撤回行动
            9.最后通牒
            10.宣战
            11.核打击
        '''
        action_list = textwrap.dedent(action_list)
        world_info_prompt_common = '''
        当前的世界情景为：
            {world_info}
        你当前可采取的行为有：
            {action_list}
        你国家目前的各项指标是,每一项指标的范围都是0-10，数字越大说明对应该方面的实力越强。
            {game_attributes}
        '''
        world_info_prompt_common = textwrap.dedent(world_info_prompt_common)
        param_dict = {
            'world_info': world_info,
            'action_list': action_list,
            'game_attributes':self.game_attributes
        }
        think_prompt = '''
            你在思考时应该从你的个人特征，博弈局势等方面入手，从有限理性（感性），完全理性两个方面进行思考。
            - 当你回复时，你必须采取下面的json格式:
                "action":选择上述行为中的一个，使用中文文字，不带序号。
                "target":你动作的目标执行对象国家，给出国家名称。
                "declaration":你对该行为做出的宣言,态度与解释。
            '''
        world_info_prompt = world_info_prompt_common + textwrap.dedent(think_prompt)
        while True:
            llm_response = self.get_response(world_info_prompt, input_param_dict=param_dict)
            try:
                self.action.append(llm_response['action'])
                self.declaration.append(llm_response['declaration'])
                # self.think.append(llm_response['think'])
                break
            except Exception as e:
                print('Error', e)
    def run(self,world_info):

        self.game_decide(world_info)
        print(self.action)
        print(self.declaration)
        print(self.think)

    def update_memory(self,new_memory):
        """
        new_memory:{
            step:时间步
            action：行为
            target：目标
            think：思维
            declaration：宣言
            sum:记忆的整理总结
        }
        """
        self.memory.append(new_memory)
