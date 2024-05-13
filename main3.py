import tempfile
import autogen


config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST.json",
    filter_dict={
        "model": ["gpt-4-turbo"],
    },
)

gpt4_config = {
    "cache_seed": False,  # change the cache_seed for different trials
    "temperature": 0.2,
    "config_list": config_list,
    "timeout": 120,
}

initializer = autogen.UserProxyAgent(
    name="Init",
    code_execution_config=False,
)

research_agent = autogen.AssistantAgent(
    name="research_agent",
    llm_config=gpt4_config,
    system_message="""You are an expert at searching web for research papers and doing in-depth research on it(you can use advanced techniques to do this). 
    Given the title, author and arXiv link of the paper, You will provide a detailed summary of the paper and find the core algorithm used in the paper by the author and explain it step by step.
""",
)
# Internet_Search_Agent=autogen.AssistantAgent(
#     name="Internet_Search_Agent",
#     llm_config=gpt4_config,
#     system_message="""You are Specialist is browsing.Please locate and download the  dataset or another relevant dataset for testing the algorithm as specified in the paper.""",
# )
code_planner_agent = autogen.AssistantAgent(
    name="code_planner_agent",
    llm_config=gpt4_config,
    system_message=""" You are an expert at planning from the extracted content to end to end code step by step.
""",
)


coder = autogen.AssistantAgent(
    name="coder",
    llm_config=gpt4_config,
    system_message="""You are the Coder. Given a core algorithm of a research paper, write code to implement the core algorithm of it.
You write python/shell code to solve tasks. Wrap the code in a code block that specifies the script type. The user can't modify your code. So do not suggest incomplete code which requires others to modify. Don't use a code block if it's not intended to be executed by the executor.
Don't include multiple code blocks in one response. Do not ask others to copy and paste the result. Check the execution result returned by the executor.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
""",
)

# coder_checker = autogen.AssistantAgent(
#     name="coder_checker",
#     llm_config=gpt4_config,
#     system_message="""You are expert at checking if the code is fully complete without any placeholders.
# """,
# )
executor = autogen.UserProxyAgent(
    name="executor",
    system_message="Executor. Execute the code written by the Coder and report the result.",
    human_input_mode="NEVER",
    code_execution_config={#"executor": executor
                           "last_n_messages": 3,
                           "work_dir": "paper",
                           "use_docker": False,
                           },
)


Evaluation_Agent = autogen.AssistantAgent(
    name="Evaluation_Agent",
    llm_config=gpt4_config,
    system_message="""You are Evalautation Scientist. your job is to focus on a particular section of the paper especially the evaluation section and evaluate the performance of the algorithm using metrics given in the paper.""",
)
#### STATE 
def state_transition(last_speaker, groupchat):
    messages = groupchat.messages

    if last_speaker is initializer:
        # init -> retrieve
        return research_agent
    elif last_speaker is research_agent:
        # retrieve: action 1 -> action 2
        return code_planner_agent
    elif last_speaker is code_planner_agent:
        # retrieve: action 1 -> action 2
        return coder
    elif last_speaker is coder:
        # retrieve: action 1 -> action 2
        return executor
    elif last_speaker is executor:
        if "exitcode: 1" in messages[-1]["content"]:
            # retrieve --(execution failed)--> retrieve
            return coder
        else:
            
            return Evaluation_Agent
        
    elif last_speaker is Evaluation_Agent:
        # research -> end
        return None


groupchat = autogen.GroupChat(
    agents=[initializer, research_agent,code_planner_agent,coder, executor,Evaluation_Agent],
    messages=[],
    max_round=30,
    speaker_selection_method=state_transition,
)

manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=gpt4_config)
chat_result = initializer.initiate_chat(
    manager, message='Write a python script to implement this paper titiled "graph2vec: Learning Distributed Representations of Graphs" available at the link https://arxiv.org/abs/1707.05005, get the core algorithm of the paper, formulalte how to implement it step by step and run and execute the code with a dummy dataset and print the final evaluation metrics such as precision, recall and F1 score.'
)

