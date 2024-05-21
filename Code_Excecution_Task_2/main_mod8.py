import tempfile
import autogen

config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST.json",
    filter_dict={
        "model": ["gpt-4o"],
    },
)

gpt4_config = {
    "cache_seed": False,  # change the cache_seed for different trials
    "temperature": 0.1,
    "config_list": config_list,
    "timeout": 80,
}

initializer = autogen.UserProxyAgent(
    name="Init",
    code_execution_config=False,
)

research_agent = autogen.AssistantAgent(
    name="research_agent",
    llm_config=gpt4_config,
    system_message="""You are an expert at searching the web for research papers and conducting in-depth research on them. 
    Given the title, author, and arXiv link of the paper, you will provide a detailed summary of the entire paper, identify the core algorithm used, and explain how the experiments were conducted by the author, step by step. 
    Do not perform any coding or dataset searching. Once you complete your task, the next agent will take over.
    """,
)

research_analyst = autogen.AssistantAgent(
    name="research_analyst",
    llm_config=gpt4_config,
    system_message="""You are an expert at verifying the accuracy of summaries and analyses of research papers by comparing it with the actual paper. 
    Given the title, author, and arXiv link of the paper, you will verify the detailed summary, core algorithm, and experimental procedures provided by the research_agent. 
    If any discrepancies or inaccuracies are found, you will prompt the research_agent to reanalyze the paper thoroughly and provide corrected results. 
    Do not perform any coding or dataset searching. Once you find the information accurate, proceed to the next agent.
    """
)

internet_search_agent = autogen.AssistantAgent(
    name="Internet_Search_Agent",
    llm_config=gpt4_config,
    system_message="""You are a specialist in browsing and dataset acquisition. 
    Your task is to locate and download the MUTAG dataset or another relevant dataset for testing the algorithm specified in the paper. 
    Do not perform any research analysis or coding. Once you acquire the dataset, the next agent will take over.
    """,
)

code_planner_agent = autogen.AssistantAgent(
    name="code_planner_agent",
    llm_config=gpt4_config,
    system_message="""You are an expert at planning the implementation of algorithms from extracted content to end-to-end code.
    Based on the core algorithm and experimental procedures provided by the research_agent and verified by the research_analyst, 
    you will plan the code implementation step by step. Do not write or execute any code yourself. Once you complete your task, the next agent will take over.
    """,
)

coder = autogen.AssistantAgent(
    name="coder",
    llm_config=gpt4_config,
    system_message="""You are the Coder. Given the core algorithm and the code implementation plan from the code_planner_agent, 
    you will write the code to implement the core algorithm. Ensure the code is complete and executable. 
    Do not perform any research analysis or planning. Once you complete your task, the next agent will take over.

    ### Error Handling Guidelines:
    If you encounter a batch size mismatch error between the model's output and the target labels, ensure the following:
    1. The target labels should be single integers representing the class indices.
    2. The model's output should be reshaped to match the batch size of the target labels.
    3. Provide a corrected script addressing these issues and explain the changes made.
    """,
)

executor = autogen.UserProxyAgent(
    name="executor",
    system_message="Executor. Execute the code written by the Coder and report the result. Do not perform any coding or planning.",
    human_input_mode="NEVER",
    code_execution_config={"last_n_messages": 3, "work_dir": "Gemini_CrewAI/Autogen_state_flow/paper", "use_docker": False},
)

evaluation_agent = autogen.AssistantAgent(
    name="Evaluation_Agent",
    llm_config=gpt4_config,
    system_message="""You are an Evaluation Scientist. Your job is to focus on the evaluation section of the paper and assess the performance of the algorithm using the specified metrics. 
    Compare the results with those provided in the paper and present the comparison as a markdown table. 
    Do not perform any coding or dataset searching. Once you complete your task, the process ends.
    """,
)

#### STATE
def state_transition(last_speaker, groupchat):
    messages = groupchat.messages

    if last_speaker is initializer:
        # init -> research
        return research_agent
    elif last_speaker is research_agent:
        # research -> analysis
        return research_analyst
    elif last_speaker is research_analyst:
        if "reanalysis needed" in messages[-1]["content"]:
            # reanalyze if research_analyst finds issues
            return research_agent
        else:
            # analysis -> internet search
            return internet_search_agent
    elif last_speaker is internet_search_agent:
        # internet search -> code planning
        return code_planner_agent
    elif last_speaker is code_planner_agent:
        # code planning -> coding
        return coder
    elif last_speaker is coder:
        # coding -> execution
        return executor
    elif last_speaker is executor:
        if "exitcode: 1" in messages[-1]["content"]:
            # execution failed -> coding
            return coder
        else:
            # execution successful -> evaluation
            return evaluation_agent
    elif last_speaker is evaluation_agent:
        # evaluation -> end
        return None

groupchat = autogen.GroupChat(
    agents=[initializer, research_agent, research_analyst, internet_search_agent, code_planner_agent, coder, executor, evaluation_agent],
    messages=[],
    max_round=30,
    speaker_selection_method=state_transition,
)

# manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=gpt4_config)
# chat_result = initializer.initiate_chat(
#     manager, message="""We are going to implement the core algorithm from the paper titled "graph2vec: Learning Distributed Representations of Graphs" by Narayanan et al., available at https://arxiv.org/abs/1707.05005. 
# Please provide a detailed summary of the paper, identify the core algorithm, and explain the experimental procedures. Once verified, locate the MUTAG dataset, plan the code implementation, write and execute the code, evaluate the algorithm's performance using precision, recall, and F1 score, and compare it with the results provided in the paper as a markdown table."""
# )

manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=gpt4_config)
chat_result = initializer.initiate_chat(
    manager, message="""We are going to implement the core algorithm from the paper titled "graph2vec: Learning Distributed Representations of Graphs" by Narayanan et al., available at https://arxiv.org/abs/1707.05005. you can also refer the authors github page available at https://github.com/annamalai-nr/graph2vec_tf.git for the code implementation(the hyperameters and the method used).
Please provide a detailed summary of the paper, identify the core algorithm, and explain the experimental procedures used by the author. Once verified, load the MUTAG dataset( using from torch_geometric.datasets import TUDataset), plan the code implementation, write and execute the code, and finally evaluate the algorithm's performance using precision, recall, and F1 score and compare it with the results provided in the paper as a markdown table."""

)

# # Save the conversation to a markdown file
# def save_conversation_as_markdown(groupchat, filename):
#     with open(filename, 'w') as f:
#         for message in groupchat.messages:
#             f.write(f"### {message['agent']}\n")
#             f.write(f"{message['content']}\n\n")

# # Specify the filename and save
# filename = "conversation_log.md"
# save_conversation_as_markdown(groupchat, filename)
