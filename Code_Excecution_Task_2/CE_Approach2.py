import chromadb
from typing_extensions import Annotated

import autogen
from autogen import AssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent


config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST.json",
    filter_dict={
        "model": ["gpt-4-turbo"],
    },
)

#config_list = autogen.config_list_from_json("OAI_CONFIG_LIST.json")

#print("LLM models: ", [config_list[i]["gemini-pro"] for i in range(len(config_list))])


def termination_msg(x):
    return isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()


llm_config = {"config_list": config_list, "timeout": 180, "temperature": 0.1, "seed": 1234}

boss = autogen.UserProxyAgent(
    name="Boss",
    is_termination_msg=termination_msg,
    human_input_mode="NEVER",
    code_execution_config=False,  # we don't want to execute code in this case.
    default_auto_reply="Reply `TERMINATE` if the task is done.",
    description="The boss who ask questions and give tasks.",
)

boss_aid = RetrieveUserProxyAgent(
    name="Boss_Assistant",
    is_termination_msg=termination_msg,
    human_input_mode="NEVER",
    default_auto_reply="Reply `TERMINATE` if the task is done.",
    max_consecutive_auto_reply=8,
    retrieve_config={
        "task": "code",
        "docs_path": "/Users/asswincr/Desktop/NTU_AI/GenAI_Sembly/Gemini_CrewAI/Autogen_state_flow/graph2vec.pdf",
        "chunk_token_size": 12000,
        "model": config_list[0]["model"],
        "collection_name": "groupchat",
        "get_or_create": True,
    },
    code_execution_config=False,  # we don't want to execute code in this case.
    description="Assistant who has extra content retrieval power for solving difficult problems.",
)

coder = AssistantAgent(
    name="Senior_Python_Engineer",
    is_termination_msg=termination_msg,
    system_message="You are a senior python engineer, you provide python code to answer questions. Reply `TERMINATE` in the end when everything is done.",
    llm_config=llm_config,
    description="Senior Python Engineer who can write code to solve problems and answer questions.",
)


pm = autogen.UserProxyAgent(
    name="executor",
    system_message="Executor. Execute the code written by the coder and report the result.",
    human_input_mode="NEVER",
    code_execution_config={#"executor": executor
                           "last_n_messages": 3,
                           "work_dir": "paper",
                           "use_docker": False,
                           },
)

reviewer = autogen.AssistantAgent(
    name="Code_Reviewer",
    is_termination_msg=termination_msg,
    system_message="You are a code reviewer. Reply `TERMINATE` in the end when everything is done.",
    llm_config=llm_config,
    description="Code Reviewer who can review the code.",
)

PROBLEM = 'Write a python script to implement this paper titiled "graph2vec: Learning Distributed Representations of Graphs" available at the link https://arxiv.org/abs/1707.05005, get the core algorithm of the paper, formulalte how to implement it step by step and run and execute the code with a dummy dataset and print the final evaluation metrics such as precision, recall and F1 score.'


def _reset_agents():
    boss.reset()
    boss_aid.reset()
    coder.reset()
    pm.reset()
    reviewer.reset()


def rag_chat():
    _reset_agents()
    groupchat = autogen.GroupChat(
        agents=[boss_aid, pm, coder, reviewer], messages=[], max_round=12, speaker_selection_method="round_robin"
    )
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    # Start chatting with boss_aid as this is the user proxy agent.
    boss_aid.initiate_chat(
        manager,
        message=boss_aid.message_generator,
        problem=PROBLEM,
        n_results=2,
    )


def norag_chat():
    _reset_agents()
    groupchat = autogen.GroupChat(
        agents=[boss, pm, coder, reviewer],
        messages=[],
        max_round=12,
        speaker_selection_method="auto",
        allow_repeat_speaker=False,
    )
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    # Start chatting with the boss as this is the user proxy agent.
    boss.initiate_chat(
        manager,
        message=PROBLEM,
    )


def call_rag_chat():
    _reset_agents()

    # In this case, we will have multiple user proxy agents and we don't initiate the chat
    # with RAG user proxy agent.
    # In order to use RAG user proxy agent, we need to wrap RAG agents in a function and call
    # it from other agents.
    def retrieve_content(
        message: Annotated[
            str,
            "Refined message which keeps the original meaning and can be used to retrieve content for code generation and question answering.",
        ],
        n_results: Annotated[int, "number of results"] = 3,
    ) -> str:
        boss_aid.n_results = n_results  # Set the number of results to be retrieved.
        # Check if we need to update the context.
        update_context_case1, update_context_case2 = boss_aid._check_update_context(message)
        if (update_context_case1 or update_context_case2) and boss_aid.update_context:
            boss_aid.problem = message if not hasattr(boss_aid, "problem") else boss_aid.problem
            _, ret_msg = boss_aid._generate_retrieve_user_reply(message)
        else:
            _context = {"problem": message, "n_results": n_results}
            ret_msg = boss_aid.message_generator(boss_aid, None, _context)
        return ret_msg if ret_msg else message

    boss_aid.human_input_mode = "NEVER"  # Disable human input for boss_aid since it only retrieves content.

    for caller in [pm, coder, reviewer]:
        d_retrieve_content = caller.register_for_llm(
            description="retrieve content for code generation and question answering.", api_style="function"
        )(retrieve_content)

    for executor in [boss, pm]:
        executor.register_for_execution()(d_retrieve_content)

    groupchat = autogen.GroupChat(
        agents=[boss, pm, coder, reviewer],
        messages=[],
        max_round=12,
        speaker_selection_method="round_robin",
        allow_repeat_speaker=False,
    )

    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    # Start chatting with the boss as this is the user proxy agent.
    boss.initiate_chat(
        manager,
        message=PROBLEM,
    )
    
rag_chat()
