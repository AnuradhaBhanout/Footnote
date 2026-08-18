
from langgraph.graph import StateGraph, END


from graph.nodes import GraphNodes
from graph.routing import after_cache,after_citation_check, after_run_agent
from graph.state import GraphState




def build_graph(llm, chatbot):

    nodes = GraphNodes(llm,chatbot)    

    graph = StateGraph(GraphState)  
    graph.add_node("check_cache",nodes.check_cache)
    graph.add_node("run_agent",nodes.run_agent)
    graph.add_node("clarify", nodes.clarify)
    graph.add_node("check_citations",nodes.check_citations)
    graph.add_node("retry_with_feedback",nodes.retry_with_feedback)
    graph.add_node("fallback",nodes.fallback)
    graph.add_node("finalize",nodes.finalize)

    graph.set_entry_point("check_cache")
    graph.add_conditional_edges("check_cache",after_cache,{
        "end":END,
        "triage_query": "run_agent"
    })



    #optimization
    graph.add_conditional_edges("run_agent",after_run_agent,{
        "clarify":"clarify",
        "retry":"run_agent",
        "ok":"check_citations",
    })

    graph.add_edge("clarify", "run_agent")


    
    graph.add_conditional_edges(
        "check_citations", after_citation_check,{
            "finalize":"finalize",
            "end_no_cache":END,
            "retry_with_feedback":"retry_with_feedback",
            "fallback":"fallback",
        }
    )


    graph.add_edge("retry_with_feedback","run_agent")
    graph.add_edge("finalize",END)
    graph.add_edge("fallback",END)

    return graph












    