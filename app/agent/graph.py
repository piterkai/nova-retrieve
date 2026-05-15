from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from app.agent import nodes
from app.agent.state import GraphState


def build_graph():
    """Build the Agentic RAG state graph.

    Flow:
        START
          -> rewrite_query
          -> route_question
              -> retrieve (vectorstore)              -> grade_documents
                                                          -> generate (if relevant)
                                                          -> transform_query -> retrieve
                                                          -> web_search (after retries)
              -> web_search                          -> generate
          -> grade_hallucination
              -> generate (regenerate if hallucinated, bounded)
              -> grade_answer
                  -> END (if useful)
                  -> transform_query -> retrieve (bounded)
    """
    g = StateGraph(GraphState)

    g.add_node("rewrite_query", nodes.rewrite_query)
    g.add_node("route_question", nodes.route_question)
    g.add_node("retrieve", nodes.retrieve_docs)
    g.add_node("grade_documents", nodes.grade_documents)
    g.add_node("transform_query", nodes.transform_query)
    g.add_node("web_search", nodes.do_web_search)
    g.add_node("generate", nodes.generate)
    g.add_node("hallucination_grader", nodes.grade_hallucination)
    g.add_node("answer_grader", nodes.grade_answer)

    g.add_edge(START, "rewrite_query")
    g.add_edge("rewrite_query", "route_question")

    g.add_conditional_edges(
        "route_question",
        nodes.edge_after_route,
        {"retrieve": "retrieve", "web_search": "web_search"},
    )

    g.add_edge("retrieve", "grade_documents")
    g.add_conditional_edges(
        "grade_documents",
        nodes.edge_after_grade_docs,
        {
            "generate": "generate",
            "transform_query": "transform_query",
            "web_search": "web_search",
        },
    )
    g.add_edge("transform_query", "retrieve")
    g.add_edge("web_search", "generate")
    g.add_edge("generate", "hallucination_grader")

    g.add_conditional_edges(
        "hallucination_grader",
        nodes.edge_after_hallucination,
        {"generate": "generate", "answer_grader": "answer_grader"},
    )
    g.add_conditional_edges(
        "answer_grader",
        nodes.edge_after_answer_grader,
        {"END": END, "transform_query": "transform_query"},
    )

    return g.compile()


@lru_cache(maxsize=1)
def get_graph():
    return build_graph()
