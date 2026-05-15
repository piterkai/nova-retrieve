"""Interactive REPL against the Agentic RAG graph.

Usage:
    python -m scripts.chat_cli
"""
import time

from app.agent.graph import get_graph
from app.core.logging import setup_logging


def main():
    setup_logging()
    graph = get_graph()
    print("Nova Retrieve CLI — type your question (Ctrl-C to quit)\n")
    while True:
        try:
            q = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        t0 = time.perf_counter()
        state = graph.invoke({"question": q})
        total_ms = (time.perf_counter() - t0) * 1000.0
        print("\n--- Answer ---")
        print(state.get("generation", "(no answer)"))
        cites = state.get("citations") or []
        if cites:
            print("\n--- Citations ---")
            for c in cites:
                print(f"  [{c['index']}] {c.get('title') or ''} {c.get('source') or ''}")
        timings = state.get("timings") or []
        if timings:
            print("\n--- Timings ---")
            for t in sorted(timings, key=lambda x: x.get("seq", 0)):
                print(f"  {t['step']:<22s} {t['elapsed_ms']:>8.1f} ms")
            print(f"  {'TOTAL':<22s} {total_ms:>8.1f} ms")
        print()


if __name__ == "__main__":
    main()
