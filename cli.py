"""Команды ask и search для локального демо."""

import argparse

from rag import SEARCH_MODES, Search, answer, load_clauses, plan


def source(c):
    interval = f"с {c.effective_from:%d.%m.%Y}"
    if c.effective_until:
        interval += f" до {c.effective_until:%d.%m.%Y} (исключая эту дату)"
    return (f"{c.title}, п. {c.number} (data/documents/{c.file}:{c.start_line}–{c.end_line})"
            f"\nРедакция {interval}.")


def main():
    parser = argparse.ArgumentParser(description="Поиск по договорам и регламентам на дату события")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("ask", "search"):
        subparser = commands.add_parser(command)
        subparser.add_argument("question")
        subparser.add_argument("--date", help="дата события, например 01.07.2025")
    commands.choices["search"].add_argument("--mode", choices=SEARCH_MODES, default="versioned")
    args = parser.parse_args()
    index = Search(load_clauses())

    if args.command == "ask":
        result = answer(args.question, index, as_of=args.date)
        print(result.summary)
        for number, clause in enumerate(result.sources, start=1):
            print(f"\n[{number}] {clause.topic}\n> {clause.text}\n{source(clause)}")
        return

    try:
        requests = plan(args.question, index, as_of=args.date)
    except ValueError as exc:
        if args.mode != "bm25":
            parser.error(str(exc))
        requests = [(args.question, None)]
    if not requests:
        parser.error("Уточните, какое условие документа нужно найти.")
    for part, scope in requests:
        if len(requests) > 1:
            print(f"\nЧасть запроса: {part}")
        hits = index.search(part, scope=scope, mode=args.mode)
        if not hits:
            print("Подходящих фрагментов нет.")
        for clause, score in hits:
            print(f"\n{score:6.3f}  {clause.id} · {source(clause)}\n{clause.text}")


if __name__ == "__main__":
    main()
