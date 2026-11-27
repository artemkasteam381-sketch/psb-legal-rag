"""Локальная оценка ранжирования, ответов и ссылок без ключа."""

import argparse
import json
from statistics import mean

from rag import DATA_DIR, TOP_K, Search, answer, load_clauses, plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--details", action="store_true", help="показать попадания по каждому вопросу")
    args = parser.parse_args()
    cases = json.loads((DATA_DIR / "evaluation.json").read_text(encoding="utf-8"))
    index = Search(load_clauses())
    clause_ids = {clause.id for clause in index.clauses}
    retrieval_cases = [case for case in cases if case["sources"]]
    if any(set(case["sources"]) - clause_ids for case in retrieval_cases):
        raise ValueError("В разметке указан отсутствующий фрагмент.")
    print(f"Корпус: {len(index.documents)} документов, "
          f"{len({clause.file for clause in index.clauses})} файлов, {len(clause_ids)} фрагментов.")
    print(f"Вопросов с размеченными источниками: {len(retrieval_cases)} из {len(cases)}.\n")
    missing_cases = [case for case in cases if case["status"] == "insufficient"]
    print(f"{'Поиск':<29} {'Recall@'+str(TOP_K):>9} {'Топ-1 всех частей':>17} "
          f"{'Точные ссылки':>14} {'Лишние ответы':>15}")
    for mode, label in (("bm25", "BM25"), ("document", "BM25 + документ"),
                        ("versioned", "BM25 + документ и дата")):
        recalls, first_hits, exact = [], [], []
        for case in retrieval_cases:
            question = case["question"]
            as_of = case.get("as_of")
            expected = set(case["sources"])
            lists = [
                [clause.id for clause, _ in index.search(part, scope=scope, mode=mode)]
                for part, scope in plan(question, index, as_of=as_of)
            ]
            retrieved = {source for hits in lists for source in hits}
            top = {hits[0] for hits in lists if hits}
            recalls.append(len(expected & retrieved) / len(expected))
            first_hits.append(top == expected)
            result = answer(question, index, as_of=as_of, mode=mode)
            cited = {clause.id for clause in result.sources}
            exact.append(cited == expected)
            if args.details and (cited != expected or not first_hits[-1]):
                print(f"  {label} · {case['id']}: топ-1 {sorted(top)}; "
                      f"ссылки {sorted(cited)}, ожидалось {sorted(expected)}")
        unsupported = sum(
            bool(answer(case["question"], index, as_of=case.get("as_of"), mode=mode).sources)
            for case in missing_cases
        )
        extra = f"{unsupported}/{len(missing_cases)}"
        print(f"{label:<29} {mean(recalls):>9.1%} {mean(first_hits):>17.1%} "
              f"{mean(exact):>14.1%} {extra:>15}")

    failures = []
    for case in cases:
        result = answer(case["question"], index, as_of=case.get("as_of"))
        citations = {clause.id for clause in result.sources}
        response = "\n".join([result.summary, *(clause.text for clause in result.sources)])
        quotes_match = True
        for clause in result.sources:
            lines = (DATA_DIR / "documents" / clause.file).read_text(encoding="utf-8").splitlines()
            heading = lines[clause.start_line - 1]
            body = "\n".join(lines[clause.start_line:clause.end_line]).strip()
            quotes_match &= heading == f"## {clause.number} | {clause.topic}" and body == clause.text
        if (result.status != case["status"] or citations != set(case["sources"])
                or not quotes_match or any(part not in response for part in case.get("contains", []))):
            failures.append(case["id"])
            print(f"  Ошибка {case['id']}: {result.status}, {sorted(citations)}, {response}")
    print(f"\nСтатус, источники, цитаты и проверяемые фразы: {len(cases) - len(failures)}/{len(cases)}.")
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
