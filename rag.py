"""Разбиение по пунктам, BM25 и выбор редакции по дате."""

import json
import re
from collections import Counter
from dataclasses import dataclass, replace
from datetime import date, datetime
from math import log1p
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent / "data"
TOP_K = 5
SEARCH_MODES = ("bm25", "document", "versioned", "contract")
SECTION = re.compile(r"^##\s+(\d+(?:\.\d+)*)\s*\|\s*(.+)$")
DOCUMENT_ID = re.compile(r"\b[а-яa-z]{1,4}-\d{2}-\d{4}\b")
DOCUMENT_KIND = re.compile(r"\b(договор|регламент)\w*")
DOCUMENT_REFERENCE = re.compile(r"\b(?:по|в|из|согласно)\s+(договор|регламент)\w*")
DATE = re.compile(r"\b(?:\d{2}\.\d{2}\.\d{4}|\d{4}-\d{2}-\d{2})\b")
TOKEN = re.compile(r"[а-яa-z0-9]+(?:-[а-яa-z0-9]+)*")
TRANSLATION = str.maketrans("ё‑–—−", "е----")
ROOTS = {
    "оплат": re.compile(r"(?:оплат|оплач|платеж|платит|платить|расчет|перечисл)\w*"),
    "прием": re.compile(r"(?:приемк|принима|принят)\w*"),
    "соглас": re.compile(r"(?:согласован|согласов|согласу)\w*"),
    "расхожд": re.compile(r"расхожд\w*"),
    "расторг": re.compile(r"(?:расторг|расторж)\w*"),
    "провер": re.compile(r"провер\w*"),
    "хран": re.compile(r"хран\w*"),
}
ENDING = re.compile(
    r"(?<=[а-я]{4})(?:иями|ями|ами|ию|иям|иях|ией|ием|ого|его|ому|ему|"
    r"ыми|ими|ать|ять|ить|ают|яют|уют|ят|ут|ой|ый|ая|ые|ий|ых|их|ую|юю|"
    r"ее|ое|ам|ям|ах|ях|ом|ем|ов|ев|ей|ия|ие|ии|[ауеыиюяь])$"
)
STOP_WORDS = frozenset(
    "в для до за и из к на о от по с со во у при что как какой каких какие какого какая "
    "какую каком каков каковы сколько когда кто где чем чего чья есть нужно нужны "
    "можно надо ли верно или если а также такой таком дату день дней дня года лет "
    "срок срока сроки время долго быстро проходит организована исчисляется считается "
    "событие события сравни товар "
    "товара партию партии покупатель поставщик деньги календарных рабочих должны должен "
    "участник условия договор договора договору регламент регламенту регламента".split()
)


@dataclass(frozen=True)
class Clause:
    file: str
    title: str
    number: str
    slot: str
    topic: str
    text: str
    document_id: str
    kind: str
    effective_from: date
    start_line: int
    end_line: int
    effective_until: date | None = None

    @property
    def id(self):
        return f"{Path(self.file).stem}#{self.number}"


@dataclass(frozen=True)
class Scope:
    document_id: str
    as_of: date


@dataclass(frozen=True)
class Answer:
    status: str
    summary: str
    sources: tuple[Clause, ...] = ()


def normalize(text):
    return text.lower().translate(TRANSLATION)


def tokenize(text):
    for token in TOKEN.findall(normalize(text)):
        if token not in STOP_WORDS:
            yield next((root for root, pattern in ROOTS.items() if pattern.fullmatch(token)),
                       ENDING.sub("", token))


def evidence_terms(text):
    text = DOCUMENT_ID.sub(" ", DATE.sub(" ", normalize(text)))
    return {term for term in tokenize(text) if term.isalpha()}


def date_from(question, *, as_of=None):
    dates = set()
    values = DATE.findall(normalize(question))
    if as_of is not None:
        as_of = normalize(as_of.strip())
        if not DATE.fullmatch(as_of):
            raise ValueError("Укажите дату в формате ДД.ММ.ГГГГ или ГГГГ-ММ-ДД.")
        values.append(as_of)
    for value in values:
        pattern = "%d.%m.%Y" if "." in value else "%Y-%m-%d"
        try:
            dates.add(datetime.strptime(value, pattern).date())
        except ValueError as exc:
            raise ValueError(f"Дата {value} не существует. Укажите дату события.") from exc
    if len(dates) != 1:
        raise ValueError("Укажите одну дату события, например 01.07.2025.")
    return dates.pop()


def plan(question, index, *, as_of=None):
    """Привязать части вопроса к документам, используя один день события."""
    normalized = normalize(question)
    explicit = {value.upper() for value in DOCUMENT_ID.findall(normalized)}
    kinds = DOCUMENT_KIND.findall(normalized)
    by_kind = {kind: {value for value in explicit if index.kind(value) == kind}
               for kind in ("договор", "регламент", "документ")}
    if any(len(values) > 1 for values in by_kind.values()):
        raise ValueError("Укажите один номер для каждого типа документа.")
    if not explicit and not kinds:
        raise ValueError("Укажите один номер договора или регламента, например ДП-17-2025.")

    instant = date_from(question, as_of=as_of)
    requests = []
    for part in re.split(r"\b(?:и|а также)\b|[?;]", normalized):
        if not evidence_terms(part):
            continue
        local = {value.upper() for value in DOCUMENT_ID.findall(part)}
        kind = DOCUMENT_REFERENCE.search(part) or DOCUMENT_KIND.search(part)
        if local:
            candidates = local
        elif kind:
            candidates = by_kind[kind[1]] or {
                value for value, value_kind in index.documents.items() if value_kind == kind[1]
            }
        else:
            candidates = explicit
            if not candidates and len(set(kinds)) == 1:
                candidates = {value for value, value_kind in index.documents.items()
                              if value_kind == kinds[0]}
        if len(candidates) != 1:
            raise ValueError("Укажите один номер документа для каждой части вопроса.")
        document_id = next(iter(candidates))
        if kind and index.kind(document_id) not in (kind[1], "документ"):
            raise ValueError(f"Уточните, к какому документу относится вопрос о {document_id}.")
        requests.append((part.strip(), Scope(document_id, instant)))
    return requests


def load_clauses(data_dir=DATA_DIR):
    catalog = json.loads((data_dir / "catalog.json").read_text(encoding="utf-8"))
    clauses = []
    base_slots = set()
    seen_versions = set()
    for document in sorted(catalog, key=lambda item: item["effective_from"]):
        file = document["file"]
        document_id = document["document_id"]
        kind = document["kind"]
        effective_from = date.fromisoformat(document["effective_from"])
        replaces = document.get("replaces")
        lines = (data_dir / "documents" / file).read_text(encoding="utf-8").splitlines()
        if not lines or not lines[0].startswith("# "):
            raise ValueError(f"Ожидается заголовок документа: {file}")
        title = lines[0][2:]
        headings = [
            (index, match[1], match[2])
            for index, line in enumerate(lines)
            if (match := SECTION.fullmatch(line))
        ]
        numbers = {number for _, number, _ in headings}
        if not headings or len(numbers) != len(headings):
            raise ValueError(f"Проверьте номера пунктов в {file}")
        if replaces is not None and replaces.keys() != numbers:
            raise ValueError(f"Для каждого пункта {file} требуется указать заменяемый пункт")
        for position, (start, number, topic) in enumerate(headings):
            end = headings[position + 1][0] if position + 1 < len(headings) else len(lines)
            while end > start + 1 and not lines[end - 1].strip():
                end -= 1
            body = "\n".join(lines[start + 1:end]).strip()
            if not body:
                raise ValueError(f"Пустой пункт {number} в {file}")

            slot = number if replaces is None else replaces[number]
            key = (document_id, slot)
            if replaces is None:
                if key in base_slots:
                    raise ValueError(f"Повтор базового пункта {key}")
                base_slots.add(key)
            elif key not in base_slots:
                raise ValueError(f"{file}, п. {number}: отсутствует базовый пункт {slot}")
            version_key = (*key, effective_from)
            if version_key in seen_versions:
                raise ValueError(f"Две версии пункта {key} на дату {effective_from}")
            seen_versions.add(version_key)
            clauses.append(Clause(file, title, number, slot, topic, body, document_id, kind,
                                  effective_from, start + 1, end))
    # Следующая замена закрывает предыдущую редакцию того же пункта.
    next_version = {}
    for index in reversed(range(len(clauses))):
        clause = clauses[index]
        key = (clause.document_id, clause.slot)
        clauses[index] = replace(clause, effective_until=next_version.get(key))
        next_version[key] = clause.effective_from
    return clauses


class Search:
    def __init__(self, clauses):
        if not clauses:
            raise ValueError("Корпус не содержит пунктов для поиска.")
        self.clauses = clauses
        self.documents = {clause.document_id: clause.kind for clause in clauses}
        self.headings = {clause.id: evidence_terms(clause.topic) for clause in clauses}
        self.evidence = {clause.id: evidence_terms(f"{clause.topic} {clause.text}")
                         for clause in clauses}
        self.terms = [
            Counter(tokenize(f"{c.title} {c.topic} {c.text}")) for c in clauses
        ]
        lengths = [counts.total() for counts in self.terms]
        average = sum(lengths) / len(lengths)
        # Слагаемое в знаменателе BM25 при k1=1.5, b=0.75.
        self.length_penalties = [1.5 * (0.25 + 0.75 * length / average) for length in lengths]
        frequency = Counter(term for counts in self.terms for term in counts)
        size = len(self.terms)
        self.idf = {
            term: log1p((size - count + 0.5) / (count + 0.5))
            for term, count in frequency.items()
        }

    def active_clauses(self, scope):
        return {
            clause.id
            for clause in self.clauses
            if clause.document_id == scope.document_id
            and clause.effective_from <= scope.as_of
            and (clause.effective_until is None or scope.as_of < clause.effective_until)
        }

    def kind(self, document_id):
        by_prefix = {"ДП": "договор", "РГ": "регламент"}
        return self.documents.get(document_id, by_prefix.get(document_id.split("-")[0], "документ"))

    def search(self, question, *, scope=None, mode=None, limit=TOP_K):
        if mode is None:
            mode = "versioned" if scope else "bm25"
        if mode == "contract":  # Имя режима в предыдущей версии CLI.
            mode = "document"
        if mode not in SEARCH_MODES or (mode != "bm25" and scope is None):
            raise ValueError("Фильтруемый поиск требует известного режима, документа и даты.")
        allowed = self.active_clauses(scope) if mode == "versioned" else None
        query = set(tokenize(DATE.sub(" ", normalize(question))))
        if scope:
            query.add(scope.document_id.lower())
        query = sorted(query & self.idf.keys())
        results = []
        for clause, counts, penalty in zip(self.clauses, self.terms, self.length_penalties):
            if mode == "document" and clause.document_id != scope.document_id:
                continue
            if allowed is not None and clause.id not in allowed:
                continue
            score = sum(self.idf[t] * 2.5 * counts[t] / (counts[t] + penalty) for t in query)
            if score > 0:
                results.append((clause, score))
        return sorted(results, key=lambda hit: (-hit[1], hit[0].id))[:limit]


def answer(question, index, *, as_of=None, mode="versioned"):
    try:
        requests = plan(question, index, as_of=as_of)
    except ValueError as exc:
        return Answer("clarify", str(exc))
    if not requests:
        return Answer("clarify", "Уточните, какое условие документа нужно найти.")

    matches = {}
    notes = []
    for part, scope in requests:
        document_id = scope.document_id
        if document_id not in index.documents:
            kind = index.kind(document_id)
            notes.append(f"В корпусе нет {kind}а {document_id}.")
            continue
        if mode == "versioned" and not index.active_clauses(scope):
            notes.append(f"В корпусе нет редакции {document_id} на {scope.as_of:%d.%m.%Y}.")
            continue
        terms = evidence_terms(part)
        candidates = [clause for clause, _ in index.search(part, scope=scope, mode=mode)
                      if terms <= index.evidence[clause.id]]
        if not candidates:
            notes.append(f"В {document_id} нет подтверждения для части запроса «{part}».")
            continue
        # Заголовок помогает отличить условие приёмки от её упоминания в другом пункте.
        clause = max(candidates, key=lambda hit: len(index.headings[hit.id] & terms))
        matches[clause.id] = clause
    if not matches:
        return Answer("insufficient", " ".join(dict.fromkeys(notes)))
    summary = f"Найденные положения на {requests[0][1].as_of:%d.%m.%Y}."
    if notes:
        summary += " " + " ".join(dict.fromkeys(notes))
    return Answer("partial" if notes else "answered", summary,
                  tuple(matches.values()))
