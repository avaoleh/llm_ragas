import os
import json
import pytest
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from datasets import Dataset
from src.rag_app import build_rag_engine, get_response

# Пороговые значения для Quality Gates
THRESHOLD_FAITHFULNESS = 0.7
THRESHOLD_ANSWER_RELEVANCY = 0.7
THRESHOLD_CONTEXT_RECALL = 0.7


@pytest.fixture(scope="module")
def query_engine():
    """Инициализируем RAG движок один раз для всех тестов."""
    # Убедимся, что токен доступен
    if not os.getenv("HF_TOKEN"):
        pytest.skip("HF_TOKEN не установлен. Пропускаем тесты.")
    return build_rag_engine()


@pytest.fixture(scope="module")
def goldens():
    """Загружаем золотые данные."""
    with open("tests/goldens.json", "r", encoding="utf-8") as f:
        return json.load(f)


def test_rag_evaluation(query_engine, goldens):
    """
    Основной тест: генерирует ответы, оценивает их через Ragas и проверяет пороги.
    """
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    # Генерация ответов моделью для каждого вопроса из goldens
    print("\n--- Генерация ответов моделью ---")
    for item in goldens:
        question = item["question"]
        gt_answer = item["answer"]
        # В реальном сценарии контекст берется из retrieval, здесь мы имитируем процесс
        # Но для метрики Context Recall нам нужен именно retrieved контекст.
        # Поэтому мы делаем запрос к движку, чтобы получить реальный контекст и ответ.
        ans, retrieved_contexts = get_response(query_engine, question)

        questions.append(question)
        answers.append(ans)
        contexts.append(retrieved_contexts)  # Список списков строк
        ground_truths.append([gt_answer])  # Ragas ожидает список строк для GT

    # Формирование датасета для Ragas
    dataset_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    eval_dataset = Dataset.from_dict(dataset_dict)

    # Запуск оценки Ragas
    print("\n--- Запуск оценки Ragas ---")
    result = evaluate(
        eval_dataset,
        metrics=[faithfulness, answer_relevancy, context_recall],
        raise_exceptions=False  # Чтобы тест не падал сразу при ошибке API, а мы обработали это
    )

    # Вывод результатов в консоль
    df_result = result.to_pandas()
    print(df_result)

    # Сохранение подробного отчета в JSON для артефактов CI
    results_json = result.to_dict()
    with open("ragas_results.json", "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)

    # Извлечение средних значений метрик
    avg_faithfulness = result['faithfulness']
    avg_relevancy = result['answer_relevancy']
    avg_recall = result['context_recall']

    print(f"\nСредний Faithfulness: {avg_faithfulness:.4f}")
    print(f"Средний Answer Relevancy: {avg_relevancy:.4f}")
    print(f"Средний Context Recall: {avg_recall:.4f}")

    # ASSERTS (Quality Gates)
    # Пайплайн упадет здесь, если метрики ниже порога
    assert avg_faithfulness >= THRESHOLD_FAITHFULNESS, \
        f"Faithfulness ({avg_faithfulness:.4f}) ниже порога ({THRESHOLD_FAITHFULNESS}). Обнаружены галлюцинации!"

    assert avg_relevancy >= THRESHOLD_ANSWER_RELEVANCY, \
        f"Answer Relevancy ({avg_relevancy:.4f}) ниже порога ({THRESHOLD_ANSWER_RELEVANCY})."

    assert avg_recall >= THRESHOLD_CONTEXT_RECALL, \
        f"Context Recall ({avg_recall:.4f}) ниже порога ({THRESHOLD_CONTEXT_RECALL}). Контекст не покрывает ответ."