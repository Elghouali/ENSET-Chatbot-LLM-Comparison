"""Validate the five-question benchmark specification without running Ollama."""
import json
import unittest
from pathlib import Path


QUESTIONS_FILE = Path(__file__).resolve().parents[1] / "questions.json"


class BenchmarkQuestionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))

    def test_exactly_five_distinct_question_ids(self):
        self.assertEqual(len(self.questions), 5)
        ids = [question["id"] for question in self.questions]
        self.assertEqual(ids, ["Q1", "Q2", "Q3", "Q4", "Q5"])

    def test_five_required_task_categories(self):
        categories = {question["type"] for question in self.questions}
        self.assertEqual(
            categories,
            {"Explication", "Raisonnement", "Code Python", "Synthèse", "Traduction"},
        )

    def test_each_question_has_a_non_empty_prompt(self):
        for question in self.questions:
            with self.subTest(question=question["id"]):
                self.assertIsInstance(question["prompt"], str)
                self.assertTrue(question["prompt"].strip())


if __name__ == "__main__":
    unittest.main()
