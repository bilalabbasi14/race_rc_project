import unittest
import sys
import os

# Add src/ to sys.path so we can import inference directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from inference import (
    load_models,
    verify_answer,
    identify_best_answer,
    generate_distractors,
    generate_hints,
    run_full_pipeline,
    PipelineResult
)

class TestInferencePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Load models once for all tests to speed up execution.
        If any required .pkl files are missing, this will fail and fail test_models_load implicitly.
        """
        cls.bundle = load_models(verbose=False)
        
        # Dummy data for pipeline execution
        cls.article = "The quick brown fox jumps over the lazy dog. It was a sunny day in the forest."
        cls.question = "What did the fox jump over?"
        cls.correct_answer = "The lazy dog"
        cls.options = {
            "A": "A tall fence",
            "B": "The lazy dog",
            "C": "A small bush",
            "D": "A sleeping cat"
        }

    def setUp(self):
        """Clear the session log before each test to ensure isolated analytics."""
        self.bundle.session_log.clear()

    def test_models_load(self):
        """Verify all required .pkl files load without error and populate the bundle."""
        self.assertIsNotNone(self.bundle.logistic_regression)
        self.assertIsNotNone(self.bundle.vectorizer)

    def test_verify_answer(self):
        """Call verify_answer and check it returns a boolean and float confidence."""
        is_correct, confidence = verify_answer(
            self.bundle, 
            self.article, 
            self.question, 
            self.options["B"]
        )
        self.assertIsInstance(is_correct, bool)
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_identify_best_answer(self):
        """Check it returns a valid label in ['A','B','C','D']."""
        best_label, confidence, all_scores = identify_best_answer(
            self.bundle,
            self.article,
            self.question,
            self.options
        )
        self.assertIn(best_label, ['A', 'B', 'C', 'D'])
        self.assertIsInstance(confidence, float)
        self.assertEqual(len(all_scores), 4)

    def test_generate_distractors(self):
        """Check it returns exactly 3 strings, none equal to the answer."""
        distractors = generate_distractors(
            self.bundle,
            self.article,
            self.question,
            self.correct_answer,
            n=3
        )
        self.assertEqual(len(distractors), 3)
        for d in distractors:
            self.assertIsInstance(d, str)
            self.assertNotEqual(d.strip().lower(), self.correct_answer.strip().lower())

    def test_generate_hints(self):
        """Check it returns 3 hints, each a non-empty string."""
        hints = generate_hints(
            self.bundle,
            self.article,
            self.question,
            self.correct_answer
        )
        self.assertEqual(len(hints), 3)
        for h in hints:
            self.assertIsInstance(h, str)
            self.assertGreater(len(h.strip()), 0)

    def test_full_pipeline(self):
        """Run run_full_pipeline end to end, check PipelineResult has all required fields populated."""
        result = run_full_pipeline(
            self.bundle,
            article=self.article,
            options=self.options,
            gold_question=self.question,
            gold_answer_label="B"
        )
        self.assertIsInstance(result, PipelineResult)
        self.assertEqual(result.question, self.question)
        self.assertIn(result.correct_label, ['A', 'B', 'C', 'D'])
        self.assertIsInstance(result.all_scores, dict)
        self.assertEqual(len(result.distractors), 3)
        self.assertEqual(len(result.hints), 3)
        self.assertGreater(result.latency_total, 0.0)

    def test_session_log(self):
        """Verify that after a pipeline call, bundle.session_log has at least one entry with task, latency keys."""
        verify_answer(self.bundle, self.article, self.question, self.options["B"])
        
        log = self.bundle.session_log
        self.assertGreater(len(log), 0)
        
        last_entry = log[-1]
        self.assertIn("task", last_entry)
        self.assertIn("latency", last_entry)

if __name__ == '__main__':
    unittest.main()
