"""Tests for src.evaluation.response_evaluator (rule-based, no LLM calls)."""

from src.evaluation.response_evaluator import evaluate_ocr_result, evaluate_scene_analysis
from src.llm.schemas import OcrClassification, SceneAnalysis


def test_evaluate_scene_analysis_rewards_navigation_cues_and_safe_hazard():
    analysis = SceneAnalysis(
        scene_summary="A chair is ahead on your left, and a table is to your right.",
        environment_type="home",
        important_objects=["A chair is ahead on your left."],
        hazard_level="Safe",
        hazard_explanation="Nothing hazardous is visible.",
        suggested_action="Proceed as normal.",
        audio_summary="You are in a home with a chair ahead and a table to your right.",
        scene_confidence=0.9,
        hazard_confidence=0.9,
    )

    evaluation = evaluate_scene_analysis(analysis)

    assert evaluation.completeness_score == 100
    assert evaluation.safety_score > 80  # Safe + high confidence
    assert evaluation.navigation_usefulness_score > 50  # "ahead"/"left"/"right" cues
    assert evaluation.ocr_usefulness_score is None


def test_evaluate_scene_analysis_penalizes_low_hazard_confidence():
    confident = SceneAnalysis(
        scene_summary="A hallway.",
        environment_type="office",
        hazard_level="Dangerous",
        hazard_explanation="Fire visible ahead.",
        important_objects=["Flames are ahead of you."],
        suggested_action="Move away immediately.",
        audio_summary="There is a fire ahead, move away immediately.",
        scene_confidence=0.9,
        hazard_confidence=0.9,
    )
    unsure = confident.model_copy(update={"hazard_confidence": 0.1})

    assert evaluate_scene_analysis(unsure).safety_score < evaluate_scene_analysis(confident).safety_score


def test_evaluate_ocr_result_scores_higher_with_classification_and_confidence():
    with_classification = evaluate_ocr_result(
        "Ibuprofen 200mg - take with food",
        ocr_confidence=0.9,
        classification=OcrClassification(
            category="Medicine label",
            suggested_action="Check the expiry date before use.",
            classification_confidence=0.8,
        ),
    )
    without_classification = evaluate_ocr_result("", ocr_confidence=0.0, classification=None)

    assert with_classification.completeness_score > without_classification.completeness_score
    assert with_classification.ocr_usefulness_score > without_classification.ocr_usefulness_score
