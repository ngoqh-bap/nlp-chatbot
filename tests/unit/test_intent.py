"""
Unit tests for Intent Classification

Tests the intent detection component of the NLP pipeline.
"""
import pytest


@pytest.mark.unit
@pytest.mark.nlp
class TestIntentClassification:
    """Test intent classification accuracy and edge cases"""

    def test_diem_chuan_intent(self, nlp_service):
        """Test detection of điểm chuẩn intent"""
        messages = [
            "Điểm chuẩn ngành Kiến trúc?",
            "Cho mình hỏi điểm chuẩn",
            "Điểm chuẩn năm 2024 là bao nhiêu?",
        ]
        for msg in messages:
            result = nlp_service.analyze_message(msg)
            assert result["intent"].startswith("hoi_diem"), \
                f"Failed for message: {msg}, got intent: {result['intent']}"
            assert result["score"] > 0.3, \
                f"Low confidence for message: {msg}, score: {result['score']}"

    def test_hoc_phi_intent(self, nlp_service):
        """Test detection of học phí intent"""
        messages = [
            "Học phí ngành CNTT?",
            "Chi phí học tập",
            "Học phí một năm là bao nhiêu?",
        ]
        for msg in messages:
            result = nlp_service.analyze_message(msg)
            assert result["intent"].startswith("hoi_hoc_phi"), \
                f"Failed for message: {msg}, got intent: {result['intent']}"

    def test_hoc_bong_intent(self, nlp_service):
        """Test detection of học bổng intent"""
        messages = [
            "Học bổng có gì?",
            "Trường có học bổng không?",
            "Thông tin học bổng",
        ]
        for msg in messages:
            result = nlp_service.analyze_message(msg)
            assert result["intent"].startswith("hoi_hoc_bong"), \
                f"Failed for message: {msg}, got intent: {result['intent']}"

    def test_fallback_intent(self, nlp_service):
        """Test fallback for unclear messages"""
        messages = [
            "asdfghjkl",  # Random chars
            "???",  # Only punctuation
        ]
        for msg in messages:
            result = nlp_service.analyze_message(msg)
            # Should either be fallback or have very low confidence
            assert result["intent"] == "fallback" or result["score"] < 0.3, \
                f"Should fallback for message: {msg}, got intent: {result['intent']}"

    def test_intent_confidence_threshold(self, nlp_service):
        """Test that confidence scores are within valid range"""
        messages = [
            "Điểm chuẩn ngành Kiến trúc",
            "Học phí",
            "abc xyz",
        ]
        for msg in messages:
            result = nlp_service.analyze_message(msg)
            assert 0.0 <= result["score"] <= 1.0, \
                f"Invalid confidence score for message: {msg}"

    def test_empty_message(self, nlp_service):
        """Test handling of empty messages"""
        messages = ["", "   "]
        for msg in messages:
            result = nlp_service.analyze_message(msg)
            assert result["intent"] == "fallback", \
                f"Empty message should fallback: '{msg}'"
            assert result["score"] == 0.0, \
                f"Empty message should have 0 confidence"


def test_pipeline_constructs_intent_detector_without_type_error():
    from nlu.pipeline import NLPPipeline

    NLPPipeline()
