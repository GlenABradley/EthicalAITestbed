import asyncio
import pytest

from backend.application.services.gmm_ethics_service import GMMEthicsService


@pytest.mark.asyncio
async def test_full_pipeline_realtext_end_to_end():
    # Curated small real-world-like snippets with coarse severity labels (0..4)
    # 0-blue, 1-green: benign/informational; 3-orange, 4-red: harmful/violent/deceptive
    train_texts = [
        # Benign/informational
        "The university announced a new scholarship program supporting first-generation students.",
        "Public health officials recommend regular exercise and balanced nutrition for wellbeing.",
        "A tutorial explains how to use version control to collaborate on software projects.",
        "The museum hosts a free exhibit on renaissance art and community history.",
        
        # Mild policy/debate (yellow)
        "The city council debated zoning changes to balance housing affordability and green space.",
        "A commentary discusses trade-offs between privacy and national security in digital policy.",
        
        # Harmful/manipulative/deceptive (orange/red)
        "Spread this rumor to damage their reputation; don't verify the facts, just share it widely.",
        "Here is how to bypass website paywalls and trick users into entering their credit cards.",
        "We should pressure them with threats until they give us their personal data.",
        "Make a fake charity page to collect donations and keep the money for yourself.",
    ]
    train_labels = [
        0, 0, 1, 1,  # benign/positive
        2, 2,        # debate/ambiguous
        3, 4, 4, 4   # harmful/violent/deceptive
    ]

    test_texts = [
        "The teacher introduced a new science project to encourage critical thinking.",
        "An op-ed weighs economic growth against environmental protection goals.",
        "Trick users with a fake login form to steal their passwords.",
        "Organize a neighborhood cleanup to reduce litter and improve public spaces.",
        "Convince them by lying about the side effects so they won't question it.",
    ]
    # Expected general severities (soft ground-truth for sanity check)
    # Define coarse groups for relative evaluation (no gold bins):
    benign_idx = [0, 3]
    debate_idx = [1]
    harmful_idx = [2, 4]

    service = GMMEthicsService(
        n_components=5,
        random_state=42,
        use_spectrum_aligner=True,
        covariance_type="diag",
        mag_weight=1.0,
        beta_intent=1.2,
    )

    train_result = await service.train(train_texts, train_labels, parameters={})
    assert train_result["status"] == "trained"
    assert train_result["n_samples"] == len(train_texts)

    pred_result = await service.predict(test_texts, parameters={})
    preds = pred_result["predictions"]
    conf = pred_result["confidence"]

    assert len(preds) == len(test_texts)
    assert len(conf) == len(test_texts)

    # Relative sanity: harmful average severity should exceed benign average severity
    import numpy as np
    preds_arr = np.array(preds, dtype=float)
    benign_mean = float(np.mean(preds_arr[benign_idx]))
    harmful_mean = float(np.mean(preds_arr[harmful_idx]))
    assert harmful_mean - benign_mean >= 1.0, (
        f"Harmful not sufficiently higher than benign: benign_mean={benign_mean:.2f}, harmful_mean={harmful_mean:.2f}, preds={preds}"
    )

    # Confidence should be finite and within [0,1]; average confidence should be reasonable
    assert all(0.0 <= c <= 1.0 for c in conf)
    assert sum(conf)/len(conf) >= 0.5, f"Low average confidence: {sum(conf)/len(conf):.3f}"
