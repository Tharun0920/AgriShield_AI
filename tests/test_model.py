import os

def test_vision_model_exists():
    """Test if the Day 3 Deep Learning model is safely stored."""
    model_path = "models/plant_disease_model.keras"
    # If the file does not exist, the test will fail and print the message
    assert os.path.exists(model_path) == True, "Vision model file is missing!"

def test_yield_model_exists():
    """Test if the Day 4 Random Forest model is safely stored."""
    model_path = "models/yield_model.pkl"
    assert os.path.exists(model_path) == True, "Yield model file is missing!"