"""
Model inference interface for serving a simple logistic regression model.

This module defines the standard functions used by a model-serving framework
(e.g., SageMaker) to load a model, preprocess input data, run predictions,
and format the output response.
"""

import json
import numpy as np


def model_fn(model_dir):
    """
    Load the trained model from disk.

    Parameters
    ----------
    model_dir : str
        Path to the directory containing model artifacts.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        A tuple containing:
        - w: Weight matrix
        - b: Bias term
    """
    w = np.load(f"{model_dir}/weights.npy")
    b = np.load(f"{model_dir}/bias.npy")
    return w, b


def input_fn(request_body, request_content_type):
    """
    Parse and preprocess the incoming request data.

    Parameters
    ----------
    request_body : str or bytes
        The raw request payload.
    request_content_type : str
        The MIME type of the request (e.g., "application/json").

    Returns
    -------
    numpy.ndarray
        Input features as a NumPy array.

    Raises
    ------
    ValueError
        If the content type is unsupported or required fields are missing.
    """
    if request_content_type != "application/json":
        raise ValueError(f"Unsupported content type: {request_content_type}")

    if isinstance(request_body, (bytes, bytearray)):
        request_body = request_body.decode("utf-8")

    data = json.loads(request_body)

    if "inputs" not in data:
        raise ValueError("JSON payload must contain an 'inputs' field")

    return np.array(data["inputs"])


def predict_fn(input_data, model):
    """
    Run inference on the input data using the loaded model.

    Parameters
    ----------
    input_data : numpy.ndarray
        Preprocessed input features.
    model : tuple
        Model parameters as returned by `model_fn` (w, b).

    Returns
    -------
    numpy.ndarray
        Predicted probabilities.
    """
    w, b = model
    z = input_data @ w + b

    # Numerically stable sigmoid
    prob = 1.0 / (1.0 + np.exp(-z))
    return prob


def output_fn(prediction, response_content_type):
    """
    Format the prediction output for the response.

    Parameters
    ----------
    prediction : numpy.ndarray
        Model prediction output.
    response_content_type : str
        Desired MIME type of the response.

    Returns
    -------
    str
        JSON-formatted prediction response.

    Raises
    ------
    ValueError
        If the response content type is unsupported.
    """
    if response_content_type != "application/json":
        raise ValueError(f"Unsupported response content type: {response_content_type}")

    probability = float(np.asarray(prediction).squeeze())

    return json.dumps({
        "probability": probability
    })
