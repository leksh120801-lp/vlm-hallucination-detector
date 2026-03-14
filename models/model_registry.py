from models.clip_model import load_clip_model


def load_model_by_name(model_name):

    if model_name == "CLIP":
        return load_clip_model()

    # future models
    # elif model_name == "BLIP":
    #     return load_blip_model()

    # elif model_name == "SigLIP":
    #     return load_siglip_model()

    else:
        raise ValueError(f"Unknown model: {model_name}")