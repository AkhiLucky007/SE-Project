def class_to_label(pred):
    return {
        0: "Low",
        1: "Medium",
        2: "High"
    }[pred]