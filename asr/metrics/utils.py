import editdistance


def calc_wer(target_text: str, pred_text: str) -> float:
    target_words = target_text.split()
    pred_words = pred_text.split()

    if target_text == "":
        return 1
    
    return editdistance.eval(target_words, pred_words) / len(target_words)


def calc_cer(target_text: str, pred_text: str):
    if target_text == "":
        return 1
    
    return editdistance.eval(target_text, pred_text) / len(target_text)
