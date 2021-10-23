import editdistance


def calc_cer(target_text: str, predicted_text: str) -> float:
    target_words = target_text.split()
    pred_words = predicted_text.split()

    if len(target_words) != 0:
        return editdistance.eval(target_words, pred_words) / len(target_words)
        
    else:
        return editdistance.eval(target_words, pred_words)

def calc_wer(target_text: str, predicted_text: str) -> float:
    if len(target_text) != 0:
        return editdistance.eval(target_text, predicted_text) / len(target_text)

    else:
        return editdistance.eval(target_text, predicted_text)
