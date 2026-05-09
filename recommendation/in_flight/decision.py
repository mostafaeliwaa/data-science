def make_decision(final_score):
    if final_score >= 0.7:
        return "KEEP"
    elif final_score >= 0.4:
        return "OPTIMIZE"
    else:
        return "STOP"
