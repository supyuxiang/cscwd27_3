"""Math prompt shared by sampling, training, and evaluation."""

# Exactly the English prompt used by exp1_math eval (prompt_type=language).
SYSTEM_EXPERT = "You are a math expert."
USER_TEMPLATE = (
    "Question: {question} Please reason step by step and put your final answer "
    "within \\boxed{{}}."
)


def build_prompt(question: str) -> str:
    return f"{SYSTEM_EXPERT}\n\n{USER_TEMPLATE.format(question=question)}"
