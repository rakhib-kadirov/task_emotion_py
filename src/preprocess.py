import re

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\d+', '', text)       # удаление чисел
    text = re.sub(r'[^\w\s]', '', text)   # удаление пунктуации
    text = text.strip()
    return text




