import together  # pip install together
import os

class Reflection:
    def __init__(self, model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"):
        self.client = together.Together(api_key=os.getenv("TOGETHER_API_KEY"))
        self.model = model

    def _concat_and_format_texts(self, data):
        concatenatedTexts = []
        for entry in data:
            role = entry.get('role', '')
            if entry.get('parts'):
                all_texts = ' '.join(part['text'] for part in entry['parts'])
            elif entry.get('content'):
                all_texts = entry['content']
            concatenatedTexts.append(f"{role}: {all_texts}\n")
        return ''.join(concatenatedTexts)

    def __call__(self, chatHistory, lastItemsConsidereds=10):
        if len(chatHistory) >= lastItemsConsidereds:
            chatHistory = chatHistory[-lastItemsConsidereds:]

        historyString = self._concat_and_format_texts(chatHistory)

        prompt = f"""Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question in Vietnamese which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is.\n\n{historyString}"""

        print(prompt)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )

        return response.choices[0].message.content
