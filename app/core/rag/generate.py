from google import genai
from app.core.config.constant import MODEL_GEMINI

class GeminiClient:
    def __init__(self, api_key: str, model: str = MODEL_GEMINI):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate_text(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        return response.text

    def generate_query_variants(self, query: str, n_variants: int = 5) -> list[str]:
        prompt = f"""
        Tôi cần {n_variants} cách diễn đạt khác nhau cho câu hỏi sau, giữ nguyên nghĩa nhưng thay đổi cách diễn đạt hoặc cấu trúc câu.

        Câu hỏi gốc: "{query}"

        Trả về kết quả dưới dạng danh sách dòng, không giải thích gì thêm.
        """

        response = self.generate_text(prompt)

        # Xử lý text: tách từng dòng có nội dung
        variants = [line.strip("-• ").strip() for line in response.strip().split("\n") if line.strip()]
        return variants[:n_variants]
