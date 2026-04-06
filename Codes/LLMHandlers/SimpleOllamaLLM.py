from Codes.AbstractClasses.AbstractOllamaLLM import AbstractOllamaLLM


class SimpleOllamaLLM(AbstractOllamaLLM):

    def generate(self, prompt=None, prompt_kwargs=None, **generation_kwargs) -> str:
        prompt = self.get_prompt(prompt, prompt_kwargs)

        response = self._make_generate_request(
            prompt=prompt,
            stream=False,
            **generation_kwargs
        )

        return response.get("response", "")

    def generate_stream(self, prompt=None, prompt_kwargs=None, **generation_kwargs):
        prompt = self.get_prompt(prompt, prompt_kwargs)

        stream = self._make_generate_request(
            prompt=prompt,
            stream=True,
            **generation_kwargs
        )

        return self._parse_stream_response(stream)