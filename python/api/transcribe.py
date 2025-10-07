from python.helpers.api import ApiHandler, Request, Response

from python.helpers import runtime, settings, whisper

class Transcribe(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict | Response:
        audio = input.get("audio")
        # ctxid = input.get("ctxid", "")

        # context = self.get_context(ctxid)
        # if not await whisper.is_downloaded():
        #     context.log.log(type="info", content="Whisper STT model is currently being initialized, please wait...")

        set = settings.get_settings()
        requested_language = input.get("language")
        language = str(requested_language).strip() if requested_language else ""
        if not language:
            language = set["stt_language"]

        result = await whisper.transcribe(set["stt_model_size"], audio, language=language) # type: ignore
        return result
