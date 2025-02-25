import os
from typing import Callable, Optional, Mapping

import pyaudio
from dotenv import load_dotenv
from elevenlabs import ElevenLabs
from elevenlabs.conversational_ai.conversation import (
    Conversation,
    ClientTools,
    AudioInterface,
)

load_dotenv()


class MacAudioInterface(AudioInterface):
    def __init__(self) -> None:
        self.pyaudio_instance: pyaudio.PyAudio = pyaudio.PyAudio()
        self.input_stream: Optional[pyaudio.Stream] = None
        self.output_stream: Optional[pyaudio.Stream] = None
        self.input_callback: Optional[Callable[[bytes], None]] = None
        self.running: bool = False

    def start(self, input_callback: Callable[[bytes], None]) -> None:
        self.input_callback = input_callback
        self.running = True
        self.input_stream = self.pyaudio_instance.open(
            format=pyaudio.paInt16,  # 16-bit PCM
            channels=1,  # モノラル
            rate=16000,  # サンプルレート 16kHz
            input=True,
            frames_per_buffer=4000,  # 推奨チャンクサイズ（約250ms）
            stream_callback=self._input_stream_callback,
        )
        self.input_stream.start_stream()

    def _input_stream_callback(
        self,
        in_data: Optional[bytes],
        frame_count: int,
        time_info: Mapping[str, float],
        status: int,
    ) -> tuple[Optional[bytes], int]:
        if self.running and self.input_callback and in_data is not None:
            self.input_callback(in_data)
        return None, pyaudio.paContinue

    def stop(self) -> None:
        self.running = False
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
            self.input_stream = None
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
            self.output_stream = None
        self.pyaudio_instance.terminate()

    def output(self, audio: bytes) -> None:
        if self.output_stream is None:
            self.output_stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16,  # 16-bit PCM
                channels=1,  # モノラル
                rate=16000,  # サンプルレート 16kHz
                output=True,
            )
        self.output_stream.write(audio)

    def interrupt(self) -> None:
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
            self.output_stream = None


def log_message(parameters: dict[str, str]) -> None:
    message = parameters.get("message", "")
    print(message)


client_tools = ClientTools()  # type: ignore
client_tools.register("logMessage", log_message)

conversation = Conversation(
    client=ElevenLabs(api_key=os.getenv("ELEVEN_LABS_API_KEY")),
    agent_id="MHxOLcV2h0PXbuuV3h6Z",
    client_tools=client_tools,
    audio_interface=MacAudioInterface(),
    requires_auth=False,
)
conversation.start_session()  # type: ignore
