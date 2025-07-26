import os
from typing import Callable, Optional, Mapping, Any

import pyaudio
from dotenv import load_dotenv
from elevenlabs import ElevenLabs
from elevenlabs.conversational_ai.conversation import (
    Conversation,
    ClientTools,
    AudioInterface,
    ConversationInitiationData,
)

load_dotenv()

AUDIO_CHANNELS = 1  # モノラル
AUDIO_FORMAT = pyaudio.paInt16  # PCM 16000 Hz
AUDIO_SAMPLE_RATE = 16000  # サンプルレート 16kHz
AUDIO_BUFFER_SIZE = 4000  # バッファサイズ（約250ms）
AGENT_ID = str(os.getenv("ELEVEN_LABS_AGENT_ID"))


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
            format=AUDIO_FORMAT,
            channels=AUDIO_CHANNELS,
            rate=AUDIO_SAMPLE_RATE,
            input=True,
            frames_per_buffer=AUDIO_BUFFER_SIZE,
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
                format=AUDIO_FORMAT,
                channels=AUDIO_CHANNELS,
                rate=AUDIO_SAMPLE_RATE,
                output=True,
            )
        self.output_stream.write(audio)

    def interrupt(self) -> None:
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
            self.output_stream = None


def log_message(parameters: dict[str, Any]) -> None:
    message = parameters.get("message", "")
    print(message)


client_tools = ClientTools()  # type: ignore
client_tools.register("logMessage", log_message)
config = ConversationInitiationData(
    dynamic_variables={},
    conversation_config_override={
        "agent": {
            # "first_message": "こんにちは。どうかされました？",
        },
    },
)

conversation = Conversation(
    client=ElevenLabs(api_key=os.getenv("ELEVEN_LABS_API_KEY")),
    agent_id=AGENT_ID,
    client_tools=client_tools,
    audio_interface=MacAudioInterface(),
    requires_auth=False,
    config=config,
)
conversation.start_session()  # type: ignore
