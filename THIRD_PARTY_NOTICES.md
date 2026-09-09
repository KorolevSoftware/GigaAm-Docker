# Third-party components

The application source is distributed under the repository's Apache-2.0
license. Model artifacts and runtime components retain their own licenses:

- GigaAM and the selected ONNX model exports: MIT. See the
  [GigaAM repository](https://github.com/salute-developers/GigaAM) and the
  [pinned export repository](https://huggingface.co/istupakov/gigaam-v3-onnx/tree/322c3b29492673eb7d0b434bfa9dfb8653e34d02).
- Silero VAD: MIT; [upstream](https://github.com/snakers4/silero-vad).
- ONNX Runtime: MIT; its LICENSE and ThirdPartyNotices.txt are copied into
  `/usr/share/licenses/onnxruntime` in the runtime image.
- onnxruntime_go: MIT; [upstream](https://github.com/yalue/onnxruntime_go).
- zerolog: MIT; [upstream](https://github.com/rs/zerolog).
- FFmpeg and system libraries: the Debian packages retain their copyright
  and license notices under `/usr/share/doc`; consult the installed build's
  configuration for the applicable LGPL/GPL components.

The validation-only Python dependencies are not included in the runtime image.
No user audio is included in the distribution. The small frontend test fixture
is a generated mathematical signal, not a recording of a person.
