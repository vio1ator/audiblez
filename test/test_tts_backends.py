import types
import unittest
from unittest.mock import patch


class FakePipeline:
    def __init__(self, lang_code: str, trf: bool):
        self.lang_code = lang_code
        self.trf = trf
        self.g2p = types.SimpleNamespace(fallback=object())

    def __call__(self, *args, **kwargs):
        return iter(())


class KokoroPipelineFallbackTest(unittest.TestCase):
    def test_misaki_fallback_preserved(self) -> None:
        fake_pipeline = FakePipeline('a', True)

        def pipeline_factory(lang_code: str, trf: bool):
            # Return the pre-built instance so we can inspect its state after wrapper init
            return fake_pipeline

        with patch('kokoro.KPipeline', pipeline_factory):
            from audiblez.tts_backends import KokoroPipelineWrapper

            wrapper = KokoroPipelineWrapper('a')

        self.assertIs(wrapper.pipeline.g2p.fallback, fake_pipeline.g2p.fallback)


if __name__ == '__main__':
    unittest.main()
