import os
import shlex
import sys
import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class CliTest(unittest.TestCase):
    def cli(self, args):
        exe = shlex.quote(sys.executable)
        root = shlex.quote(str(PROJECT_ROOT))
        cmd = f'cd {root} && {exe} -m audiblez.cli {args}'
        with os.popen(cmd) as pipe:
            return pipe.read()

    def test_help(self):
        out = self.cli('--help')
        self.assertIn('af_sky', out)
        self.assertIn('usage:', out)

    def test_epub(self):
        out = self.cli('epub/mini.epub')
        self.assertIn('Found cover image', out)
        self.assertIn('Creating M4B file', out)
        self.assertTrue(Path('mini.m4b').exists())
        self.assertTrue(Path('mini.m4b').stat().st_size > 256 * 1024)

    def test_epub_voice_and_output_folder(self):
        out = self.cli('epub/mini.epub -v af_sky -o test/prova')
        self.assertIn('Found cover image', out)
        self.assertIn('Creating M4B file', out)
        self.assertTrue(Path('test/prova/mini.m4b').exists())
        self.assertTrue(Path('test/prova/mini.m4b').stat().st_size > 256 * 1024)

    @unittest.skip('Not implemented yet')
    def test_md(self):
        content = (
            '## Italy\n'
            'Italy, officially the Italian Republic, is a country in '
            '(Southern)[https://en.wikipedia.org/wiki/Southern_Europe] and Western Europe. '
            'It consists of a peninsula that extends into the Mediterranean Sea, '
            'with the Alps on its northern land border, '
            'as well as nearly 800 islands, notably Sicily and Sardinia.')
        file_name = NamedTemporaryFile('w', suffix='.txt', delete=False).write(content)
        out = self.cli(file_name)
        self.assertIn('Creating M4B file', out)
        self.assertTrue(Path(file_name).exists())
        self.assertTrue(Path('file_name').stat().st_size > 256 * 1024)

    @unittest.skip('Not implemented yet')
    def test_txt(self):
        content = (
            'Italy, officially the Italian Republic, is a country in Southern and Western Europe. '
            'It consists of a peninsula that extends into the Mediterranean Sea, '
            'with the Alps on its northern land border, '
            'as well as nearly 800 islands, notably Sicily and Sardinia.')
        file_name = NamedTemporaryFile('w', suffix='.txt', delete=False).write(content)
        out = self.cli(file_name)
        self.assertIn('Creating M4B file', out)
        self.assertTrue(Path('text.mp4').exists())
        self.assertTrue(Path('text.mp4').stat().st_size > 256 * 1024)
