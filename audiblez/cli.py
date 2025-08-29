# -*- coding: utf-8 -*-
import argparse
import sys

from audiblez.voices import voices, available_voices_str


def cli_main():
    voices_str = ', '.join(voices)
    epilog = ('example:\n' +
              '  audiblez book.epub -l en-us -v af_sky\n\n' +
              'to run GUI just run:\n'
              '  audiblez-ui\n\n' +
              'available voices:\n' +
              available_voices_str)
    default_voice = 'af_sky'
    parser = argparse.ArgumentParser(epilog=epilog, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('epub_file_path', help='Path to the epub file')
    parser.add_argument('-v', '--voice', default=default_voice, help=f'Choose narrating voice: {voices_str}')
    parser.add_argument('-p', '--pick', default=False, help=f'Interactively select which chapters to read in the audiobook', action='store_true')
    parser.add_argument('-s', '--speed', default=1.0, help=f'Set speed from 0.5 to 2.0', type=float)
    parser.add_argument('-c', '--cuda', default=False, help=f'Use GPU via CUDA (Torch) if available', action='store_true')
    parser.add_argument('--mps', default=False, help='Use Apple GPU via MPS (Torch) if available', action='store_true')
    parser.add_argument('-o', '--output', default='.', help='Output folder for the audiobook and temporary files', metavar='FOLDER')

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    # Device selection: CUDA, MPS, or CPU
    if args.cuda or args.mps:
        import torch
        chosen = None
        if args.cuda and args.mps:
            print('Both --cuda and --mps specified; preferring CUDA if available, otherwise MPS.')
        if args.cuda and torch.cuda.is_available():
            chosen = 'cuda'
        elif args.mps and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            chosen = 'mps'
        elif args.cuda and not torch.cuda.is_available() and args.mps and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            chosen = 'mps'
        if chosen:
            torch.set_default_device(chosen)
            print(f'Using device: {chosen}')
        else:
            print('Requested GPU not available. Using CPU.')

    from core import main
    main(args.epub_file_path, args.voice, args.pick, args.speed, args.output)


if __name__ == '__main__':
    cli_main()
