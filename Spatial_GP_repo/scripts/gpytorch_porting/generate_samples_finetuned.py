"""Generate samples from the fine-tuned DDPM.

Thin wrapper around generate_samples.py with fine-tuned defaults.

Usage:
    python generate_samples_finetuned.py
    python generate_samples_finetuned.py --num-images 4 --num-steps 50
"""

from generate_samples import create_parser, main, DEFAULT_FINETUNED_PATH


if __name__ == '__main__':
    parser = create_parser()
    parser.set_defaults(
        model_path=DEFAULT_FINETUNED_PATH,
        output_dir='generated_samples_finetuned',
    )
    args = parser.parse_args()
    main(args)
