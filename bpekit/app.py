import typer

from bpekit.commands import *
from bpekit.utils import get_rank_and_world_size

from pathlib import Path
import os


app = typer.Typer()


@app.command()
def train(

    path: Path = typer.Argument(
        ..., 
        help="Path to the dataset."
    ),

    vocab_size: int = typer.Argument(
        ..., 
        help="Vocabulary size (must be > 256)."
    ),

    merges_path: Path = typer.Option(
        Path("tokenizers/tokenizer.pkl"),
        "--merges-path",
        "--merges_path",
        "-m",
        help="Path to save merges to.",
    ),
    ndocs: int | None = typer.Option(
        None,
        "--ndocs",
        "-n",
        help="Number of dataset entries to encode."
    )
):
    """
    Train and save a new tokenizer.
    """
    rank, world_size = get_rank_and_world_size()
    try:
        train_tokenizer(
            path=path,
            vocab_size=vocab_size,
            merges_path=merges_path,
            ndocs=ndocs,
            rank=rank,
            world_size=world_size,
        )
    except AssertionError as e:
        typer.secho(f"Assertion Error: {e}", fg=typer.colors.RED)
    except Exception as e:
        typer.secho(f"An unexpected error occurred: {e}", fg=typer.colors.RED)


@app.command()
def encode(

    path: Path = typer.Argument(
        ...,
        help="Path to the dataset."
    ),

    merges_path: Path = typer.Argument(
        ...,
        help="Path to tokenizer merges."
    ),

    tokens_path: Path = typer.Option(
        Path("tokens/"),
        "--tokens-path",
        "--tokens_path",
        "-t",
        help="Path to save encoded tokenizer corpus shards to.",
    ),
    shard_size: int = typer.Option(
        int(1e8),
        "--shard-size",
        "--shard_size",
        "-s",
        help="Number of tokens per shard.",
    ),
    batch_size: int = typer.Option(
        16,
        "--batch-size",
        "--batch_size",
        "-b",
        help="Number of datapoints to concatenate an process per iteration.",
    ),
    ndocs: int | None = typer.Option(
        None,
        "--ndocs",
        "-n",
        help="Number of dataset entries to encode."
    )
):
    """
    Encode dataset using a pretrained tokenizer.
    """

    rank, world_size = get_rank_and_world_size()
    try:
        encode_dataset(
            path=path,
            merges_path=merges_path,
            tokens_path=tokens_path,
            shard_size=shard_size,
            batch_size=batch_size,
            ndocs=ndocs,
            rank=rank,
            world_size=world_size,
        )
        typer.echo(f"Dataset encoded and saved to {tokens_path or 'default location'}")
    except AssertionError as e:
        typer.secho(f"Assertion Error: {e}", fg=typer.colors.RED)
    except FileExistsError as e:
        typer.secho(f"File Exists Error: {e}", fg=typer.colors.YELLOW)
    except Exception as e:
        typer.secho(f"An unexpected error occurred: {e}", fg=typer.colors.RED)


@app.command()
def download(
    hf_path: str = typer.Argument(
        ...,
        help="Path to the dataset on the Hugging Face Hub.",
    ),
    path: str = typer.Argument(
        ...,
        help="Path to save the downloaded dataset to.",
    ),
    cache_dir: str = typer.Option(
        None,
        "--cache-dir",
        "--cache_dir",
        "-c",
        help="Directory for Hugging Face to use for caching the dataset. Defaults to ~/.cache/huggingface if not set.",
    ),
    name: str = typer.Option(
        None, 
        "--name", 
        "-n",
        help="Optional name of a specific part of the dataset."
    ),
    split: str = typer.Option(
        "train",
        "--split",
        help="Dataset split to download.",
    ),
    n_proc: int = typer.Option(
        os.cpu_count(),
        "--num-proc",
        "--num_proc",
        "--n-proc",
        "--n_proc",
        "-np",
        help="Number of processes to use. Defaults to the number of CPU cores available.",
    ),
):
    """
    Download a Hugging Face dataset, save it to disk, and clean up cached files.
    """
    try:
        download_dataset(
            hf_path=hf_path,
            path=path,
            cache_dir=cache_dir,
            name=name,
            split=split,
            n_proc=n_proc,
        )
        typer.echo(f"📂 Dataset downloaded and saved to {path}")
    except Exception as e:
        typer.secho(f"An unexpected error occurred: {e}", fg=typer.colors.RED)


if __name__ == "__main__":
    app()
