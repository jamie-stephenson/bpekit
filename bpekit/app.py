import typer

from bpekit import train_tokenizer, encode_dataset

from pathlib import Path


app = typer.Typer()

@app.command()
def train(

    path: Path = typer.Argument(
        ..., 
        help="Path to the dataset."
    ),

    vocab_size: int = typer.Argument(
        ..., 
        help="Vocabulary size."
    ),

    merges_path: Path = typer.Option(
        Path('tokenizers/'),
        "--merges-path",
        "--merges_path",
        "-m",
        help="Path to save merges to."
    ),

    tokens_path: Path | None = typer.Option(
        None,
        "--tokens-path",
        "--tokens_path",
        "-t",
        help="Path to save encoded tokenizer corpus shards to."
    ),

    shard_size: int = typer.Option(
        int(1e8),
        "--shard-size",
        "--shard_size",
        "-s",
        help="Number of tokens per shard."
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
    try:
        train_tokenizer(
            path=path,
            vocab_size=vocab_size,
            merges_path=merges_path,
            tokens_path=tokens_path,
            shard_size=shard_size,
            ndocs=ndocs
        )
        typer.echo(f"Tokenizer trained and saved to {merges_path}")
        if tokens_path:
            typer.echo(f"Encoded tokenizer corpus shards saved to {tokens_path}")
    except AssertionError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED)
    except Exception as e:
        typer.secho(f"An unexpected error occurred: {e}", fg=typer.colors.RED)

@app.command()
def encode(

    path: Path = typer.Argument(
        ...,
        help="Path to the dataset."
    ),

    merges_path: Path = typer.Option(
        ...,
        "--merges-path",
        "--merges_path",
        "-m",
        help="Path to tokenizer merges."
    ),

    tokens_path: Path | None = typer.Option(
        None,
        "--tokens-path",
        "--tokens_path",
        "-t",
        help="Path to save encoded tokenizer corpus shards to."
    ),

    shard_size: int = typer.Option(
        int(1e8),
        "--shard-size",
        "--shard_size",
        "-s",
        help="Number of tokens per shard."
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
    try:
        encode_dataset(
            path=path,
            merges_path=merges_path,
            tokens_path=tokens_path,
            shard_size=shard_size,
            ndocs=ndocs
        )
        typer.echo(f"Dataset encoded and saved to {tokens_path or 'default location'}")
    except AssertionError as e:
        typer.secho(f"Assertion Error: {e}", fg=typer.colors.RED)
    except FileExistsError as e:
        typer.secho(f"File Exists Error: {e}", fg=typer.colors.YELLOW)
    except Exception as e:
        typer.secho(f"An unexpected error occurred: {e}", fg=typer.colors.RED)

if __name__ == "__main__":
    app()