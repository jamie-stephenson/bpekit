use crate::utils::progress::Progress;

use std::fs;
use std::path::Path;

use anyhow::Result;
use ndarray::{Array1, ArrayView1, s};
use ndarray_npy::write_npy;

pub(super) fn save_tokens<I>(tokens_iter: I, path: &Path, shard_size: usize, rank: i32) -> Result<()>
where
    I: Iterator<Item = Vec<u32>>,
{
    // Create directory if it doesn't exist
    fs::create_dir_all(path)?;

    let mut split = "train";
    let mut shard_index = 0;
    let mut all_tokens_np = Array1::<u32>::zeros(shard_size);
    let mut token_count = 0;

    // Initialize progress bar if rank is 0
    let mut progress = Progress::new(
        Some(shard_size),
        rank,
        &format!("Shard {}", shard_index),
        None
    );

    for mut tokens in tokens_iter {
        while token_count + tokens.len() >= shard_size {
            let remainder = shard_size - token_count;

            // Copy tokens into the buffer
            all_tokens_np
                .slice_mut(s![token_count..token_count + remainder])
                .assign(&ArrayView1::from(&tokens[..remainder]));

            progress.inc(remainder as u64);

            // Save the current shard to a file
            let filename = format!("{}_{}__{:06}", rank, split, shard_index);
            let filepath = path.join(filename);
            write_npy(&filepath, &all_tokens_np)?;

            shard_index += 1;
            token_count = 0;
            tokens = tokens[remainder..].to_vec();

            // Reset progress bar for the new shard
            progress.finish();
            progress = Progress::new(
                    Some(shard_size),
                    rank,
                    &format!("Shard {}", shard_index),
                    None
            );
        }

        // Append remaining tokens to the current shard
        let len = tokens.len();
        all_tokens_np
            .slice_mut(s![token_count..token_count + len])
            .assign(&ArrayView1::from(&tokens));
        token_count += len;

        progress.inc(len as u64);
    }

    // Save any remaining tokens
    if token_count != 0 {
        split = if shard_index == 0 { "train" } else { "val" };
        let filename = format!("{}_{}__{:06}", rank, split, shard_index);
        let filepath = path.join(filename);
        let slice = all_tokens_np.slice(s![0..token_count]).to_owned();
        write_npy(&filepath, &slice)?;
    }

    // Finish progress bar
    progress.finish();

    Ok(())
}
