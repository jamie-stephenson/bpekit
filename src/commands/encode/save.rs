use std::fs::{self, File};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;
use ndarray::{Array1, ArrayView1, s};
use ndarray_npy::{write_npy, WriteNpyExt};
use rayon::prelude::*;

/// Saves tokens from a parallel iterator to fixed-size Numpy shard files.
///
/// # Arguments
///
/// - `tokens_iter`: A Rayon `ParallelIterator` that yields vectors of `u16` tokens.
/// - `path`: The directory path where the shard files will be saved.
/// - `shard_size`: The fixed size of each shard (number of tokens per shard).
/// - `rank`: An identifier used in the file naming (useful in distributed settings).
pub fn save_tokens(
    tokens_iter: impl ParallelIterator<Item = Vec<u16>>,
    path: &Path,
    shard_size: usize,
    rank: usize,
) -> Result<()> {
    // Create the directory if it doesn't exist
    fs::create_dir_all(&path)?;

    // Atomic shard index to ensure unique shard numbering across threads
    let shard_counter = AtomicUsize::new(0);

    // Process tokens in parallel
    tokens_iter.try_for_each(|tokens| {
        let mut token_count = 0;
        let mut all_tokens = Array1::<u16>::zeros(shard_size);
        let mut tokens_offset = 0;

        while token_count + (tokens.len() - tokens_offset) >= shard_size {
            let space_left = shard_size - token_count;
            let tokens_to_copy = &tokens[tokens_offset..tokens_offset + space_left];

            // Copy tokens into the shard buffer
            all_tokens
                .slice_mut(s![token_count..])
                .assign(&ArrayView1::from(tokens_to_copy));

            // Atomically get the next shard index
            let shard_index = shard_counter.fetch_add(1, Ordering::SeqCst);

            // Save the full shard to a Numpy file
            let filename = format!("shard_{:06}_rank_{:03}.npy", shard_index, rank);
            let filepath = path.join(filename);
            let file = File::create(filepath)?;
            all_tokens.write_npy(file)?;

            // Reset counters and increment offsets
            token_count = 0;
            tokens_offset += space_left;
        }

        // Copy remaining tokens to the shard buffer
        let remaining_tokens = &tokens[tokens_offset..];
        let end = token_count + remaining_tokens.len();
        all_tokens
            .slice_mut(s![token_count..end])
            .assign(&ArrayView1::from(remaining_tokens));

        token_count = end;

        // If the buffer is full, write it out
        if token_count == shard_size {
            let shard_index = shard_counter.fetch_add(1, Ordering::SeqCst);
            let filename = format!("shard_{:06}_rank_{:03}.npy", shard_index, rank);
            let filepath = path.join(filename);
            let file = File::create(filepath)?;
            all_tokens.write_npy(file)?;
            token_count = 0;
        }

        // If there are leftover tokens after processing, save them
        if token_count > 0 {
            let shard_index = shard_counter.fetch_add(1, Ordering::SeqCst);
            let filename = format!("shard_{:06}_rank_{:03}_partial.npy", shard_index, rank);
            let filepath = path.join(filename);
            let file = File::create(filepath)?;
            all_tokens.slice(s![0..token_count]).write_npy(file)?;
        }

        Ok(())
    })
}
