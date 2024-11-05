use atty::Stream;
use indicatif::{ProgressBar,ProgressDrawTarget,ProgressStyle,ProgressFinish};

use std::env;
use std::io::{self, Write};
use std::iter::Iterator;


/// Progress that either shows a full progress bar/spinner or prints simple messages.
/// It is designed to work somewhat gracefully with both tty stdout and non-tty stdout.
pub(crate) struct Progress {
    pub bar: ProgressBar,
    to_tty: bool,
    finish_message: Option<String>
}

impl Progress {

    /// - If stdout is a TTY, it creates a standard progress bar.
    /// - If stdout is redirected, it just prints start and finish messages.
    pub(crate) fn new(
        len: Option<usize>,
        rank: i32,
        message: &str,
        finish_message: Option<&str>
    ) -> Self {

        if rank != 0 {
            // This progress bar will not interfere at all
            return Progress{
                bar: ProgressBar::hidden(),
                to_tty: true,
                finish_message: Some("".to_string())
            }
        }

        // Determine if stdout is a terminal
        let to_tty = match env::var("OMPI_COMM_WORLD_SIZE") {
            // Hacky way to ensure that when ran with `mpirun` we 
            // automatically assume we are sending stdout to a file
            Ok(_) => false, 
            Err(_) => atty::is(Stream::Stdout)
        };    

        let msg = String::from(message);
        let finish_msg = finish_message.map(|s| s.to_string());
        
        let finish = match finish_msg.clone() {
            Some(s) => ProgressFinish::WithMessage(s.into()),
            None => ProgressFinish::AndLeave
        };    

        let pb = match (len, to_tty) {
            (Some(l), true) => {
                let bar = ProgressBar::new(l as u64);
                bar.set_draw_target(ProgressDrawTarget::stdout_with_hz(3));
                bar.set_style(
                    ProgressStyle::with_template(
                        "{msg:20} [{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} ({eta})",
                    )
                    .unwrap()
                );
                bar
            }
            (None, true) => {
                let spinner = ProgressBar::new_spinner();
                spinner.set_draw_target(ProgressDrawTarget::stdout_with_hz(3));
                spinner.set_style(
                    ProgressStyle::with_template(
                        "{msg:20} [{elapsed_precise}] {spinner} {pos:>7}",
                    )
                    .unwrap()
                    .tick_chars("🌖🌗🌘🌑🌒🌓🌔🌕"),
                );
                spinner
            }
            (_, false) => {
                println!("{}",msg);
                ProgressBar::hidden()
            }
        };

        pb.set_message(msg);

        Progress { 
            bar: pb.with_finish(finish),
            to_tty,
            finish_message: finish_msg 
        }
    }

    pub fn inc(&mut self, delta: u64) {
        self.bar.inc(delta);
    }

    /// For manual finishing of Progress (e.g. when used in a while loop)
    pub fn finish(&self) {
        if !self.to_tty {
            let msg = self.finish_message.as_deref().unwrap_or("Progress 100% complete");
            
            println!(
                "{}, {} iterations completed in {} seconds",
                msg,
                self.bar.position(),
                self.bar.elapsed().as_secs()
            );
            io::stdout().flush().unwrap(); // Force flush
        } else if let Some(msg) = &self.finish_message {
            self.bar.finish_with_message(msg.clone());
        } else {
            self.bar.finish();
        }
    }
}

/// Now we need to allow `Progress` to be "attached" to iterators
pub struct ProgressIter<I>
where
    I: Iterator,
{
    iter: I,
    progress: Progress,
}

impl<I> ProgressIter<I>
where
    I: Iterator,
{
    fn new(iter: I, progress: Progress) -> Self {
        ProgressIter { iter, progress }
    }
}

impl<I> Iterator for ProgressIter<I>
where
    I: Iterator,
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {

        let item = self.iter.next();

        if let Some(_) = item {

            self.progress.bar.inc(1);
        
        // if we are printing to a tty then finishing will be automatically
        // handled. Otherwise, we finish manually:
        } else if !self.progress.to_tty {
            self.progress.finish();
        }
        
        item
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

pub trait ProgressIteratorExt: Iterator + Sized {
    fn attach_progress(self, progress: Progress) -> ProgressIter<Self> {
        ProgressIter::new(self, progress)
    }
}

impl<I> ProgressIteratorExt for I where I: Iterator {}
