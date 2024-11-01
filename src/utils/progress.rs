use atty::Stream;
use indicatif::{ProgressBar,ProgressDrawTarget,ProgressStyle};

use std::io::{self, Write};

pub(crate) enum ProgressReporter {
    Bar(ProgressBar),
    Spinner(ProgressBar),
    Printer(ProgressPrinter),
    None,
}

impl ProgressReporter {
    pub fn inc(&mut self, delta: u64) {
        match self {
            Self::Bar(pb) => pb.inc(delta),
            Self::Spinner(ps) => ps.inc(delta),
            Self::Printer(pp) => pp.inc(delta),
            Self::None => {},
        }
    }

    pub fn finish(&self) {
        match self {
            Self::Bar(pb) => pb.finish(),
            Self::Spinner(ps) => ps.finish(),
            Self::Printer(pp) => pp.finish(),
            Self::None => {},
        }
    }
}

// For printing progess to files
pub(crate) struct ProgressPrinter {
    length: Option<u64>,
    current: u64,
    interval: u64,
    last_print: u64,
    verbose: bool
}

impl ProgressPrinter {

    fn new(length: Option<u64>, interval: u64, verbose: bool) -> Self {

        ProgressPrinter{
            length,
            current: 0,
            interval,
            last_print: 0,
            verbose
        }
    }


    fn inc(&mut self, delta: u64) {

        self.current += delta;

        if !self.verbose || self.current - self.last_print < self.interval {
            return;
        }

        self.last_print = self.current;

        match self.length {
            Some(length) => {
                let progress_percentage = (self.current * 100) / length;
                println!("Progress: {}%", progress_percentage);
            }
            None => {
                println!("Progress: {} iterations completed", self.current);
            }
        }

        // Ensure output is flushed immediately
        io::stdout().flush().unwrap();
    }

    fn finish(&self) {
        println!("Progress: 100% complete, {} iterations completed", self.current);
    }
}

/// Gets `ProgressReporter` that suits scenario, based on `rank` 
/// and whether or not we are writing to a terminal. 
/// `interval` and `verbose` are only relevant in the latter scenario 
/// (e.g. when writing output to a log file).
pub fn get_progress_reporter(
    length: Option<u64>, 
    rank: i32, 
    interval: u64, 
    verbose: bool
) -> ProgressReporter {

    if rank != 0 {
        return ProgressReporter::None
    }

    // Determine if the standard output is a terminal
    let is_tty = atty::is(Stream::Stdout);

    // Match on the combination of `total` and `is_tty`
    match (length, is_tty) {
        (Some(len), true) => {
            let pb = ProgressBar::new(len);
            pb.set_draw_target(ProgressDrawTarget::stdout_with_hz(3));
            pb.set_style(
                ProgressStyle::with_template(
                    "[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} ({eta})",
                )
                .unwrap(),
            );
            ProgressReporter::Bar(pb)
        }
        (length, false) => {
            let pp = ProgressPrinter::new(length, interval, verbose);
            ProgressReporter::Printer(pp)
        }
        (None, true) => {
            let ps = ProgressBar::new_spinner();
            ProgressReporter::Spinner(ps)
        }
    }
}