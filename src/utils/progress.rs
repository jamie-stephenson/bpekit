use atty::Stream;
use indicatif::{ProgressBar,ProgressDrawTarget,ProgressStyle};

use std::io::{self, Write};
use std::env;
use std::time::Instant;

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

    // pub fn finish(&self) {
    //     match self {
    //         Self::Bar(pb) => pb.finish(),
    //         Self::Spinner(ps) => ps.finish(),
    //         Self::Printer(pp) => pp.finish(),
    //         Self::None => {},
    //     }
    // }
    
    pub fn finish_with_message(&self, message: &str) {
        let msg = String::from(message);
        match self {
            Self::Bar(pb) => pb.finish_with_message(msg),
            Self::Spinner(ps) => ps.finish_with_message(msg),
            Self::Printer(pp) => pp.finish_with_message(message),
            Self::None => {},
        }
    }
}

// For printing progess to files
pub(crate) struct ProgressPrinter {
    length: Option<usize>,
    current: u64,
    interval: u64,
    last_print: u64,
    verbose: bool,
    start: Instant
}

impl ProgressPrinter {

    fn new(length: Option<usize>, interval: u64, verbose: bool) -> Self {

        ProgressPrinter{
            length,
            current: 0,
            interval,
            last_print: 0,
            verbose,
            start: Instant::now()
        }
    }


    fn inc(&mut self, delta: u64) {

        self.current += delta;

        if !self.verbose || self.current - self.last_print < self.interval {
            return;
        }

        self.last_print = self.current;

        let elapsed = self.start.elapsed().as_secs();

        match self.length {
            Some(length) => {
                let progress_percentage = (self.current * 100) / length as u64;
                println!(
                    "Progress: {}%, {} seconds elapsed",
                    progress_percentage,
                    elapsed
                );
            }
            None => {
                println!(
                    "Progress: {} iterations completed, {} seconds elapsed",
                    self.current,
                    elapsed
                );
            }
        }

        // Ensure output is flushed immediately
        io::stdout().flush().unwrap();
    }

    // fn finish(&self) {
    //     println!(
    //         "Progress: 100%, {} iterations completed in {} seconds",
    //         self.current,
    //         self.start.elapsed().as_secs()
    //     );
    // }

    fn finish_with_message(&self, msg: &str) {
        println!(
            "{}, {} iterations completed in {} seconds",
            msg,
            self.current,
            self.start.elapsed().as_secs()
        );
    }
}

/// Gets `ProgressReporter` that suits scenario, based on `rank` 
/// and whether or not we are writing to a terminal. 
/// `interval` and `verbose` are only relevant in the latter scenario 
/// (e.g. when writing output to a log file).
pub fn get_progress_reporter(
    length: Option<usize>, 
    rank: i32, 
    message: &str,
    interval: u64, 
    verbose: bool
) -> ProgressReporter {

    if rank != 0 {
        return ProgressReporter::None
    }

    // Determine if stdout is a terminal
    let is_tty = match env::var("OMPI_COMM_WORLD_SIZE") {
        // Hacky way to ensure that when ran with `mpirun` we 
        // automatically assume we are sending stdout to a file
        Ok(_) => false, 
        Err(_) => atty::is(Stream::Stdout)
    };

    let msg = String::from(message);

    // Match on the combination of `total` and `is_tty`
    match (length, is_tty) {
        (Some(len), true) => {
            let pb = ProgressBar::new(len as u64);
            pb.set_draw_target(ProgressDrawTarget::stdout_with_hz(3));
            pb.set_style(
                ProgressStyle::with_template(
                    "{msg:20} [{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} ({eta})",
                )
                .unwrap()
            );
            pb.set_message(msg);
            ProgressReporter::Bar(pb)
        }
        (None, true) => {
            let ps = ProgressBar::new_spinner();
            ps.set_draw_target(ProgressDrawTarget::stdout_with_hz(3));
            ps.set_style(
                ProgressStyle::with_template(
                    "{msg:20} [{elapsed_precise}] {spinner} {pos:>7}",
                )
                .unwrap()
                .tick_chars("🌖🌗🌘🌑🌒🌓🌔🌕"),
            );
            ps.set_message(msg);
            ProgressReporter::Spinner(ps)
        }
        (length, false) => {
            println!("{}",message);
            let pp = ProgressPrinter::new(length, interval, verbose);
            ProgressReporter::Printer(pp)
        }
    }
}