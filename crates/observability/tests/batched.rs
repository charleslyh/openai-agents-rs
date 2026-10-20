//! The exporter must cost the run nothing, and must not lose events in the normal case.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, SystemTime};

use opai_agents_core::tracing::{Span, SpanData, TracingProcessor};
use opai_agents_observability::{BatchedProcessor, ConsoleSink, EventSink, ExportEvent, JsonlSink};

/// Records what it was given, so a test can wait for it.
struct RecordingSink {
    events: Arc<Mutex<Vec<ExportEvent>>>,
    fail: bool,
}

impl EventSink for RecordingSink {
    fn write(&mut self, events: &[ExportEvent]) -> Result<(), String> {
        if self.fail {
            return Err("the sink is broken".into());
        }
        self.events
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .extend(events.iter().cloned());
        Ok(())
    }

    fn name(&self) -> &'static str {
        "recording"
    }
}

fn span(id: &str) -> Span {
    Span {
        span_id: id.to_string(),
        parent_id: None,
        data: SpanData::Function {
            name: "search".into(),
        },
        started_at: SystemTime::UNIX_EPOCH,
        ended_at: Some(SystemTime::UNIX_EPOCH),
        error: None,
    }
}

fn wait_for(events: &Arc<Mutex<Vec<ExportEvent>>>, count: usize) -> bool {
    for _ in 0..200 {
        if events
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
            >= count
        {
            return true;
        }
        thread::sleep(Duration::from_millis(10));
    }
    false
}

#[test]
fn every_event_reaches_the_sink() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let processor = BatchedProcessor::spawn(
        RecordingSink {
            events: Arc::clone(&events),
            fail: false,
        },
        64,
        Duration::from_millis(20),
    );
    for i in 0..10 {
        processor.on_span_start(&span(&format!("start-{i}")));
        processor.on_span_end(&span(&format!("end-{i}")));
    }
    assert!(wait_for(&events, 20), "all events arrived");
    assert_eq!(processor.dropped(), 0);
    let events = events
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    assert!(matches!(events[0], ExportEvent::SpanStart(_)));
    assert!(matches!(events[1], ExportEvent::SpanEnd(_)));
    let json = events[1].to_json();
    assert_eq!(json["kind"], "span_end");
    assert_eq!(json["span_kind"], "function");
    assert_eq!(json["name"], "search");
    assert_eq!(json["span_id"], "end-0");
}

/// A full queue drops events and counts them; it never waits.
#[test]
fn a_full_queue_drops_instead_of_blocking() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let processor = BatchedProcessor::spawn(
        RecordingSink {
            events: Arc::clone(&events),
            fail: false,
        },
        2,
        Duration::from_secs(60),
    );
    let started = SystemTime::now();
    for i in 0..5_000 {
        processor.on_span_end(&span(&format!("s-{i}")));
    }
    let elapsed = started.elapsed().unwrap_or_default();
    assert!(
        elapsed < Duration::from_secs(5),
        "5 000 spans took {elapsed:?}: the callback is blocking"
    );
    assert!(processor.dropped() > 0, "a queue of 2 cannot hold 5 000");
}

/// `force_flush` waits for what is queued.
#[test]
fn force_flush_writes_what_is_queued() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let processor = BatchedProcessor::spawn(
        RecordingSink {
            events: Arc::clone(&events),
            fail: false,
        },
        64,
        Duration::from_secs(60),
    );
    processor.on_span_end(&span("one"));
    processor.force_flush();
    assert_eq!(
        events
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len(),
        1
    );
}

/// A sink that fails is reported, and its events are counted as lost rather than retried forever.
#[test]
fn a_failing_sink_is_reported_and_counted() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let processor = BatchedProcessor::spawn(
        RecordingSink {
            events: Arc::clone(&events),
            fail: true,
        },
        64,
        Duration::from_millis(10),
    );
    processor.on_span_end(&span("one"));
    processor.force_flush();
    assert!(
        processor.dropped() >= 1,
        "the lost event is counted: {}",
        processor.dropped()
    );
}

/// The file sink writes one JSON object per line.
#[test]
fn the_jsonl_sink_writes_one_line_per_event() {
    let path = std::env::temp_dir().join(format!(
        "agents-observability-{}.jsonl",
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default()
    ));
    let processor = BatchedProcessor::spawn(
        JsonlSink::open(&path).expect("open"),
        64,
        Duration::from_millis(10),
    );
    processor.on_span_end(&span("one"));
    processor.on_span_end(&span("two"));
    processor.force_flush();

    let written = std::fs::read_to_string(&path).expect("read");
    let lines: Vec<&str> = written.lines().collect();
    assert_eq!(lines.len(), 2, "{written}");
    for line in lines {
        let value: serde_json::Value = serde_json::from_str(line).expect("a JSON line");
        assert_eq!(value["kind"], "span_end");
    }
    std::fs::remove_file(&path).ok();
}

/// A bad path is an error at construction, not silence later.
#[test]
fn a_jsonl_sink_reports_a_path_it_cannot_use() {
    let error = JsonlSink::open("/definitely/not/a/directory/agents.jsonl")
        .expect_err("the path is unusable");
    assert!(error.contains("cannot"), "{error}");
    // The console sink exists for a terminal or a container log.
    let _ = ConsoleSink;
}

/// A sink used from many threads at once is fine: the counts add up.
#[test]
fn concurrent_spans_are_all_counted() {
    static SEEN: AtomicU64 = AtomicU64::new(0);
    struct Counting;
    impl EventSink for Counting {
        fn write(&mut self, events: &[ExportEvent]) -> Result<(), String> {
            SEEN.fetch_add(events.len() as u64, Ordering::Relaxed);
            Ok(())
        }
    }
    let processor = Arc::new(BatchedProcessor::spawn(
        Counting,
        1024,
        Duration::from_millis(10),
    ));
    let mut tasks = Vec::new();
    for t in 0..8 {
        let processor = Arc::clone(&processor);
        tasks.push(thread::spawn(move || {
            for i in 0..25 {
                processor.on_span_end(&span(&format!("{t}-{i}")));
            }
        }));
    }
    for task in tasks {
        task.join().expect("finished");
    }
    processor.force_flush();
    assert_eq!(SEEN.load(Ordering::Relaxed), 200);
}
