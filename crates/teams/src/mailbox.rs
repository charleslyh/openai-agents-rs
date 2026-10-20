//! Mailboxes: one bounded inbox per member.
//!
//! [`Mailbox`] is a trait so a persistent backend (a file or a database) can replace
//! [`InMemoryMailbox`] later without touching the team code. Broadcast is not part of the trait: the
//! team expands it into one `send` per recipient, which keeps delivery accounting in one place.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use async_trait::async_trait;
use tokio::sync::Notify;

use crate::error::TeamError;
use crate::types::{MemberId, Message};

/// Default capacity of one inbox.
pub const DEFAULT_INBOX_CAPACITY: usize = 64;

/// Storage and delivery of messages between members.
///
/// There is one receiver per member (its own loop), so `recv` does not have to support several
/// concurrent callers for the same member.
#[async_trait]
pub trait Mailbox: Send + Sync {
    /// Create the inbox of `who`. Registering twice keeps the existing inbox.
    async fn register(&self, who: &MemberId);

    /// Close the inbox: `send` fails from now on and `recv` returns `None` once it is empty.
    async fn close(&self, who: &MemberId);

    /// Put `message` in the inbox of `message.to`.
    ///
    /// Fails with [`TeamError::UnknownMember`] when there is no open inbox and with
    /// [`TeamError::MailboxFull`] when it is at capacity.
    async fn send(&self, message: Message) -> Result<(), TeamError>;

    /// Wait for the next message of `who`. `None` means the inbox is closed (or never existed).
    ///
    /// Cancel-safe: dropping the future loses nothing.
    async fn recv(&self, who: &MemberId) -> Option<Message>;

    /// Take every message that is waiting, oldest first, without waiting.
    async fn drain(&self, who: &MemberId) -> Vec<Message>;

    /// How many messages are waiting.
    async fn pending(&self, who: &MemberId) -> usize;
}

struct Inbox {
    queue: Mutex<VecDeque<Message>>,
    notify: Notify,
    closed: AtomicBool,
}

impl Inbox {
    fn new() -> Self {
        Self {
            queue: Mutex::new(VecDeque::new()),
            notify: Notify::new(),
            closed: AtomicBool::new(false),
        }
    }

    /// A poisoned queue is still a valid queue: every update is a single push or pop.
    fn queue(&self) -> MutexGuard<'_, VecDeque<Message>> {
        self.queue.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// Process-local mailbox. Critical sections only touch memory and never span an `.await`.
pub struct InMemoryMailbox {
    capacity: usize,
    inboxes: Mutex<HashMap<MemberId, Arc<Inbox>>>,
}

impl InMemoryMailbox {
    /// A mailbox whose inboxes hold [`DEFAULT_INBOX_CAPACITY`] messages.
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_INBOX_CAPACITY)
    }

    /// A mailbox whose inboxes hold `capacity` messages (at least 1).
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            inboxes: Mutex::new(HashMap::new()),
        }
    }

    fn inbox(&self, who: &MemberId) -> Option<Arc<Inbox>> {
        self.inboxes
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(who)
            .cloned()
    }
}

impl Default for InMemoryMailbox {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Mailbox for InMemoryMailbox {
    async fn register(&self, who: &MemberId) {
        self.inboxes
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .entry(who.clone())
            .or_insert_with(|| Arc::new(Inbox::new()));
    }

    async fn close(&self, who: &MemberId) {
        if let Some(inbox) = self.inbox(who) {
            inbox.closed.store(true, Ordering::SeqCst);
            inbox.notify.notify_one();
        }
    }

    async fn send(&self, message: Message) -> Result<(), TeamError> {
        let inbox = self
            .inbox(&message.to)
            .ok_or_else(|| TeamError::UnknownMember(message.to.clone()))?;
        if inbox.closed.load(Ordering::SeqCst) {
            return Err(TeamError::UnknownMember(message.to));
        }
        {
            let mut queue = inbox.queue();
            if queue.len() >= self.capacity {
                return Err(TeamError::MailboxFull(message.to));
            }
            queue.push_back(message);
        }
        inbox.notify.notify_one();
        Ok(())
    }

    async fn recv(&self, who: &MemberId) -> Option<Message> {
        let inbox = self.inbox(who)?;
        loop {
            // Register interest before looking, so a send between the look and the wait is not lost.
            let notified = inbox.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if let Some(message) = inbox.queue().pop_front() {
                return Some(message);
            }
            if inbox.closed.load(Ordering::SeqCst) {
                return None;
            }
            notified.await;
        }
    }

    async fn drain(&self, who: &MemberId) -> Vec<Message> {
        match self.inbox(who) {
            Some(inbox) => inbox.queue().drain(..).collect(),
            None => Vec::new(),
        }
    }

    async fn pending(&self, who: &MemberId) -> usize {
        self.inbox(who).map_or(0, |inbox| inbox.queue().len())
    }
}
