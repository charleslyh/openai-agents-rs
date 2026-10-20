//! The shared task list.
//!
//! A task is owned by at most one member. Claiming is a single atomic step: status, dependencies
//! and owner are checked and the owner is written under one lock, so two members racing for the same
//! task cannot both win. Dependencies are evaluated when asked (a task is unblocked as soon as its
//! prerequisites are `Completed`), so completing a task needs no extra bookkeeping.

use std::sync::{Mutex, MutexGuard};

use async_trait::async_trait;

use crate::error::TeamError;
use crate::types::{ClaimOutcome, MemberId, NewTask, Task, TaskId, TaskStatus, TaskUpdate};

/// Storage of the team's tasks.
#[async_trait]
pub trait TaskBoard: Send + Sync {
    /// Add a task. Every id in `blocked_by` must already exist, which also rules out cycles.
    async fn create(&self, new: NewTask) -> Result<Task, TeamError>;

    /// One task.
    async fn get(&self, id: &TaskId) -> Result<Option<Task>, TeamError>;

    /// Try to take a pending task for `who`. Atomic.
    async fn claim(&self, id: &TaskId, who: &MemberId) -> Result<ClaimOutcome, TeamError>;

    /// Change a task. Only its owner may; see [`TaskUpdate`] for the allowed transitions.
    async fn update(
        &self,
        id: &TaskId,
        who: &MemberId,
        update: TaskUpdate,
    ) -> Result<Task, TeamError>;

    /// Give every unfinished task of `who` back (used when a member fails or leaves).
    async fn release_owned_by(&self, who: &MemberId) -> Result<Vec<TaskId>, TeamError>;

    /// All tasks in creation order.
    async fn list(&self) -> Result<Vec<Task>, TeamError>;
}

#[derive(Default)]
struct Inner {
    tasks: Vec<Task>,
    next: u64,
}

impl Inner {
    fn find(&self, id: &TaskId) -> Option<usize> {
        self.tasks.iter().position(|t| &t.id == id)
    }

    fn unfinished_deps(&self, task: &Task) -> Vec<TaskId> {
        task.blocked_by
            .iter()
            .filter(|dep| {
                self.find(dep)
                    .map(|i| self.tasks[i].status != TaskStatus::Completed)
                    .unwrap_or(true)
            })
            .cloned()
            .collect()
    }
}

/// Process-local task board. Critical sections only touch memory and never span an `.await`.
#[derive(Default)]
pub struct InMemoryTaskBoard {
    inner: Mutex<Inner>,
}

impl InMemoryTaskBoard {
    /// An empty board.
    pub fn new() -> Self {
        Self::default()
    }

    /// A poisoned board is still consistent: each operation validates first and mutates last.
    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }
}

#[async_trait]
impl TaskBoard for InMemoryTaskBoard {
    async fn create(&self, new: NewTask) -> Result<Task, TeamError> {
        let mut inner = self.lock();
        for dep in &new.blocked_by {
            if inner.find(dep).is_none() {
                return Err(TeamError::UnknownTask(dep.clone()));
            }
        }
        inner.next += 1;
        let task = Task {
            id: TaskId::new(format!("t{}", inner.next)),
            title: new.title,
            description: new.description,
            status: TaskStatus::Pending,
            owner: None,
            blocked_by: dedup(new.blocked_by),
            result: None,
            created_by: new.created_by,
        };
        inner.tasks.push(task.clone());
        Ok(task)
    }

    async fn get(&self, id: &TaskId) -> Result<Option<Task>, TeamError> {
        let inner = self.lock();
        Ok(inner.find(id).map(|i| inner.tasks[i].clone()))
    }

    async fn claim(&self, id: &TaskId, who: &MemberId) -> Result<ClaimOutcome, TeamError> {
        let mut inner = self.lock();
        let Some(index) = inner.find(id) else {
            return Ok(ClaimOutcome::NotFound);
        };
        let task = &inner.tasks[index];
        match task.status {
            TaskStatus::Completed => return Ok(ClaimOutcome::AlreadyCompleted),
            TaskStatus::InProgress => {
                return Ok(ClaimOutcome::AlreadyTaken {
                    owner: task.owner.clone(),
                })
            }
            TaskStatus::Pending => {}
        }
        let waiting_on = inner.unfinished_deps(task);
        if !waiting_on.is_empty() {
            return Ok(ClaimOutcome::Blocked { waiting_on });
        }
        let task = &mut inner.tasks[index];
        task.status = TaskStatus::InProgress;
        task.owner = Some(who.clone());
        Ok(ClaimOutcome::Claimed { task: task.clone() })
    }

    async fn update(
        &self,
        id: &TaskId,
        who: &MemberId,
        update: TaskUpdate,
    ) -> Result<Task, TeamError> {
        let mut inner = self.lock();
        let index = inner
            .find(id)
            .ok_or_else(|| TeamError::UnknownTask(id.clone()))?;
        let task = &mut inner.tasks[index];
        if task.owner.as_ref() != Some(who) {
            return Err(TeamError::NotOwner {
                task: id.clone(),
                caller: who.clone(),
            });
        }
        if task.status == TaskStatus::Completed {
            return Err(TeamError::InvalidTransition(format!(
                "task `{id}` is already completed"
            )));
        }
        match update.status {
            Some(TaskStatus::Completed) => {
                task.status = TaskStatus::Completed;
                task.result = update.result;
            }
            Some(TaskStatus::Pending) => {
                task.status = TaskStatus::Pending;
                task.owner = None;
            }
            Some(TaskStatus::InProgress) | None => {
                if update.result.is_some() {
                    task.result = update.result;
                }
            }
        }
        Ok(task.clone())
    }

    async fn release_owned_by(&self, who: &MemberId) -> Result<Vec<TaskId>, TeamError> {
        let mut inner = self.lock();
        let mut released = Vec::new();
        for task in &mut inner.tasks {
            if task.status == TaskStatus::InProgress && task.owner.as_ref() == Some(who) {
                task.status = TaskStatus::Pending;
                task.owner = None;
                released.push(task.id.clone());
            }
        }
        Ok(released)
    }

    async fn list(&self) -> Result<Vec<Task>, TeamError> {
        Ok(self.lock().tasks.clone())
    }
}

fn dedup(mut ids: Vec<TaskId>) -> Vec<TaskId> {
    let mut seen = std::collections::HashSet::new();
    ids.retain(|id| seen.insert(id.clone()));
    ids
}
